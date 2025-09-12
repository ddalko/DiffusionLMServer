# server.py
import os, time, torch
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel

# ---------------------
# Model bootstrap
# ---------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/models/LLaDA-8B-Instruct")
DTYPE = torch.bfloat16 if os.getenv("TORCH_DTYPE", "bf16") == "bf16" else torch.float16
GEN_LENGTH = int(os.getenv("GEN_LENGTH", "128"))
STEPS = int(os.getenv("STEPS", "128"))
BLOCK_LENGTH = int(os.getenv("BLOCK_LENGTH", "32"))

DEVICE = os.getenv("DEVICE", "cuda")

print(f"Loading {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=DTYPE
).to(DEVICE).eval()
print("Model loaded.")

# ---------------------
# OpenAI-like schemas
# ---------------------
class Message(BaseModel):
    role: str
    content: str

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSONSchema

class ToolDef(BaseModel):
    type: str  # "function"
    function: ToolFunction

class ChatCompletionReq(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = None  # "auto" | {"type":"function","function":{"name":...}}

# ---------------------
# Helper: chat template → input_ids
# ---------------------
def build_input_ids(messages: List[Dict[str, str]]) -> torch.Tensor:
    templ = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(templ, return_tensors="pt")["input_ids"]

def decode_new_tokens(output_ids: torch.Tensor, prompt_len: int) -> str:
    cpu_ids = output_ids.to("cpu", non_blocking=True).contiguous()
    gen_ids = cpu_ids[0, prompt_len:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "llada-8b-instruct", "object": "model"}, {"id": "llada-8b-base", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionReq, request: Request):
    gen_length = GEN_LENGTH
    steps = STEPS
    block_length = BLOCK_LENGTH
    mask_id = 126336 

    msgs = [m.dict() for m in req.messages]
    input_ids_cpu = build_input_ids(msgs)

    stop_event = asyncio.Event()
    async def watch_disconnect():
        while not stop_event.is_set():
            if await request.is_disconnected():
                stop_event.set()
                break
            await asyncio.sleep(0.1)
    watcher = asyncio.create_task(watch_disconnect())

    gen_ids = None
    input_ids = None
    try:
        input_ids = input_ids_cpu.to(DEVICE, non_blocking=True)
        prompt_len = input_ids.shape[1]

        if gen_length % block_length != 0:
            raise HTTPException(status_code=400, detail="gen_length must be multiple of block_length")
        num_blocks = gen_length // block_length

        if steps % num_blocks != 0:
            raise HTTPException(status_code=400, detail="steps must be multiple of gen_length // block_length")
        steps_per_block = steps // num_blocks
    
        def build_transfer_schedule(block_mask_index: torch.Tensor, steps_per_block: int) -> torch.Tensor:
            m = block_mask_index.sum(dim=1)
            sched = []
            prev = torch.zeros_like(m)
            for i in range(steps_per_block):
                cur = torch.ceil(m.float() * (i + 1) / steps_per_block).to(torch.long)
                sched.append((cur - prev).clamp_min(0))
                prev = cur
            return torch.stack(sched, dim=1)
        
        @torch.inference_mode()
        def step_once(x: torch.Tensor):
            logits = model(x).logits
            x0 = logits.argmax(dim=-1)
            logZ = logits.logsumexp(dim=-1)
            chosen = logits.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p = (chosen - logZ).exp()
            del logits, logZ, chosen
            return x0, x0_p
        
        x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=DEVICE)
        x[:, :prompt_len] = input_ids

        try:
            with torch.inference_mode():
                for nb in range(num_blocks):
                    start = prompt_len + nb * block_length
                    end   = start + block_length

                    block_mask_index = (x[:, start:end] == mask_id)
                    num_transfer_tokens = build_transfer_schedule(block_mask_index, steps_per_block)

                    for i in range(steps_per_block):
                        if stop_event.is_set():
                            raise HTTPException(status_code=499, detail="Client Closed Request")

                        mask_index = (x == mask_id)
                        x0, x0_p = step_once(x)

                        x0_p[:, end:] = float('-inf')

                        neg_inf = torch.full_like(x0_p, float('-inf'))
                        confidence = torch.where(mask_index, x0_p, neg_inf)

                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
                        k = int(num_transfer_tokens[0, i].item())
                        if k > 0:
                            _, idx = torch.topk(confidence[0], k=k, largest=True, sorted=False)
                            transfer_index[0, idx] = True

                        x = torch.where(transfer_index, x0, x)
                        del x0, x0_p, confidence, mask_index, transfer_index
                gen_ids = x
        except torch.cuda.OutOfMemoryError:
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="GPU OOM during generation. Try shorter prompt or gen_length.")

        if stop_event.is_set():
            raise HTTPException(status_code=499, detail="Client Closed Request")

        text = decode_new_tokens(gen_ids, prompt_len)

        message: Dict[str, Any] = {"role": "assistant", "content": text}
        resp = {
            "id": f"chatcmpl_{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if "tool_calls" in message else "stop"
            }]
        }
        with open(f"json_results/{resp['id']}.json", "w") as f:
            import json
            tmp_resp = resp.copy()
            tmp_resp["request_messages"] = req.dict().get("messages", [])
            json.dump(tmp_resp, f, indent=2)
        return JSONResponse(resp)

    except asyncio.CancelledError:
        stop_event.set()
        raise

    finally:
        stop_event.set()
        watcher.cancel()
        if gen_ids is not None:
            del gen_ids
        if input_ids_cpu is not None:
            del input_ids_cpu
        if 'input_ids' in locals():
            del input_ids
        
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
