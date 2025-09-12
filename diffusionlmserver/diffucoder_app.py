# server.py
import os, time, torch
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/models/DiffuCoder-7B-cpGRPO" 

print(f"Loading {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda").eval()
print("Model loaded.")

TOKEN_PER_STEP = 1 # diffusion timesteps * TOKEN_PER_STEP = total new tokens

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

def build_inputs(messages: List[Dict[str, str]]):
    template = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = template.input_ids.to(device="cuda")
    attention_mask = template.attention_mask.to(device="cuda")
    return input_ids, attention_mask

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "DiffuCoder-7B-cpGRPO", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionReq, request: Request):
    msgs = [m.dict() for m in req.messages]

    stop_event = asyncio.Event()
    async def watch_disconnect():
        while not stop_event.is_set():
            if await request.is_disconnected():
                stop_event.set()
                break
            await asyncio.sleep(0.1)
    watcher = asyncio.create_task(watch_disconnect())

    input_ids = None
    try:
        input_ids, attention_mask = build_inputs(msgs)

        try:
            with torch.inference_mode():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=512,
                    temperature=0.2,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                )
                generations = [
                    tokenizer.decode(g[len(p) :].tolist())
                    for p, g in zip(input_ids, output.sequences)
                ]
        except torch.cuda.OutOfMemoryError:
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="GPU OOM during generation. Try shorter prompt or gen_length.")

        if stop_event.is_set():
            raise HTTPException(status_code=499, detail="Client Closed Request")

        text = generations[0].split('<|dlm_pad|>')[0]
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
        return JSONResponse(resp)

    except asyncio.CancelledError:
        stop_event.set()
        raise

    finally:
        stop_event.set()
        watcher.cancel()
        if 'input_ids' in locals():
            del input_ids
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
