# server.py
import os, time, json, torch
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer, AutoModel

from generate import generate

# ---------------------
# Model bootstrap
# ---------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/models/LLaDA-8B-Instruct")
DTYPE = torch.bfloat16 if os.getenv("TORCH_DTYPE", "bf16") == "bf16" else torch.float16
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
    # LLaDA는 chat template을 제공하므로, 그대로 사용
    templ = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(templ, return_tensors="pt")["input_ids"].to(DEVICE)
    return ids

def decode_new_tokens(output_ids: torch.Tensor, prompt_len: int) -> str:
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

# ---------------------
# Tool-calling 래핑(옵션)
# ---------------------
def text_to_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    모델이 JSON을 냈다고 가정하고 파싱하여 OpenAI tool_calls 포맷으로 변환.
    JSON이 아니면 None을 리턴해 일반 content로 내려보냅니다.
    """
    t = text.strip().strip("`")
    if t.lower().startswith("json"):
        t = t[4:].strip()
    try:
        obj = json.loads(t)
        action = obj.get("action", "none")
        args = obj.get("args", {})
        return [{
            "id": f"call_{int(time.time()*1000)}",
            "type": "function",
            "function": {
                "name": action,
                "arguments": json.dumps(args, ensure_ascii=False)
            }
        }]
    except Exception:
        return None

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "llada-8b-instruct-transformers", "object": "model"}]}

@app.post("/v1/chat/completions")
def chat(req: ChatCompletionReq):
    gen_length = 256
    steps = 256

    # 1) messages → input_ids
    msgs = [m.dict() for m in req.messages]

    if req.tools:
        msgs = [{"role":"system","content":
                 "You must answer with ONE valid JSON object. No extra text."}] + msgs

    input_ids = build_input_ids(msgs)
    prompt_len = input_ids.shape[1]

    # 2) generate
    with torch.no_grad():
        gen_ids = generate(model, input_ids, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

    text = decode_new_tokens(gen_ids, prompt_len)

    # 3) 스트리밍 모드 (단순 content 스트리밍만 지원; tool_calls는 완성본만 권장)
    if req.stream and not req.tools:
        def event_stream():
            # OpenAI SSE 형식 흉내
            chunk_id = int(time.time()*1000)
            for ch in text:
                delta = {"id": f"chatcmpl_{chunk_id}",
                         "object": "chat.completion.chunk",
                         "choices": [{"index":0, "delta":{"content": ch}, "finish_reason": None}]}
                yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
            done = {"id": f"chatcmpl_{chunk_id}",
                    "object": "chat.completion.chunk",
                    "choices": [{"index":0, "delta":{}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 4) tool_calls 또는 일반 content로 래핑
    message: Dict[str, Any] = {"role": "assistant"}
    if req.tools:
        tool_calls = text_to_tool_calls(text)
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            # JSON 파싱 실패 시, 안전하게 일반 content로 반환
            message["content"] = text
    else:
        message["content"] = text

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
