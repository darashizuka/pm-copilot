"""vLLM inference server with PagedAttention — run inside SageMaker Studio or Lightning AI."""

import argparse
import os
import time
import json

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PM Copilot API")

MERGED_MODEL_PATH = os.getenv("MODEL_PATH", "./outputs/merged")

SYSTEM_PROMPT = (
    "You are an expert AI Product Manager. Structure all responses with: "
    "Problem Definition, User Persona, Prioritized Solutions, Trade-offs, "
    "Primary Metrics & Counter-Metrics."
)

engine = None
tokenizer = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    use_rag: bool = False


class InferenceResponse(BaseModel):
    response: str
    tokens_generated: int
    latency_seconds: float
    tokens_per_second: float


def load_model():
    global engine, tokenizer
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model from {MERGED_MODEL_PATH}...")
    engine = LLM(
        model=MERGED_MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded with PagedAttention enabled.")


def get_rag_context(query: str) -> str:
    try:
        qdrant_url = os.getenv("QDRANT_URL", "")
        if not qdrant_url:
            return ""
        from src.rag.vector_store import query as vector_query
        results = vector_query(query_text=query, n_results=3)
        if results and results["documents"] and results["documents"][0]:
            return "\n\n".join(results["documents"][0])
    except Exception as e:
        print(f"RAG lookup failed: {e}")
    return ""


@app.on_event("startup")
def startup():
    load_model()


@app.post("/generate", response_model=InferenceResponse)
def generate(req: InferenceRequest):
    from vllm import SamplingParams

    user_content = req.prompt
    if req.use_rag:
        context = get_rag_context(req.prompt)
        if context:
            user_content = f"Use the following context to inform your answer:\n\n{context}\n\nQuestion: {req.prompt}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    start = time.perf_counter()
    outputs = engine.generate([formatted], sampling_params)
    elapsed = time.perf_counter() - start

    generated_text = outputs[0].outputs[0].text
    num_tokens = len(outputs[0].outputs[0].token_ids)
    tps = num_tokens / elapsed if elapsed > 0 else 0

    return InferenceResponse(
        response=generated_text,
        tokens_generated=num_tokens,
        latency_seconds=round(elapsed, 3),
        tokens_per_second=round(tps, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MERGED_MODEL_PATH, "paged_attention": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
