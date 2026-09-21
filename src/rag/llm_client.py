"""Unified LLM client for self-query parsing, query expansion, and metadata tagging.

Supports multiple backends so you can swap between free APIs during dev
and your fine-tuned model in production.
"""

import json
import os

BACKEND = os.getenv("LLM_BACKEND", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def llm_call(prompt: str, max_tokens: int = 512) -> str:
    if BACKEND == "groq":
        return _groq_inference(prompt, max_tokens)
    elif BACKEND == "local":
        return _local_inference(prompt, max_tokens)
    elif BACKEND == "sagemaker":
        return _sagemaker_inference(prompt, max_tokens)
    else:
        raise ValueError(f"Unknown LLM backend: {BACKEND}")


def _groq_inference(prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def _local_inference(prompt: str, max_tokens: int) -> str:
    import requests

    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8000")
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()


def _sagemaker_inference(prompt: str, max_tokens: int) -> str:
    import boto3

    client = boto3.client("sagemaker-runtime")
    endpoint_name = os.getenv("SAGEMAKER_ENDPOINT", "pm-copilot-endpoint")
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.1},
    })
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode())
    return result[0]["generated_text"].strip()
