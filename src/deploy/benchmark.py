"""Benchmark the PM Copilot inference server — measures latency, throughput, and token rates."""

import argparse
import json
import time
import requests
import statistics

PROMPTS = [
    "How would you prioritize features for a new B2B SaaS analytics product?",
    "Write a PRD for a mobile food delivery app targeting college students.",
    "Should we build or buy an analytics solution for our e-commerce platform?",
    "How would you improve user retention for a fitness app?",
    "Design a referral program for a FinTech lending platform.",
    "How would you measure success for a new social feature in a productivity app?",
    "What metrics would you track to evaluate a marketplace's health?",
    "How should a HealthTech startup approach HIPAA compliance in their product roadmap?",
    "Design an onboarding flow for a complex enterprise SaaS tool.",
    "How would you handle feature cannibalization between two product lines?",
]


def run_benchmark(base_url: str, n_requests: int, max_tokens: int, use_rag: bool):
    print(f"Benchmarking {base_url}/generate")
    print(f"Requests: {n_requests} | Max tokens: {max_tokens} | RAG: {use_rag}\n")

    latencies = []
    tps_list = []
    tokens_list = []

    for i in range(n_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "use_rag": use_rag,
        }

        start = time.perf_counter()
        resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
        wall_time = time.perf_counter() - start

        if resp.status_code != 200:
            print(f"  [{i+1}] FAILED: {resp.status_code} {resp.text[:100]}")
            continue

        data = resp.json()
        latencies.append(data["latency_seconds"])
        tps_list.append(data["tokens_per_second"])
        tokens_list.append(data["tokens_generated"])

        print(
            f"  [{i+1}/{n_requests}] {data['tokens_generated']} tokens | "
            f"{data['latency_seconds']:.2f}s | "
            f"{data['tokens_per_second']:.1f} tok/s | "
            f"{prompt[:50]}..."
        )

    if not latencies:
        print("No successful requests.")
        return

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({n_requests} requests)")
    print(f"{'='*60}")
    print(f"Latency (seconds):")
    print(f"  Mean:   {statistics.mean(latencies):.3f}")
    print(f"  Median: {statistics.median(latencies):.3f}")
    print(f"  P95:    {sorted(latencies)[int(len(latencies)*0.95)]:.3f}")
    print(f"  Min:    {min(latencies):.3f}")
    print(f"  Max:    {max(latencies):.3f}")
    print(f"\nThroughput:")
    print(f"  Mean tokens/sec:   {statistics.mean(tps_list):.1f}")
    print(f"  Median tokens/sec: {statistics.median(tps_list):.1f}")
    print(f"  Total tokens:      {sum(tokens_list)}")
    print(f"  Total time:        {sum(latencies):.1f}s")
    print(f"\nPagedAttention: enabled")
    print(f"RAG augmented:  {use_rag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rag", action="store_true")
    args = parser.parse_args()

    run_benchmark(args.url, args.n, args.max_tokens, args.rag)
