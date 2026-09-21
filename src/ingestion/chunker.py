"""Chunk raw documents from HuggingFace into ~500-token sections with metadata tagging.
Pushes chunked dataset back to HuggingFace."""

import json
import os
import uuid

from datasets import Dataset, load_dataset

HF_RAW_DATASET = os.getenv("HF_DATASET_NAME", "shizukadara2/pm-copilot-raw")
HF_CHUNKS_DATASET = os.getenv("HF_CHUNKS_DATASET", "shizukadara2/pm-copilot-chunks")
HF_TOKEN = os.getenv("HF_TOKEN", "")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

FRAMEWORK_SIGNALS = {
    "CIRCLES": ["circles framework", "comprehension", "identify customer", "list solutions", "evaluate tradeoffs"],
    "RICE": ["rice score", "rice framework", "reach impact confidence effort", "prioritization score"],
    "Jobs-to-be-Done": ["jobs to be done", "jtbd", "job story", "when i", "so i can", "hiring a product"],
    "Product Design": ["user flow", "wireframe", "mockup", "usability", "ux design", "information architecture"],
    "Metrics": ["north star metric", "counter-metric", "guardrail metric", "leading indicator", "lagging indicator", "conversion rate", "retention rate"],
}

DOMAIN_SIGNALS = {
    "B2B SaaS": ["saas", "enterprise", "b2b", "annual contract", "arr", "mrr", "seat-based", "self-serve"],
    "Marketplace": ["marketplace", "two-sided", "supply and demand", "gmv", "take rate", "liquidity"],
    "FinTech": ["fintech", "payments", "banking", "lending", "credit score", "transaction"],
    "Consumer": ["consumer app", "social media", "engagement", "dau", "mau", "viral", "notification"],
    "E-commerce": ["e-commerce", "ecommerce", "cart", "checkout", "catalog", "sku", "fulfillment"],
}


def detect_framework(text: str) -> str:
    text_lower = text.lower()
    best, best_score = "General", 0
    for framework, keywords in FRAMEWORK_SIGNALS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best, best_score = framework, score
    return best


def detect_domain(text: str) -> str:
    text_lower = text.lower()
    best, best_score = "General", 0
    for domain, keywords in DOMAIN_SIGNALS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best, best_score = domain, score
    return best


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def process_all(push_to_hub: bool = True, dry_run: bool = False):
    print(f"Loading raw dataset from {HF_RAW_DATASET}...")
    raw_ds = load_dataset(HF_RAW_DATASET, split="train")
    print(f"Loaded {len(raw_ds)} documents\n")

    all_chunks = []
    for idx, doc in enumerate(raw_ds):
        text = doc.get("text", "")
        if not text or len(text.split()) < 20:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "doc_type": doc.get("doc_type", "Strategy Note"),
                "framework": detect_framework(chunk),
                "domain": detect_domain(chunk),
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(raw_ds)} documents ({len(all_chunks)} chunks)")

    print(f"\nCreated {len(all_chunks)} chunks from {len(raw_ds)} documents")

    if dry_run:
        print("Dry run — not uploading.")
        return all_chunks

    if not all_chunks:
        print("No chunks to upload.")
        return all_chunks

    chunks_ds = Dataset.from_list(all_chunks)

    if push_to_hub:
        chunks_ds.push_to_hub(HF_CHUNKS_DATASET, token=HF_TOKEN)
        print(f"Pushed {len(all_chunks)} chunks to {HF_CHUNKS_DATASET}")
    else:
        from pathlib import Path
        output_dir = Path.cwd() / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_ds.save_to_disk(str(output_dir / "chunks"))
        print(f"Saved {len(all_chunks)} chunks locally")

    return all_chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Save locally instead of HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Test without uploading")
    args = parser.parse_args()
    process_all(push_to_hub=not args.local, dry_run=args.dry_run)
