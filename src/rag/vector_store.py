"""Qdrant Cloud vector store — load chunks from HuggingFace, embed, and query."""

import os
from typing import Optional

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

HF_CHUNKS_DATASET = os.getenv("HF_CHUNKS_DATASET", "shizukadara2/pm-copilot-chunks")
QDRANT_URL = os.getenv("QDRANT_URL", "")  # e.g. https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pm_knowledge"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def load_from_huggingface(force: bool = False):
    print(f"Loading chunks from {HF_CHUNKS_DATASET}...")
    ds = load_dataset(HF_CHUNKS_DATASET, split="train")
    print(f"Loaded {len(ds)} chunks")

    client = get_client()

    if client.collection_exists(COLLECTION_NAME):
        if force:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        else:
            info = client.get_collection(COLLECTION_NAME)
            print(f"Collection already has {info.points_count} points. Use --force to reload.")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    embedder = get_embedder()
    batch_size = 64

    for i in range(0, len(ds), batch_size):
        batch = ds.select(range(i, min(i + batch_size, len(ds))))
        texts = batch["text"]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id=idx,
                vector=emb,
                payload={
                    "chunk_id": row["id"],
                    "text": row["text"],
                    "source": row["source"],
                    "url": row["url"],
                    "title": row["title"],
                    "doc_type": row["doc_type"],
                    "framework": row["framework"],
                    "domain": row["domain"],
                },
            )
            for idx, (row, emb) in enumerate(zip(batch, embeddings), start=i)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Loaded {min(i + batch_size, len(ds))}/{len(ds)} chunks")

    info = client.get_collection(COLLECTION_NAME)
    print(f"Done. Collection has {info.points_count} points.")


def query(
    query_text: str,
    n_results: int = 5,
    framework: Optional[str] = None,
    domain: Optional[str] = None,
) -> dict:
    client = get_client()
    embedder = get_embedder()

    query_vector = embedder.encode(query_text).tolist()

    conditions = []
    if framework:
        conditions.append(FieldCondition(key="framework", match=MatchValue(value=framework)))
    if domain:
        conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=n_results,
        with_payload=True,
    )

    return {
        "ids": [[p.id for p in results.points]],
        "documents": [[p.payload["text"] for p in results.points]],
        "metadatas": [[{k: v for k, v in p.payload.items() if k != "text"} for p in results.points]],
        "scores": [[p.score for p in results.points]],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Delete and reload collection")
    args = parser.parse_args()
    load_from_huggingface(force=args.force)
