"""Production RAG retriever: LLM-based self-query, query expansion, and cross-encoder reranking."""

import json
from sentence_transformers import CrossEncoder
from .vector_store import query as vector_query
from .llm_client import llm_call

RERANKER_MODEL = "BAAI/bge-reranker-base"
_reranker = None

SELF_QUERY_PROMPT = """Analyze the following user query and extract structured metadata filters.
Return a JSON object with these fields (use null if not detectable):
- "domain": one of ["B2B SaaS", "Marketplace", "FinTech", "Consumer", "E-commerce", "HealthTech", "EdTech", null]
- "framework": one of ["CIRCLES", "RICE", "Jobs-to-be-Done", "Product Design", "Metrics", null]

User query: {query}

Return ONLY valid JSON, no explanation."""

QUERY_EXPANSION_PROMPT = """You are a product management expert. Given the user's question, generate 3 diverse search queries
that would help find relevant case studies, frameworks, and product teardowns to answer it.

User question: {query}

Return a JSON array of 3 strings. No explanation, just the JSON array."""


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def parse_self_query(user_query: str) -> dict:
    try:
        response = llm_call(SELF_QUERY_PROMPT.format(query=user_query))
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            filters = json.loads(response[start:end])
            return {k: v for k, v in filters.items() if v is not None}
    except Exception as e:
        print(f"  Self-query parse failed: {e}")
    return {}


def expand_query(user_query: str) -> list[str]:
    try:
        response = llm_call(QUERY_EXPANSION_PROMPT.format(query=user_query))
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            expansions = json.loads(response[start:end])
            return [user_query] + expansions[:3]
    except Exception as e:
        print(f"  Query expansion failed: {e}")
    return [user_query]


def _search_with_queries(queries: list[str], initial_k: int, framework=None, domain=None) -> list[dict]:
    all_results = []
    seen_ids = set()

    for q in queries:
        results = vector_query(
            query_text=q,
            n_results=initial_k,
            framework=framework,
            domain=domain,
        )
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append({
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                    })
    return all_results


def retrieve_and_rerank(
    user_query: str,
    top_k: int = 3,
    initial_k: int = 15,
) -> list[dict]:
    filters = parse_self_query(user_query)
    expanded_queries = expand_query(user_query)

    print(f"  Filters: {filters}")
    print(f"  Expanded queries: {[q[:50] for q in expanded_queries]}")

    all_results = _search_with_queries(
        expanded_queries, initial_k,
        framework=filters.get("framework"),
        domain=filters.get("domain"),
    )

    if not all_results and filters:
        print("  Filtered search returned 0 results, retrying without filters...")
        all_results = _search_with_queries(expanded_queries, initial_k)

    if not all_results:
        return []

    reranker = get_reranker()
    pairs = [[user_query, r["text"]] for r in all_results]
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        all_results[i]["rerank_score"] = float(score)

    all_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return all_results[:top_k]


def build_augmented_prompt(user_query: str, top_k: int = 3) -> str:
    retrieved = retrieve_and_rerank(user_query, top_k=top_k)

    if not retrieved:
        context = "No relevant case studies found in the knowledge base."
    else:
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            meta = doc["metadata"]
            context_parts.append(
                f"[Source {i}] ({meta.get('doc_type', 'Unknown')} | "
                f"{meta.get('domain', 'General')} | {meta.get('framework', 'General')})\n"
                f"{doc['text']}"
            )
        context = "\n\n".join(context_parts)

    return (
        "You are an expert AI Product Manager. Use the following real-world case studies "
        "and product knowledge to answer the question. Structure your response as:\n"
        "1. Problem Definition\n"
        "2. User Persona\n"
        "3. Prioritized Solutions\n"
        "4. Trade-offs\n"
        "5. Primary Metrics & Counter-Metrics\n\n"
        f"--- Retrieved Context ---\n{context}\n\n"
        f"--- Question ---\n{user_query}"
    )
