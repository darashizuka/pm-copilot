# PM Copilot: AI Product Manager & Product Sense Copilot

An end-to-end ML system that writes PRDs, solves product sense cases, and evaluates feature trade-offs using real industry case studies. Built with a full production pipeline: data ingestion, RAG, QLoRA fine-tuning (SFT + DPO), and vLLM deployment.

## Architecture

```
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│   Feature Engineering       │    │   Inference Stage                │
│                             │    │                                  │
│   ETL Pipeline              │    │   RAG Pipeline                   │
│   ┌─────────┐               │    │   Query                         │
│   │ Blogs   │──┐            │    │     ├─ Self-Query (LLM)         │
│   │ GitHub  │  ├─ Clean ──► │    │     ├─ Query Expansion (LLM)    │
│   │ YouTube │──┘   Chunk    │    │     ▼                            │
│   └─────────┘       │      │    │   Qdrant Vector Search           │
│                     ▼      │    │     ├─ BGE Embeddings (768d)     │
│               HuggingFace   │    │     ├─ Metadata Filtering       │
│               Datasets      │    │     ▼                            │
│                             │    │   BGE Cross-Encoder Reranking    │
├─────────────────────────────┤    │     ▼                            │
│   Training Stage            │    │   Augmented Prompt + Model       │
│                             │    │     ├─ PagedAttention (vLLM)    │
│   Llama-3.1-8B-Instruct    │    │     └─ Structured PM Output     │
│     ▼                       │    └──────────────────────────────────┘
│   QLoRA SFT (2000 samples)  │
│     ▼                       │
│   QLoRA DPO (90 pairs)      │
│     ▼                       │
│   Merged Model → vLLM       │
└─────────────────────────────┘
```

## Key Features

- **RAG Pipeline**: BGE embeddings → Qdrant Cloud vector search → LLM-based self-query & query expansion → BGE cross-encoder reranking
- **QLoRA SFT**: Fine-tuned Llama-3.1-8B-Instruct on 2000 PM instruction-response pairs (Kaggle PRDs + Self-Distilled CoT from DeepSeek-V3)
- **DPO Alignment**: Preference optimization using chosen (27B model + RAG context) vs rejected (SFT model, no context) pairs
- **vLLM Serving**: FastAPI server with PagedAttention for efficient inference
- **Benchmark**: 15.8 tok/s median throughput, 13.2s mean latency (NVIDIA L4 GPU)

## Project Structure

```
LLM_finetune/
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters
│   └── dpo_config.yaml          # DPO hyperparameters
├── src/
│   ├── ingestion/
│   │   ├── web_scraper.py       # Trafilatura + GitHub API scraper
│   │   ├── youtube_scraper.py   # yt-dlp subtitle downloader
│   │   └── chunker.py           # 500-word chunks with overlap
│   ├── rag/
│   │   ├── vector_store.py      # Qdrant Cloud + BGE embeddings
│   │   ├── retriever.py         # Self-query, expansion, reranking
│   │   └── llm_client.py        # Swappable LLM backend (Groq/local/SageMaker)
│   ├── training/
│   │   ├── sft_train.py         # QLoRA SFT with TRL
│   │   ├── dpo_train.py         # QLoRA DPO with TRL
│   │   ├── convert_prd_dataset.py   # Kaggle PRD → SFT format
│   │   ├── generate_sft_cot.py      # Self-Distilled Reasoning (DeepSeek-V3)
│   │   └── generate_dpo_pairs.py    # DPO preference pair generation
│   └── deploy/
│       ├── merge_model.py       # Merge LoRA adapter into base model
│       ├── vllm_server.py       # FastAPI + vLLM inference server
│       ├── sagemaker_deploy.py  # SageMaker endpoint deployment
│       └── benchmark.py         # Latency & throughput benchmarking
├── requirements.txt
└── benchmark_results.txt
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (L4 or better) for training/inference
- Accounts: [HuggingFace](https://huggingface.co), [Qdrant Cloud](https://cloud.qdrant.io) (free tier), [Groq](https://console.groq.com) (free tier)

### Installation

```bash
git clone https://github.com/darashizuka/pm-copilot.git
cd pm-copilot
pip install -r requirements.txt
```

### Environment Variables

```bash
export HF_TOKEN=hf_...                    # HuggingFace token
export QDRANT_URL=https://...qdrant.io:6333  # Qdrant Cloud URL
export QDRANT_API_KEY=...                 # Qdrant API key
export GROQ_API_KEY=gsk_...              # Groq API key (for self-query/expansion)
```

## Usage

### 1. Data Ingestion

```bash
# Scrape PM articles from blogs and GitHub
python -m src.ingestion.web_scraper

# Chunk and upload to HuggingFace
python -m src.ingestion.chunker
```

### 2. RAG Pipeline

```bash
# Load embeddings into Qdrant
python -m src.rag.vector_store

# Test retrieval with reranking
python -c "
from src.rag.retriever import build_augmented_prompt
print(build_augmented_prompt('How to prioritize features for B2B SaaS?'))
"
```

### 3. Training

```bash
# SFT training (QLoRA on Llama-3.1-8B-Instruct)
python -m src.training.sft_train

# Generate DPO preference pairs
python -m src.training.generate_dpo_pairs --n-pairs 200

# DPO training
python -m src.training.dpo_train
```

### 4. Inference

```bash
# Merge adapter into base model
python -m src.deploy.merge_model

# Start vLLM server
python -m src.deploy.vllm_server --port 8000

# Run benchmark
python -m src.deploy.benchmark --n 10 --max-tokens 256 --rag
```

### 5. Query the API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a referral program for a FinTech lending platform", "max_tokens": 512, "use_rag": true}'
```

## Model & Data

All artifacts are hosted on HuggingFace under [`shizukadara2`](https://huggingface.co/shizukadara2):

| Artifact | Link |
|---|---|
| DPO Adapter | `shizukadara2/pm-copilot-dpo-adapter` |
| SFT Adapter | `shizukadara2/pm-copilot-sft-adapter` |
| SFT Dataset | `shizukadara2/pm-copilot-sft` |
| DPO Dataset | `shizukadara2/pm-copilot-dpo` |
| Raw Articles | `shizukadara2/pm-copilot-raw` |
| Chunks | `shizukadara2/pm-copilot-chunks` |

## Training Details

| Stage | Base Model | Method | Data | Epochs | Hardware |
|---|---|---|---|---|---|
| SFT | Llama-3.1-8B-Instruct | QLoRA (r=16, α=32) | 2000 samples | 3 | NVIDIA L4 24GB |
| DPO | SFT checkpoint | QLoRA (r=16, α=32) | 90 pairs | 3 | NVIDIA L4 24GB |

**SFT Data Sources:**
- Kaggle PRD-Synthetic dataset (structured PRDs across industries)
- Self-Distilled CoT reasoning traces generated by DeepSeek-V3

**DPO Pair Construction:**
- Chosen: Qwen-27B with RAG context + structured PM prompt
- Rejected: SFT model with same prompt, no RAG context

## Benchmark Results

```
Latency (seconds):
  Mean:   13.215
  Median: 15.177
  P95:    15.836

Throughput:
  Mean tokens/sec:   15.8
  Median tokens/sec: 16.3

PagedAttention: enabled
RAG augmented:  True
```

## Tech Stack

- **Model**: Llama-3.1-8B-Instruct + QLoRA (PEFT)
- **Training**: TRL (SFTTrainer, DPOTrainer), bitsandbytes (4-bit quantization)
- **Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions)
- **Reranker**: BAAI/bge-reranker-base
- **Vector DB**: Qdrant Cloud
- **Inference**: vLLM with PagedAttention
- **API**: FastAPI + Uvicorn
- **Data**: HuggingFace Datasets, trafilatura, yt-dlp
