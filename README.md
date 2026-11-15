# AI Python Tutoring System

An adaptive, multi-agent tutoring system for Python programming. It uses a **Student Knowledge Base (SKB)** to personalize every interaction — routing new students through a diagnostic test, fielding concept questions with a RAG-backed tutor, and serving adaptive quizzes calibrated to each student's current mastery level.

---

## Table of Contents

- [Architecture](#architecture)
- [Models](#models)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Data](#data)
- [Retrieval (RAG)](#retrieval-rag)
- [Evaluation Results](#evaluation-results)
- [Reproducibility Notes](#reproducibility-notes)

---

## Architecture

The system is built on **LangGraph** as a stateful agent graph. Every student message flows through the following pipeline:

```
User Message
     │
     ▼
 Orchestrator  (Qwen3-4B, thinking mode, greedy)
     │
     ├──► Diagnosis Agent   — one-time onboarding MCQ test → builds SKB
     ├──► Tutor Agent       — hybrid RAG + Qwen3-4B → personalized explanation
     └──► Assessment Agent  — adaptive MCQ quiz → updates SKB mastery
```

**Orchestrator** (`agents/orchestrator.py`): Uses Qwen3-4B with chain-of-thought (`enable_thinking=True`) to parse the student's intent and route to the correct sub-agent. New students (no SKB record) are always sent to diagnosis first.

**Diagnosis Agent** (`agents/diagnosis_agent.py`): Administers a static MCQ battery. Scores are recorded per topic and sub-topic; wrong answers update the student's misconception map. Run exactly once per student.

**Tutor Agent** (`agents/tutor_agent.py`): Retrieves the top-4 course chunks using hybrid BM25 + dense search (Qwen3-Embedding-0.6B), injects them into a system prompt alongside the student's mastery profile, and generates a personalized explanation. If the reply hits the token limit it automatically continues up to two times.

**Assessment Agent** (`agents/assessment_agent.py`): Selects up to 10 MCQs from the 455-question bank, targeting the difficulty level that matches the student's current mastery. Grading updates `score_history`; wrong answers feed back into the SKB.

**Student Knowledge Base** (`models/skb_schema.py`, `data/skb_database.json`): A per-student record that holds topic mastery probabilities (0–1), strengths/weaknesses lists, and active misconceptions. It drives retrieval personalization and tutor pedagogy rules.

---

## Models

| Role | Model | Mode | Max New Tokens |
|---|---|---|---|
| Orchestrator / Router | `Qwen/Qwen3-4B` | Greedy, thinking enabled | 5 000 |
| Tutor | `Qwen/Qwen3-4B` | Sampling (T=0.7, top-p=0.9) | 2 048 |
| Assessment intro | `Qwen/Qwen3-4B` | Greedy | 512 |
| Embeddings (RAG) | `Qwen/Qwen3-Embedding-0.6B` | — | — |

Both models are loaded locally from the HuggingFace cache using `transformers`. The base LLM is shared across all roles through `llm_factory.py` to avoid loading it twice.

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 20 GB (Qwen3-4B in bf16 ≈ 16 GB + embedding model ≈ 1.5 GB) | 24 GB (e.g. RTX 3090 / A10G) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB (models + embeddings cache) | — |
| CUDA | 11.8+ | 12.x |

> **CPU-only**: Not recommended for the LLM. The embedding model (0.6 B parameters) can run on CPU but will be slow at build time (embedding the full corpus once).

### Software

- Python 3.11 or 3.12
- CUDA toolkit matching your PyTorch build

### Python Dependencies

```
torch>=2.6.0
transformers>=4.51.0
accelerate>=1.0.0
safetensors>=0.4.5
langchain-core>=0.3.0
langchain-huggingface>=0.1.0
langgraph>=0.2.0
streamlit>=1.40.0
pydantic>=2.7.0
sentence-transformers>=2.7.0
numpy>=1.26
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vidhijain0204/LMA_Group_Project.git
cd LMA_Group_Project
git checkout dev
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the models

#### LLM — Qwen3-4B

```bash
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/b968826d9c46
```

Or via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B")
```

#### Embedding model — Qwen3-Embedding-0.6B

Downloaded automatically by `sentence-transformers` on first use. To pre-fetch:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
```

### 5. Update the model path

Open `agents/llm_factory.py` and set `SLM_MODEL_PATH` to the snapshot directory on your machine:

```python
SLM_MODEL_PATH = "/your/path/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<snapshot-hash>"
```

If you want to load directly by model name (requires internet or a pre-populated HF cache), you can also just set:

```python
SLM_MODEL_PATH = "Qwen/Qwen3-4B"
```

### 6. Verify the setup (optional)

```bash
python scripts/preflight.py
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**First-time flow:**
1. Enter a student ID in the sidebar (e.g. `S001`).
2. Click **Start Diagnostic Test** — the system shows a one-time MCQ battery that builds the Student Knowledge Base.
3. After submitting, the student can chat freely, ask for explanations, or request a quiz.

**Switching students:** Type a different ID in the sidebar and click **Load / Switch Student**.

**CLI / scripted usage:** See `main.py` for a minimal example that calls the tutor directly without the Streamlit UI.

---

## Project Structure

```
.
├── agents/
│   ├── orchestrator.py        # Intent parsing and routing (LangGraph Command)
│   ├── diagnosis_agent.py     # One-time onboarding diagnostic test
│   ├── tutor_agent.py         # RAG-augmented explanation agent
│   ├── assessment_agent.py    # Adaptive MCQ quiz generator and grader
│   ├── course_retriever.py    # BM25 / TF-IDF / dense / hybrid retrieval engine
│   ├── llm_factory.py         # Shared LLM loader (cached, one model in memory)
│   └── workflow.py            # LangGraph StateGraph wiring
├── models/
│   └── skb_schema.py          # Pydantic schema for Student Knowledge Base
├── data/
│   ├── common/
│   │   ├── corpus.jsonl       # Course material chunks (the retrieval corpus)
│   │   ├── mcqs_list.json     # 455-question assessment bank (13 topics)
│   │   ├── topics.json        # Topic catalogue with misconception maps
│   │   └── retrieval_eval.csv # 22 annotated queries for retrieval evaluation
│   ├── diagnosis/
│   │   └── static_diagnosis_mcq_list.json   # Onboarding test questions
│   ├── evaluation/            # RAG and method-comparison reports + CSVs
│   ├── traces/                # Sample student session traces (JSONL)
│   └── skb_database.json      # Persistent Student Knowledge Base store
├── scripts/
│   ├── preflight.py           # Quick environment / path sanity check
│   ├── evaluate_rag.py        # RAG retrieval evaluation (Recall@5, MRR, hit@5)
│   ├── compare_methods.py     # Compare BM25 / TF-IDF / dense / hybrid
│   └── ...                    # Other analysis and report scripts
├── reports/                   # Retrieval comparison CSVs
├── app.py                     # Streamlit entry point
├── main.py                    # CLI entry point (tutor only)
└── requirements.txt
```

---

## Data

| File | Description |
|---|---|
| `data/common/corpus.jsonl` | Course material in JSON-Lines format. Each line is a chunk with `chunk_id`, `topic_id`, `type`, `difficulty` (1–3), and optional `uses_later_topics`, `duplicate_group`, `needs_context` safety flags. |
| `data/common/mcqs_list.json` | 455 multiple-choice questions used by the Assessment Agent. Fields: `prob_id`, `prob_body`, `options` (dict A–D), `correct_option`, `difficulty`, `related_topics`. |
| `data/common/topics.json` | 13 Python topics (`T01`–`T13`) with names, subtopics, and common misconception tags. |
| `data/diagnosis/static_diagnosis_mcq_list.json` | Questions used exclusively for the one-time onboarding diagnostic. |
| `data/skb_database.json` | Live database of Student Knowledge Bases. Each entry is the JSON serialization of a `StudentKnowledgeBase` Pydantic model. |

---

## Retrieval (RAG)

The `CourseRetriever` (`agents/course_retriever.py`) supports four methods:

| Method | Description |
|---|---|
| `bm25` | BM25 over an in-memory inverted index. Default. |
| `tfidf` | TF-IDF cosine similarity. |
| `dense` | Qwen3-Embedding-0.6B cosine similarity. Corpus embeddings are computed once and cached under `.cache/course_retriever/`. |
| `hybrid` | Reciprocal Rank Fusion of BM25 + dense scores. Used by the Tutor Agent. |

The tutor always calls `method="hybrid"` and passes the student's mastery scores so that chunk difficulty is weighted toward what the student is ready for (`difficulty_weight=0.5`). Safety checks (later-topic avoidance, deduplication, context injection) are enabled by default.

The embedding model uses an instruction prefix on the query side only:

```
Instruct: Given a student's question about Python programming,
retrieve the course passages that help answer it
Query: <student question>
```

Corpus embeddings are saved as a `.npy` file keyed by a hash of the corpus and the model name, so rebuilding on a different machine or after a corpus update is automatic.

---

## Reproducibility Notes

- **Model snapshot**: The repo was developed against `Qwen/Qwen3-4B` snapshot `b968826d9c46`. Pinning to that snapshot ensures identical tokenization and generation behaviour.
- **Random seeds**: The Assessment Agent shuffles MCQs with Python's built-in `random`; no global seed is set, so quiz question order will vary across runs. Set `random.seed(<int>)` before calling `select_adaptive_mcqs` if you need a fixed order.
- **Embedding cache**: On first run with dense/hybrid retrieval, the full corpus is embedded and saved to `.cache/course_retriever/`. Delete that directory and re-run `evaluate_rag.py` to rebuild it from scratch.
- **SKB database**: `data/skb_database.json` ships with sample student records (`S001`, `S201`, `S211`). To start from a clean state, delete or empty this file — the app will re-run the diagnostic test for any student ID not found in it.
- **Evaluation scripts**: All evaluation scripts in `scripts/` read from `data/common/retrieval_eval.csv` and `data/common/corpus.jsonl`. Running `python scripts/evaluate_rag.py` reproduces the retrieval numbers in the table above without needing the LLM.
- **Python version**: Tested on Python 3.11 and 3.12. Python 3.10 is not supported (uses `match` statement patterns internally via LangGraph).
- **`device_map="auto"`**: The LLM is loaded with `device_map="auto"`, so PyTorch distributes layers across all available GPUs automatically. On a single-GPU machine this fills that GPU; on CPU-only it falls back to CPU (slow).
