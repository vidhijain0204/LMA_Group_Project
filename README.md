# AI Python Tutoring System

An adaptive, multi-agent tutoring system for Python programming. It uses a **Student Knowledge Base (SKB)** to personalize every interaction — routing new students through a diagnostic test, answering concept questions with a RAG-backed tutor, and serving adaptive quizzes calibrated to each student's current mastery level.

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

```text
User Message
     |
     v
Orchestrator (Qwen3-4B, thinking mode, greedy)
     |
     |----> Diagnosis Agent
     |       One-time onboarding MCQ test -> Builds SKB
     |
     |----> Tutor Agent
     |       Hybrid RAG + Qwen3-4B -> Personalized explanation
     |
     |----> Assessment Agent
             Adaptive MCQ quiz -> Updates SKB
```

**Orchestrator** (`agents/orchestrator.py`): Uses Qwen3-4B with chain-of-thought (`enable_thinking=True`) to understand the student's intent and route the request to the appropriate sub-agent. New students without an existing SKB record are always directed to diagnosis first.

**Diagnosis Agent** (`agents/diagnosis_agent.py`): Administers a static MCQ battery. Scores are recorded per topic and sub-topic. Incorrect answers update the student's misconception map. The diagnostic assessment is conducted exactly once per student.

**Tutor Agent** (`agents/tutor_agent.py`): Retrieves the top-4 relevant course chunks using hybrid BM25 + dense retrieval (Qwen3-Embedding-0.6B). The retrieved context is combined with the student's mastery profile and supplied to Qwen3-4B to generate personalized explanations. If a response reaches the token limit, generation automatically continues for up to two additional attempts.

**Assessment Agent** (`agents/assessment_agent.py`): Selects up to 10 MCQs from the 455-question bank, targeting difficulty levels that match the student's current mastery. Grading updates `score_history`, while incorrect answers provide feedback to the SKB.

**Student Knowledge Base** (`models/skb_schema.py`, `data/skb_database.json`): Maintains an individual record for each student, including topic mastery probabilities (0–1), strengths, weaknesses, and active misconceptions. The SKB drives retrieval personalization and determines the tutor's pedagogical approach.

---

## Models

The system uses **Qwen3-4B** as its primary language model across multiple agent roles. A separate embedding model is used for semantic retrieval.

| Role | Model | Mode | Max New Tokens |
|---|---|---|---|
| Orchestrator / Router | `Qwen/Qwen3-4B` | Greedy, thinking enabled | 5,000 |
| Tutor | `Qwen/Qwen3-4B` | Sampling (T=0.7, top-p=0.9) | 2,048 |
| Assessment Introduction | `Qwen/Qwen3-4B` | Greedy | 512 |
| Embeddings (RAG) | `Qwen/Qwen3-Embedding-0.6B` | Dense embeddings | — |

### Model Loading

Both models are loaded locally from the Hugging Face cache using `transformers` and `sentence-transformers`.

The primary Qwen3-4B language model is shared across all agent roles through `llm_factory.py`. This avoids loading multiple copies of the same model into GPU memory and reduces overall resource consumption.

The embedding model is loaded separately for document and query embeddings.

---

## Requirements

### Hardware

The system uses Qwen3-4B Model.

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB | 16–24 GB |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB | 20 GB or more |
| CUDA | 11.8+ | 12.x |

**GPU memory considerations:**

- Qwen3-4B in BF16 requires approximately 8 GB for model weights.
- Qwen3-Embedding-0.6B requires approximately 1.2 GB for BF16 weights.
- Additional GPU memory is required for inference activations, attention/KV cache, and other runtime overhead.
- A 16 GB GPU is a practical recommendation for running both models, while 24 GB provides additional headroom.

Actual memory consumption depends on the loading configuration, inference settings, sequence lengths, and retrieval workload.

> **CPU-only:** Not recommended for language model inference. The embedding model can run on CPU, but embedding the complete corpus will be considerably slower.

### Software

- Python 3.11 or 3.12
- CUDA toolkit compatible with the installed PyTorch build

### Python Dependencies

```text
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

### 1. Clone the Repository

```bash
git clone https://github.com/vidhijain0204/LMA_Group_Project.git
cd LMA_Group_Project
git checkout dev
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Models

#### Language Model — Qwen3-4B

```bash
huggingface-cli download Qwen/Qwen3-4B
```

Alternatively, download using Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-4B")
```

#### Embedding Model — Qwen3-Embedding-0.6B

The embedding model is downloaded automatically by `sentence-transformers` on first use.

To download it beforehand:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
```

### 5. Configure the Model Path

Open `agents/llm_factory.py` and set `SLM_MODEL_PATH` to the local Qwen3-4B model directory.

```python
SLM_MODEL_PATH = "/your/path/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<snapshot-hash>"
```

If you prefer loading directly by Hugging Face model ID, use:

```python
SLM_MODEL_PATH = "Qwen/Qwen3-4B"
```

This requires internet access or an already populated Hugging Face cache.

### 6. Verify the Setup (Optional)

Run the preflight script to verify the environment and model configuration:

```bash
python scripts/preflight.py
```

---

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

### First-Time Student Flow

1. Enter a student ID in the sidebar (e.g., `S001`).
2. Click **Start Diagnostic Test**. The system administers a one-time MCQ assessment to build the Student Knowledge Base.
3. After submitting the diagnostic test, the student can interact with the system, ask Python-related questions, request explanations, or take adaptive quizzes.

### Switching Students

Enter a different student ID in the sidebar and click **Load / Switch Student** to load another student's knowledge profile.

### CLI / Scripted Usage

Refer to `main.py` for a minimal example of directly invoking the tutor without using the Streamlit interface.

---

## Project Structure

```text
.
├── agents/
│   ├── orchestrator.py        # Intent parsing and routing (LangGraph Command)
│   ├── diagnosis_agent.py     # One-time onboarding diagnostic test
│   ├── tutor_agent.py         # RAG-augmented explanation agent
│   ├── assessment_agent.py    # Adaptive MCQ quiz generator and grader
│   ├── course_retriever.py    # BM25 / TF-IDF / dense / hybrid retrieval
│   ├── llm_factory.py         # Shared Qwen3-4B model loader
│   └── workflow.py            # LangGraph StateGraph wiring
│
├── models/
│   └── skb_schema.py          # Pydantic schema for Student Knowledge Base
│
├── data/
│   ├── common/
│   │   ├── corpus.jsonl       # Course material chunks
│   │   ├── mcqs_list.json     # 455-question assessment bank
│   │   ├── topics.json        # Topic catalogue and misconception maps
│   │   └── retrieval_eval.csv # 22 annotated retrieval queries
│   │
│   ├── diagnosis/
│   │   └── static_diagnosis_mcq_list.json
│   │
│   ├── evaluation/            # RAG and method-comparison reports
│   ├── traces/                # Sample student session traces (JSONL)
│   └── skb_database.json      # Persistent Student Knowledge Base
│
├── scripts/
│   ├── preflight.py           # Environment and path sanity check
│   ├── evaluate_rag.py        # Retrieval evaluation
│   ├── compare_methods.py     # BM25 / TF-IDF / dense / hybrid comparison
│   └── ...                    # Other analysis and report scripts
│
├── reports/                   # Retrieval comparison CSVs
├── app.py                     # Streamlit entry point
├── main.py                    # CLI entry point (tutor only)
└── requirements.txt
```

---

## Data

| File | Description |
|---|---|
| `data/common/corpus.jsonl` | Course material in JSON-Lines format. Each line represents a chunk containing `chunk_id`, `topic_id`, `type`, `difficulty` (1–3), and optional safety flags such as `uses_later_topics`, `duplicate_group`, and `needs_context`. |
| `data/common/mcqs_list.json` | 455 multiple-choice questions used by the Assessment Agent. Fields include `prob_id`, `prob_body`, `options` (A–D), `correct_option`, `difficulty`, and `related_topics`. |
| `data/common/topics.json` | Catalogue of 13 Python topics (`T01`–`T13`) with corresponding subtopics and common misconception tags. |
| `data/diagnosis/static_diagnosis_mcq_list.json` | Questions used exclusively for the one-time student onboarding diagnostic assessment. |
| `data/skb_database.json` | Persistent database of Student Knowledge Bases. Each entry is a JSON serialization of the `StudentKnowledgeBase` Pydantic model. |

---

## Retrieval (RAG)

The `CourseRetriever` (`agents/course_retriever.py`) supports four retrieval methods:

| Method | Description |
|---|---|
| `bm25` | BM25 retrieval over an in-memory inverted index. |
| `tfidf` | TF-IDF cosine similarity. |
| `dense` | Dense semantic retrieval using Qwen3-Embedding-0.6B cosine similarity. Corpus embeddings are computed once and cached. |
| `hybrid` | Reciprocal Rank Fusion (RRF) combining BM25 and dense retrieval. Used by the Tutor Agent. |

### Personalized Retrieval

The Tutor Agent always uses:

```python
method="hybrid"
```

It retrieves the top-4 relevant course chunks and incorporates the student's mastery scores to adjust retrieval toward content that matches the student's current learning level.

The difficulty adjustment is controlled by:

```python
difficulty_weight=0.5
```

Additional retrieval safety mechanisms are enabled by default:

- Later-topic avoidance
- Duplicate chunk removal
- Context injection for passages requiring additional background information

### Embedding Configuration

The Qwen3-Embedding-0.6B model uses an instruction prefix on the query side only:

```text
Instruct: Given a student's question about Python programming,
retrieve the course passages that help answer it
Query: <student question>
```

### Embedding Cache

Corpus embeddings are saved as a `.npy` file under:

```text
.cache/course_retriever/
```

The cache is keyed using a hash of the corpus and embedding model name. Consequently, embeddings are automatically rebuilt when the corpus or embedding model changes.

---

## Evaluation Results

The project includes retrieval evaluation and retrieval-method comparison experiments.

Evaluation resources are maintained under:

```text
data/common/retrieval_eval.csv
data/common/corpus.jsonl
data/evaluation/
reports/
```

The retrieval evaluation uses 22 annotated queries to compare retrieval performance.

The evaluation framework includes:

- Recall@5
- MRR (Mean Reciprocal Rank)
- Hit@5

The following retrieval approaches are compared:

- BM25
- TF-IDF
- Dense retrieval
- Hybrid retrieval (BM25 + Dense using Reciprocal Rank Fusion)

To reproduce retrieval evaluation:

```bash
python scripts/evaluate_rag.py
```

To compare retrieval methods:

```bash
python scripts/compare_methods.py
```

These experiments do not require loading the Qwen3-4B language model because they evaluate the retrieval component independently.

Detailed evaluation reports and CSV outputs are available in the `data/evaluation/` and `reports/` directories.

---

## Reproducibility Notes

- **Language Model:** The system uses `Qwen/Qwen3-4B` as its primary language model for orchestration, tutoring, and assessment introductions.
- **Model Snapshot:** The repository was developed against the Qwen3-4B snapshot `b968826d9c46`. Pinning the same snapshot helps maintain consistent tokenization and generation behavior.
- **Model Sharing:** The Qwen3-4B model is loaded through `llm_factory.py` and shared across agent roles to avoid redundant GPU memory consumption.
- **Generation Configuration:** The orchestrator uses greedy decoding with thinking enabled. The tutor uses sampling with temperature 0.7 and top-p 0.9.
- **Random Seeds:** The Assessment Agent shuffles MCQs using Python's built-in `random` module. No global seed is set by default, so question order may vary across runs. Set `random.seed(<int>)` before calling `select_adaptive_mcqs()` if a fixed order is required.
- **Embedding Cache:** On the first run with dense or hybrid retrieval, the complete corpus is embedded and saved to `.cache/course_retriever/`. Delete this directory and rerun `evaluate_rag.py` to rebuild the cache from scratch.
- **SKB Database:** `data/skb_database.json` ships with sample student records (`S001`, `S201`, `S211`). To start with a clean database, delete or empty this file. The application will then initiate the diagnostic assessment for student IDs that do not exist in the database.
- **Evaluation Scripts:** Retrieval evaluation scripts read from `data/common/retrieval_eval.csv` and `data/common/corpus.jsonl`. They can reproduce retrieval metrics without loading the primary language model.
- **Python Version:** Tested with Python 3.11 and 3.12. Python 3.10 is not supported because the implementation uses Python syntax and dependency features requiring newer versions.
- **Device Mapping:** The language model is loaded using `device_map="auto"`. Accelerate automatically places model layers across available GPUs and, when necessary, other available devices. On a single-GPU machine, the model is placed on that GPU when sufficient memory is available. CPU offloading may occur when GPU memory is insufficient, potentially reducing inference speed.

---

## License and Acknowledgements

This project uses open-source models and libraries, including Qwen3, LangGraph, LangChain, Hugging Face Transformers, Sentence Transformers, and Streamlit.

Please refer to the respective repositories and licenses for the models and dependencies used.
