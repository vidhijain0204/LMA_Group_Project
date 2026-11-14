"""Check that the project is ready for the actual run.

    python scripts/preflight.py                 # everything (stop Streamlit first: it loads the embedding model)
    python scripts/preflight.py --skip-model    # do not load the embedding model
    python scripts/preflight.py --skip-tests    # do not run the unit tests

Each check prints OK, WARN or FAIL. FAIL means the run will not work; WARN means it will work but something
is missing or worth knowing. Nothing is modified (one temporary file is created in data/ and deleted again).
"""
import argparse
import importlib.metadata
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"  {status:<4} {name}" + (f"  -  {detail}" if detail else ""))


def read(path):
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def version_tuple(text):
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def section(title):
    print(f"\n{title}")


# ------------------------------------------------------------------------------------ 1 environment
def check_environment():
    section("1. Python environment")
    print(f"       python {sys.version.split()[0]}")
    packages = [("torch", None, "FAIL"), ("transformers", "4.51.0", "FAIL"), ("sentence-transformers", "2.7.0", "FAIL"),
                ("langchain-core", None, "FAIL"), ("langchain-huggingface", None, "FAIL"), ("langgraph", None, "FAIL"),
                ("streamlit", "1.40.0", "WARN"), ("pydantic", None, "FAIL")]
    for name, minimum, severity in packages:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report(severity, name, "not installed" + (" (pip install sentence-transformers)" if name == "sentence-transformers" else ""))
            continue
        if minimum and version_tuple(installed) < version_tuple(minimum):
            report(severity, name, f"{installed} is older than {minimum}")
        else:
            report("OK", name, installed)


def check_gpu():
    section("2. GPU and ports")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            detail = f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB"
            if free < 20e9:
                report("WARN", "GPU memory", detail + ". The 8B tutor model needs about 16 GB, plus the embedding model "
                       "(1-2.5 GB) and room for the replies. If Streamlit or another job is running, stop it first")
            else:
                report("OK", "GPU memory", detail)
        else:
            report("WARN", "GPU", "no GPU visible: the tutor model will be very slow")
    except ImportError:
        report("FAIL", "GPU", "torch is not installed")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 8501)) == 0:
            report("WARN", "port 8501", "something is already listening (is Streamlit still running? it holds GPU memory)")
        else:
            report("OK", "port 8501", "free")


# ------------------------------------------------------------------------------------ 3 data
def check_data():
    section("3. Data files")
    corpus = read("data/common/corpus.jsonl")
    if corpus is None:
        report("FAIL", "data/common/corpus.jsonl", "missing")
    else:
        chunks = [json.loads(line) for line in corpus.splitlines() if line.strip()]
        needed = {"chunk_id", "topic_id", "text", "type", "difficulty"}
        incomplete = sum(1 for c in chunks if not needed <= set(c))
        report("OK" if chunks and not incomplete else "FAIL", "corpus", f"{len(chunks)} chunks" + (f", {incomplete} missing fields" if incomplete else ""))
        extras = [k for k in ("subconcept_ids", "uses_later_topics", "duplicate_group", "context_chunk_ids") if chunks and k not in chunks[0]]
        if extras:
            report("WARN", "corpus safety fields", "missing " + ", ".join(extras) + ": the retriever's safety checks will do nothing")
    for path, label in (("data/common/topics.json", "topics"), ("data/common/mcqs_list.json", "quiz bank"),
                        ("data/diagnosis/static_diagnosis_mcq_list.json", "diagnostic questions")):
        text = read(path)
        if text is None:
            report("FAIL", path, "missing")
            continue
        try:
            report("OK", label, f"{len(json.loads(text))} entries")
        except json.JSONDecodeError as error:
            report("FAIL", path, f"not valid JSON ({error})")
    text = read("data/skb_database.json")
    if text is None:
        report("WARN", "data/skb_database.json", "missing: it is created when the first student takes the diagnostic")
    else:
        try:
            report("OK", "student knowledge base", f"{len(json.loads(text))} students")
        except json.JSONDecodeError as error:
            report("FAIL", "data/skb_database.json", f"not valid JSON ({error})")
    for path, label in (("data/common/retrieval_eval.csv", "annotated queries"), ("data/evaluation/student_questions.csv", "student-style questions")):
        report("OK" if read(path) else "WARN", label, path if read(path) else f"{path} missing (only the evaluation scripts need it)")
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.join(ROOT, "data"), delete=True):
            report("OK", "data/ is writable", "student records can be saved")
    except OSError as error:
        report("FAIL", "data/ is writable", str(error))


# ------------------------------------------------------------------------------------ 4 code state
def check_code():
    section("4. Code state (which updates are in place)")
    required = [
        ("agents/course_retriever.py", 'DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"', "retriever uses Qwen3-Embedding"),
        ("agents/course_retriever.py", "original = dict(raw)", "retriever topic boost scaled for dense and hybrid"),
        ("agents/tutor_agent.py", "RETRIEVAL_METHOD = os.getenv", "tutor uses the configurable method with BM25 fallback"),
        ("agents/tutor_agent.py", "def generate_complete_reply", "tutor continues replies that hit the length limit"),
        ("agents/llm_factory.py", "SETTINGS_APPLIED_TO", "tutor generation settings are applied"),
        ("agents/orchestrator.py", "def normalize_route", "orchestrator routes quiz requests correctly"),
        ("agents/assessment_agent.py", "mcqs_list.json", "quizzes use the 455-question bank"),
    ]
    for path, marker, label in required:
        text = read(path)
        report("OK" if text and marker in text else "FAIL", label, "" if text and marker in text else f"not found in {path}")
    important = [
        ("agents/workflow.py", "update_after_quiz", "a quiz updates the student's knowledge base",
         "without it every student keeps the diagnostic profile for the whole run"),
        ("models/skb_schema.py", "history", "the knowledge base keeps a quiz history", "needed by the knowledge-base update"),
        ("agents/course_retriever.py", "Only ever lower the limit", "retriever input-limit fix",
         "only matters if you switch to an embedding model with a short position table (e.g. MiniLM); harmless to apply"),
    ]
    for path, marker, label, why in important:
        text = read(path)
        report("OK" if text and marker in text else "WARN", label, "" if text and marker in text else why)
    app = read("app.py") or ""
    report("OK" if "diagnosis_started_for" in app else "WARN", "the diagnostic opens automatically for a new student",
           "" if "diagnosis_started_for" in app else "optional: otherwise the student clicks 'Start Diagnostic Test' or types a message")
    report("OK" if read("scripts/evaluate_rag.py") else "WARN", "RAG evaluation script", "" if read("scripts/evaluate_rag.py") else "scripts/evaluate_rag.py not installed")
    report("OK" if "sentence-transformers" in (read("requirements.txt") or "") else "WARN", "requirements.txt lists sentence-transformers",
           "" if "sentence-transformers" in (read("requirements.txt") or "") else "a fresh install would not get it, and hybrid retrieval needs it")
    log_markers = [m for m in ("traces", ".jsonl") if m in (read("agents/workflow.py") or "") + app]
    report("OK" if log_markers else "WARN", "a record of what students do", "" if log_markers else
           "nothing writes a log of student actions: the run will leave no trace beyond the knowledge base")


# ------------------------------------------------------------------------------------ 5 models
def check_models(skip_model):
    section("5. Models")
    factory = read("agents/llm_factory.py") or ""
    match = re.search(r'SLM_MODEL_PATH\s*=\s*\(?\s*((?:"[^"]*"\s*)+)\)?', factory)
    if match:
        path = "".join(re.findall(r'"([^"]*)"', match.group(1)))
        files = os.listdir(path) if os.path.isdir(path) else []
        weights = [f for f in files if f.endswith(".safetensors")]
        report("OK" if "config.json" in files and weights else "FAIL", "tutor model files",
               f"{path} ({len(weights)} weight files)" if weights else f"{path} not found or incomplete")
    else:
        report("WARN", "tutor model files", "could not read SLM_MODEL_PATH from agents/llm_factory.py")
    if skip_model:
        report("WARN", "embedding model", "not loaded (--skip-model)")
        return
    try:
        from agents.course_retriever import CourseRetriever
        retriever = CourseRetriever()
        start = time.perf_counter()
        retriever.search("warm up question", method="hybrid")
        report("OK", "embedding model loads", f"{retriever.embedding_model_name}, {len(retriever.documents)} chunks embedded or loaded from cache in {time.perf_counter() - start:.0f} s")
        cache = os.path.join(ROOT, ".cache", "course_retriever")
        cached = [f for f in os.listdir(cache) if f.endswith(".npy")] if os.path.isdir(cache) else []
        report("OK" if cached else "WARN", "chunk embeddings cached", f"{len(cached)} file(s) in .cache/course_retriever" if cached else "no cache file was written")
    except Exception as error:
        report("FAIL", "embedding model loads", f"{type(error).__name__}: {error}. The tutor would fall back to BM25")
        return
    samples = [("teach me list in python", "T08"), ("why does my loop never stop", "T06"), ("how do i read a file line by line", "T13")]
    latencies, misses = [], []
    for question, topic in samples:
        retriever.clear_query_cache()
        start = time.perf_counter()
        found = retriever.search(question, method="hybrid", preferred_topics=[topic], limit=4)
        latencies.append((time.perf_counter() - start) * 1000)
        if not any(r["topic_id"] == topic for r in found):
            misses.append(question)
        print(f"       {question!r} -> {', '.join(r['chunk_id'] for r in found)}")
    report("OK" if not misses else "WARN", "hybrid search returns on-topic chunks", f"median {statistics.median(latencies):.0f} ms per search" +
           (f"; no {topic} chunk for: {', '.join(misses)}" if misses else ""))


# ------------------------------------------------------------------------------------ 6 tests
def check_tests():
    section("6. Unit tests")
    modules = [m for m in ("tests.test_course_retriever", "tests.test_tutor_rag") if os.path.exists(os.path.join(ROOT, *m.split(".")) + ".py")]
    if not modules:
        report("WARN", "unit tests", "no test files found")
        return
    outcome = subprocess.run([sys.executable, "-m", "unittest", *modules], cwd=ROOT, capture_output=True, text=True, timeout=600)
    last = (outcome.stderr.strip().splitlines() or ["(no output)"])[-1]
    ran = next((l for l in outcome.stderr.splitlines() if l.startswith("Ran ")), "")
    report("OK" if outcome.returncode == 0 else "FAIL", "retriever and tutor tests", f"{ran}; {last}")


def main():
    parser = argparse.ArgumentParser(description="Check that the project is ready for the actual run")
    parser.add_argument("--skip-model", action="store_true", help="do not load the embedding model")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the unit tests")
    args = parser.parse_args()
    print("Preflight check for the AI Python tutor")
    check_environment()
    check_gpu()
    check_data()
    check_code()
    check_models(args.skip_model)
    if not args.skip_tests:
        check_tests()
    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"\n{results.count('OK')} OK, {warns} WARN, {fails} FAIL")
    print("READY for the run." + (" Read the WARN lines first." if warns else "") if not fails else "NOT READY: fix the FAIL lines above, then run this check again.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
