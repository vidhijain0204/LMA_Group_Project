## New Changes and Results

### Adaptive Pedagogy Framework

The tutor now selects from six teaching strategies based on the student's topic mastery
and saved assessment history:

1. **Hint** — give a targeted clue and guiding question without immediately revealing
   the full solution.
2. **Step Scaffold** — break the task into ordered, manageable steps.
3. **Explicit Explanation** — explain the rule and rationale directly, with a concise
   example.
4. **Worked Example** — demonstrate a complete solution step by step, then invite the
   student to try a similar problem.
5. **Self-Explanation** — prompt the student to predict or explain a step, with a
   focused clue instead of the full answer.
6. **Contrastive Feedback** — compare a correct example with a plausible error and
   explain the difference.

The first strategy is selected from mastery: below 0.4 uses Step Scaffold, 0.8 or
higher uses Self-Explanation, and other scores use Explicit Explanation. The remaining
strategies can be selected as the controller adapts.

### New Components and File Changes

The newly added implementation files are:

- `agents/pedagogy_controller.py` — rule-based strategy selection, lesson and assessment
  history updates, learning-signal tracking, and mastery-based strategy reasons.
- `agents/error_analysis_agent.py` — analyzes quiz answers, limits error signals and
  topic accuracy to the requested topic where question metadata permits, and maps
  incorrect answers to misconception tags or related subtopics.

Existing files updated for this implementation include:

- `agents/tutor_agent.py` — adds the chosen strategy, reason, and recurring error
  context to the tutor prompt; returns a lesson record for the workflow to save.
- `agents/workflow.py` — connects tutor and assessment outcomes to the SKB, persists
  updates, and reports quiz error signals.
- `agents/llm_factory.py` — configures the local MLX model client described below.
- `models/skb_schema.py` — adds fields for pedagogy and learning-signal history.
- `app.py` — displays recent teaching history and quiz areas to revisit.

`agents/assessment_agent.py` was not changed for this feature: it continues selecting
and introducing adaptive MCQs. The workflow performs the grading follow-up, analysis,
and persistence after quiz submission.

### Adaptive Strategy Selection

`agents/pedagogy_controller.py` records each tutor lesson's topic, strategy, selection
reason, and timestamp. When a quiz is submitted, `agents/error_analysis_agent.py`
analyzes the questions and answers. The workflow associates the result with the most
recent lesson for the same topic that has not yet been assessed, and records the
strategy used for that lesson.

Two rules can change the next lesson's strategy:

1. If an error signal repeats across at least two of the latest three assessments for a
   topic, the controller prioritizes Contrastive Feedback (unless the latest lesson
   already used it). The quiz bank currently has no `misconception_tags`, so errors
   normally use related subtopic IDs and names as learning-signal proxies.
2. If at least two assessments are associated with the current strategy and their
   average accuracy is below 60%, the controller chooses the next strategy in its
   six-strategy cycle.

Assessment accuracy used for these decisions is topic-aligned accuracy, not necessarily
the overall quiz score: questions tied to other topics are excluded when their metadata
identifies that mismatch. The current topic mastery estimate is updated using 80% of
its previous value and 20% of the topic-aligned assessment accuracy. If the quiz has no
questions that can be aligned to the requested topic, no topic mastery value is changed.

### SKB Persistence

Student records in `data/skb_database.json` gain these fields:

- `pedagogy_history` — a bounded history (up to 100 events) of lesson strategy choices
  and assessment outcomes. Lesson entries include the strategy, topic, reason, and,
  after a matched quiz, its accuracy and error signals. Assessment entries include
  topic, accuracy, associated strategy, and error signals.
- `learning_signals` — counts of observed misconception or subtopic error IDs.
- `learning_signal_descriptions` — human-readable descriptions for those IDs, used to
  inform future tutor responses.

Assessment events are matched to the latest unassessed lesson with the same topic ID;
the quiz's associated strategy is copied onto the assessment entry. Existing SKB
records remain readable because these fields have empty defaults in the schema and are
initialized when needed.

`active_misconceptions` remains the diagnosis-time field for misconception descriptions
and counts. Quiz errors are currently recorded in `learning_signals` and
`learning_signal_descriptions`; the quiz workflow does not update
`active_misconceptions`. Since the current quiz bank has no misconception tags, those
quiz signals are generally related subtopics rather than confirmed misconceptions.

### Local MLX LLM Backend

`agents/llm_factory.py` now configures `Qwen/Qwen3-4B-MLX-4bit` through MLX-LM's
OpenAI-compatible HTTP server and LangChain's `ChatOpenAI` client. The model runs as a
separate local server; the application sends requests to `http://127.0.0.1:8080/v1`.
It does not load the model into the application process. The client uses separate
generation profiles for routing, tutoring, judging, and quiz introductions.

Install the backend packages:

```bash
python -m pip install mlx-lm langchain-openai
```

In another terminal, start the model server:

```bash
mlx_lm.server --model "Qwen/Qwen3-4B-MLX-4bit" --host 127.0.0.1
```

Then launch the app as described below. Note that the current `requirements.txt` still
lists the previous Transformers/Hugging Face dependencies and does not yet include
`mlx-lm` or `langchain-openai`; install the backend packages explicitly as above.


# AI Python Tutoring System

This repository contains a multi-agent Python tutoring application that personalizes instruction using a student knowledge base (SKB), a diagnostic onboarding flow, adaptive quizzes, and a tutor connected to a locally served MLX model.

The project is designed to:
- onboard a new student with a diagnostic test,
- store topic-level mastery data in a JSON knowledge base,
- route requests to the right agent,
- explain Python topics with personalization,
- generate adaptive MCQ quizzes based on student ability,
- run as a Streamlit web app.

## What is in this directory?

- `app.py` — main Streamlit app for the student-facing tutor experience.
- `agents/` — all agent logic and orchestration.
- `data/` — JSON course data, question banks, and student knowledge base.
- `models/` — model/schema definitions for the student knowledge base.
- `requirements.txt` — Python dependencies.
- `scripts/` — offline tools: misconception tagging, weight fitting and the evaluation report.
- `docs/` — `mid_report.pdf` (project report) and `part1_flow.jpeg` (architecture diagram).

## High-level architecture

The system is built around a LangGraph workflow:

1. The orchestrator decides whether the request should go to:
   - `diagnosis` for new students,
   - `tutor_agent` for explanations,
   - `assessment_agent` for quizzes.
2. The `DiagnosisAgent` asks a diagnostic set of MCQs and stores a student knowledge base.
3. The `TutorAgent` uses the student's mastery profile and topic focus to explain concepts adaptively.
4. The `AssessmentAgent` selects a quiz tailored to the student's current performance.
5. After each quiz, the `ErrorAnalysisAgent` grades it and the `RootCauseAgent` finds the subtopic
   (and, when tagged, the misconception) that most likely caused the wrong answers.

## Project structure

```text
.
├── agents/
│   ├── assessment_agent.py      # Picks adaptive quiz questions
│   ├── chat_memory.py           # Short summaries of the last 3 exchanges
│   ├── course_retriever.py      # TF-IDF search over course text (not connected yet)
│   ├── diagnosis_agent.py       # Diagnostic onboarding test and SKB creation
│   ├── error_analysis_agent.py  # Grades a quiz and lists error signals per topic
│   ├── evaluation_store.py      # Saves the students' "why was this wrong?" answers
│   ├── llm_factory.py           # Loads the local Qwen model
│   ├── orchestrator.py          # Routes each message to the right agent
│   ├── pedagogy_controller.py   # Teaching strategy choice and mastery updates
│   ├── root_cause_agent.py      # Finds the root cause of quiz errors (3 methods)
│   ├── struggle_tracker.py      # Logs learning turns and computes Learning Struggle
│   ├── topic_classifier.py      # Detects the topic of a student message
│   ├── turn_judge.py            # Classifies a message (answer / new question, confusion)
│   ├── tutor_agent.py           # Writes the lesson with the chosen strategy
│   └── workflow.py              # LangGraph graph that connects all agents
├── data/
│   ├── common/
│   │   ├── corpus.jsonl         # Course material chunks (topic_classifier, course_retriever)
│   │   ├── mcqs_list.json       # Quiz bank
│   │   └── topics.json          # Topics, subtopics, prerequisites, misconceptions
│   ├── diagnosis/
│   │   └── static_diagnosis_mcq_list.json
│   └── skb_database.json        # Student knowledge base records
├── models/
│   └── skb_schema.py            # Pydantic schema for a student record
├── scripts/
│   ├── tag_mcq_misconceptions.py  # Tags wrong quiz options with misconceptions (run once)
│   ├── fit_weights.py             # Learns the 5 hypothesis weights from labelled quizzes
│   ├── evaluate.py                # Builds the evaluation report
│   └── eval_data.py               # Shared data loader for the two scripts above
├── docs/
│   ├── mid_report.pdf           # Project report
│   └── part1_flow.jpeg          # Architecture diagram
├── app.py                       # Streamlit tutor UI
├── requirements.txt
└── README.md
```

## Requirements

This project recommends Python 3.11 or 3.12 (see `requirements.txt`) and requires a
running local MLX-LM server for `Qwen/Qwen3-4B-MLX-4bit`. MLX-LM is intended for
supported Apple Silicon systems.

The current dependency manifest has not yet been migrated from the Transformers
backend. In addition to the repository dependencies, install the MLX model server and
OpenAI-compatible LangChain client:

```bash
python -m pip install -r requirements.txt
python -m pip install mlx-lm langchain-openai
```

See [Local MLX LLM Backend](#local-mlx-llm-backend) above for how to start the
separate inference server.

## Setup

From the project root:

```bash
cd /path/to/LMA_Group_Project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```


## How to run the app

### 1) Launch the student web app

This is the main usage path for the tutoring system.

```bash
cd /path/to/LMA_Group_Project
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

### What the UI does

In the sidebar:
- enter a `Student ID`
- load or switch student profiles
- see topic strengths/weaknesses
- start a diagnostic test for a new student
- reset the session

In the main chat:
- ask Python questions,
- ask for explanations,
- request a quiz,
- answer diagnostic or quiz interrupts.

The app automatically routes the request through the workflow.

### New student flow

A new student sees the onboarding diagnostic flow:
1. The workflow notices there is no SKB entry.
2. It goes to the diagnosis node.
3. The student answers diagnostic MCQs.
4. The system computes topic mastery and saves a record in `data/skb_database.json`.
5. The student can then ask questions or take quizzes.

### Existing student flow

When a student already exists:
- the orchestrator examines the user message,
- chooses whether to tutor or quiz,
- uses topic score information to personalize explanations and challenge level.

## How the components work

### `agents/orchestrator.py`

The orchestrator decides routing using the message, current student status, and a topic catalog. It supports:
- `diagnosis`
- `tutor_agent`
- `assessment_quiz`

It also handles fallback logic and JSON parsing when the model returns imperfect output.

Example behavior:
- no SKB record for student => route to `diagnosis`
- concept explanation request => route to tutor
- request for a quiz/test => route to assessment

### `agents/diagnosis_agent.py`

This agent loads:
- topic definitions from `data/common/topics.json`
- diagnosis questions from `data/diagnosis/static_diagnosis_mcq_list.json`

It:
- presents MCQs,
- grades the student's answers,
- computes topic-wise mastery scores,
- records weakness and misconception information,
- saves a new student record to `data/skb_database.json`.

### `agents/tutor_agent.py`

This is the educational explanation model.

It uses:
- the student KB,
- topic focus,
- mastery level,
- recorded misconceptions,
- system prompts for teaching style.

It responds with more basic explanations for low-mastery topics and more advanced answers for strong students.

### `agents/assessment_agent.py`

This agent:
- reads the quiz bank from `data/common/mcqs_list.json`,
- picks questions based on the current topic and mastery,
- adjusts difficulty: easy/foundation vs. intermediate vs. advanced,
- collects student answers,
- calculates accuracy and stores result information.

### Adaptive pedagogy and error analysis

For each explanation, the tutor selects one of six teaching strategies: hints, step
scaffolding, explicit explanation, worked example, self-explanation, or contrastive
feedback. The selected strategy and its reason are saved in that student's
`pedagogy_history` in `data/skb_database.json`.

When a student completes a quiz, the error-analysis component records accuracy and
incorrect-answer signals from misconception tags (when available) or the question's
related subtopics. It pairs the outcome with the most recent unassessed lesson on that
topic and updates the student's topic mastery estimate. If the same strategy is linked
to at least two quizzes averaging below 60%, the next lesson switches strategy. A
recurring signal across multiple quizzes prioritizes contrastive feedback. This uses
assessment outcomes rather than asking students to rate the tutor; a new profile starts
with a mastery-based strategy until enough evidence is available.

### `models/skb_schema.py`

Defines the student knowledge base schema using Pydantic.

Key fields:
- `student_id`
- `topics`: topic mastery probabilities
- `skills`: subtopic mastery probabilities
- `strengths`: above-threshold topics
- `weaknesses`: below-threshold topics
- `active_misconceptions`: issue counts
- `pedagogy_history`: lesson strategies paired with quiz outcomes
- `learning_signals`: recurring misconception/subtopic error counts
- `learning_signal_descriptions`: human-readable learning-signal descriptions
- timestamps

### `data/`

The data directory is the data layer.

- `common/topics.json` — topic definitions, prerequisites, and common misconceptions
- `common/mcqs_list.json` — quiz bank with multiple-choice question data
- `common/corpus.jsonl` — course content chunks for retrieval
- `diagnosis/static_diagnosis_mcq_list.json` — onboarding questions
- `skb_database.json` — saved student profiles

## How to inspect or modify student data

The SKB is stored in JSON. You can open:

```text
data/skb_database.json
```

and inspect entries like:

```json
{
  "student_id": "S001",
  "topics": {"T01": 0.9, "T02": 0.85},
  "strengths": ["T01", "T02"],
  "weaknesses": []
}
```

If you want to reset a student profile, remove the matching record from this file or create a new student ID.

## Running the study and evaluating it

**Before the study (once, on the GPU machine):**
1. Tag the wrong options of the quiz bank with misconceptions:
   `python scripts/tag_mcq_misconceptions.py --limit 10 --dry-run` (check), then
   `python scripts/tag_mcq_misconceptions.py`. Review a sample in
   `data/evaluation/mcq_misconception_review.csv` and fix wrong tags in `data/common/mcqs_list.json`.
2. Download the topic-classifier model once: `TOPIC_EMBED_DOWNLOAD=1 python -c "from agents.topic_classifier import get_classifier; print(get_classifier().backend)"`
   (prints `embedding`; without it the app uses the weaker TF-IDF backend).

**Per student** (`streamlit run app.py`):
1. Click **🆕 New anonymous student** (S101, S102, ...).
2. Diagnosis → learn with the tutor → "ready for assessment" → quiz.
3. After the quiz: the student fills **🧾 why did you get these wrong?**, clicks **📘 Review**,
   learns, then **🔁 Re-test** (this measures the learning gain).
4. For an A/B comparison of the root-cause methods, turn on *Assign method by student id* in
   **🔎 Root-cause settings** before the student starts.

**After the study:**
- `python scripts/evaluate.py` → `data/evaluation/report/report.md` (+ CSVs): root-cause accuracy
  per method/weight set/ablation, agreement, misconception check, learning gain, judge vs rules,
  latency.
- Optional expert labels: `python scripts/evaluate.py --export-labeling`, fill
  `data/evaluation/labeling_sheet.csv`, then `python scripts/evaluate.py --import-labels <file>`.
- `python scripts/fit_weights.py --save` learns the weights (report its leave-one-student-out
  accuracy). The app's **Custom weights** sliders then start from these values; with custom
  weights off, all five parameters get the same weight (plain average).

Environment switches: `TUTOR_LLM_JUDGE=0` (rules instead of the LLM turn judge),
`TOPIC_EMBED_MODEL=none` (TF-IDF topic classifier).

## Common issues and fixes

### Cannot connect to the local model

Start the MLX-LM server in a separate terminal using the command in
[Local MLX LLM Backend](#local-mlx-llm-backend). The client expects it at
`http://127.0.0.1:8080/v1`; model-backed requests fail while that server is stopped or
unavailable.

### `python` command not found

Use:

```bash
python3
```

instead of `python` if your system only exposes the Python 3 binary.

### Streamlit does not open

Make sure the app dependencies are installed and the local MLX-LM model server is
running.

### A student cannot be diagnosed twice

The workflow intentionally prevents multiple diagnosis runs for the same student ID. This is enforced in `DiagnosisAgent.administer_test()`.

## Typical usage examples

### Ask for an explanation

In the app:

```text
Why does my file get erased when I open it?
```

The orchestrator will choose the tutoring path and answer using the student's assessment profile.

### Request a quiz

In the app:

```text
Give me a quiz on loops.
```

The system will generate a topic-specific adaptive quiz.

### Run the diagnostic flow

For a new student ID in the sidebar, press:
- `Start Diagnostic Test`

The app gives the onboarding assessment and then saves the new KB.

## Summary

This directory contains a full educational AI tutoring system with:
- a student diagnostic engine,
- adaptive explanations,
- quiz generation,
- course retrieval,
- local LLM integration,
- Streamlit UI,
- saved student knowledge state.

The primary entry point for users is:

```bash
python3 -m streamlit run app.py
```

If you want to customize the system, start by editing the files in the `agents/` folder and the JSON datasets in `data/`.
