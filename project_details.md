# AI Python Tutoring System — Project Details

## Abstract

Learning Python is difficult when every learner receives the same explanation,
practice questions, and feedback despite having different prior knowledge. A student
may understand variables but struggle with loops, repeat a misconception about lists,
or need a hint rather than another complete solution. This project addresses that need
with a local, multi-agent tutoring application that combines a diagnostic assessment,
an individual Student Knowledge Base (SKB), adaptive explanations and quizzes, and
follow-up analysis of quiz errors.

The application is intended to make practice more responsive: it tracks topic and
subtopic mastery, keeps a compact learning history, selects teaching strategies from
that history, identifies likely causes of incorrect answers, and can guide a student
through a remediation lesson followed by a targeted re-test. Its study features also
record consent, student-provided explanations of mistakes, feedback, agent traces, and
before/after quiz outcomes so that the instructional and root-cause methods can be
evaluated.

The system runs as a Streamlit application. The LLM is a locally served
`Qwen/Qwen3-4B-MLX-4bit` model accessed through an OpenAI-compatible HTTP endpoint.
LangGraph coordinates the agent workflow and its interactive pauses. Most assessment
selection, student-state updates, struggle signals, pedagogy choices, and root-cause
scoring are implemented as local Python logic rather than delegated to the LLM.

## Project at a glance

| Area | Implementation |
|---|---|
| User interface | Streamlit chat and assessment forms in [app.py](./app.py) |
| Workflow | LangGraph state graph in [agents/workflow.py](./agents/workflow.py) |
| Routing | LLM intent routing with local topic and follow-up checks in [agents/orchestrator.py](./agents/orchestrator.py) |
| LLM | Local MLX-LM server; LangChain `ChatOpenAI` client in [agents/llm_factory.py](./agents/llm_factory.py) |
| Learner state | Pydantic schema and JSON records in [models/skb_schema.py](./models/skb_schema.py) and `data/skb_database.json` |
| Assessments | Fixed diagnostic set and adaptive selection from the MCQ bank |
| Pedagogy | Six strategy prompts, selected and updated using mastery and quiz evidence |
| Root-cause analysis | Three local methods and five evidence parameters in [agents/root_cause_agent.py](./agents/root_cause_agent.py) |
| Study evaluation | JSONL logs, labels, report generation, and optional weight fitting |

## How a student session works

1. **Open the app and select a profile.** The app defaults to an anonymous-style
   student ID. New-study participants see the consent screen while Study mode is on.
2. **Check the SKB.** The orchestrator loads the selected student's record from
   `data/skb_database.json`.
3. **Diagnose new students.** If the student has no record, the workflow pauses and
   presents the fixed diagnostic questions. After answers are submitted, the diagnosis
   agent calculates topic/subtopic scores and creates the SKB.
4. **Route existing students.** The orchestrator considers the request, recent chat,
   topic catalog, and turn-judge analysis, then sends it to tutoring or assessment.
5. **Teach or assess.** The tutor chooses a teaching strategy and generates a response,
   or the assessment agent selects MCQs from the local question bank. A quiz pauses
   until the student submits answers.
6. **Update learning state.** Quiz answers are graded; error signals, root-cause
   hypotheses, topic mastery, subtopic mastery, and learning history are updated.
7. **Close the loop.** When a root cause is available, the app offers a focused review
   and re-test. The re-test record compares performance before and after remediation.

The graph's main nodes are `orchestrator`, `diagnosis`, `tutor_agent`, and
`assessment_agent`. A LangGraph `interrupt` pauses diagnosis or quiz execution; the
Streamlit form submits answers through `workflow.resume()`. LangGraph's in-memory
`MemorySaver` checkpointer keeps an interrupted thread resumable while the application
process is running. It is not a durable database for graph state across a process
restart; the SKB and study files provide the longer-lived records.

## Agents and supporting components

### Orchestrator

[agents/orchestrator.py](./agents/orchestrator.py) decides what should happen next.
New students are sent directly to diagnosis. For existing students, it handles
application-forced remediation/re-test routes, checks whether the message answers the
tutor's previous question, calls the turn judge when appropriate, and otherwise asks
the router LLM for an intent and topic.

The LLM is asked for a JSON decision. The code extracts and validates that JSON,
normalizes alternate route names, and falls back to tutoring if routing output is
unusable. Topic selection is further checked locally: explicitly named topics and
keyword evidence take precedence, a confident classifier can resolve ambiguous
messages, and a current lesson topic is retained for follow-up messages.

For a quiz request without an explicit topic, the orchestrator uses the topics from
the student's recent tutor conversations before falling back to the LLM's suggestion.

### Diagnosis agent

[agents/diagnosis_agent.py](./agents/diagnosis_agent.py) serves the one-time onboarding
assessment from `data/diagnosis/static_diagnosis_mcq_list.json`. It grades submitted
answers, maps questions to catalog topics and subtopics, calculates accuracy-based
mastery, records diagnosis responses and tagged misconceptions, and saves the new SKB.
It refuses to diagnose a student ID that already has an SKB.

The diagnostic questions use a legacy field format (for example, `problem_id`,
`correct option`, and `related topics`). The diagnosis agent accepts those spellings;
the main practice-question bank uses a different format.

### Tutor agent

[agents/tutor_agent.py](./agents/tutor_agent.py) builds the tutor system prompt from
the student's selected topic, mastery, active diagnosis misconceptions, recurring
quiz-error signals, recent root cause, and conversation mode. It asks the local model
to explain the concept using the selected pedagogy strategy. It also handles a
follow-up answer by asking for a verdict and continuing the current lesson without
changing strategy mid-exchange.

The tutor returns the message, strategy, reason, and (where relevant) an answer
verdict. The workflow records the learning turn and updates chat and pedagogy memory.

### Assessment agent

[agents/assessment_agent.py](./agents/assessment_agent.py) selects questions from
`data/common/mcqs_list.json`. It divides the requested question count across selected
topics, prefers weak subtopics and questions the student has not seen, and ranks
questions by fit to the student's target difficulty. A re-test restricts selection to
the root-cause subtopic and uses a shorter quiz.

The assessment LLM call generates only a short introduction. The questions and answer
keys come from the question bank; the model does not invent the quiz questions.
`workflow.py` handles grading after the student submits answers.

### Turn judge

[agents/turn_judge.py](./agents/turn_judge.py) classifies a student message as an
answer to the tutor's question, a new question, or other. When it is an answer, the
judge may also label it correct, partly correct, or incorrect. It detects confusion
and hint requests, and stores rule-based signals alongside its LLM judgment for
comparison.

Set `TUTOR_LLM_JUDGE=0` to disable this additional model call and use the local
rule-based analysis. If the judge call fails or its JSON cannot be parsed, the tutor
continues with rule-based values.

### Pedagogy controller

[agents/pedagogy_controller.py](./agents/pedagogy_controller.py) contains the six
teaching-strategy instructions and determines which one the tutor should use. It
records lesson events, associates quiz outcomes with lessons, tracks recurring errors,
updates topic mastery, and applies a simple exponential moving average to subtopic
mastery.

### Error analysis agent

[agents/error_analysis_agent.py](./agents/error_analysis_agent.py) calculates
overall and topic-aligned quiz accuracy. For incorrect answers, it emits a signal from
the question's misconception tags when available; otherwise it uses the question's
related subtopics or topics. This lets the system adapt even when a question has no
explicit misconception annotation.

### Root-cause agent

[agents/root_cause_agent.py](./agents/root_cause_agent.py) estimates which subtopic
most likely explains the student's wrong answers. The UI can select one of three
methods:

1. **Without KB (`no_kb`)** — choose the subtopic with the largest contribution to
   wrong answers in the current quiz.
2. **Lowest mastery (`kb_low_score`)** — among subtopics associated with wrong
   answers, choose the one with the lowest stored mastery.
3. **Hypothesis testing (`hypothesis`)** — score up to three candidate subtopics using
   five evidence parameters:

| Parameter | Meaning |
|---|---|
| `LS` | Learning struggle observed during tutoring, including confusion, re-asking, turns needed, hint requests, unsuccessful self-explanation, and strategy switches |
| `CE` | Counter-evidence: how often the student answered questions on the subtopic correctly |
| `RP` | Repeated errors on the subtopic in recent quizzes |
| `PW` | Weakness in a prerequisite topic |
| `PM` (`PM'`) | Low or fragile prior mastery |

The hypothesis score is a weighted sum:

```text
Score(H) = wLS*LS - wCE*CE + wRP*RP + wPW*PW + wPM*PM'
PM'(H)   = min(1, (1 - mastery) + 0.5 * LS * mastery)
```

`CE` is subtracted because correct answers are evidence against a suspected root
cause. The top candidates are normalized into probabilities. Presets include
`balanced`, `equal`, `struggle_first`, `history_first`, and `mastery_first`; the app
also supports custom weights and can use a fitted `learned` preset.

The workflow runs a method/preset comparison on each quiz for study evaluation, even
when the UI-selected method determines the displayed root cause. This comparison
includes leave-one-parameter-out ablations of the selected preset.

### Topic classifier and struggle tracker

[agents/topic_classifier.py](./agents/topic_classifier.py) finds likely topic and
subtopic labels by comparing a student message with labeled catalog entries, course
chunks, and quiz questions. It uses a local Sentence Transformers embedding model when
available; otherwise it uses a scikit-learn TF-IDF model. It can abstain if similarity
or vote share is too low. `TOPIC_EMBED_MODEL=none` disables the embedding option.

[agents/struggle_tracker.py](./agents/struggle_tracker.py) provides deterministic
keyword/topic matching, confusion and hint-request flags, subtopic mapping, and the
six-component `LS` struggle estimate used by root-cause analysis. It logs bounded
learning-turn history in the SKB.

### Chat memory

[agents/chat_memory.py](./agents/chat_memory.py) stores a compact summary of the
latest three exchanges in each student's SKB. A lesson summary retains the student's
message, topic, strategy, key points from the reply, and the tutor's closing question.
Quiz summaries retain topics, score, and the main error reason. This supplies context
to future prompts without repeatedly sending the full conversation.

### Course retriever — implementation status

[agents/course_retriever.py](./agents/course_retriever.py) implements a local lexical
TF-IDF retriever over `data/common/corpus.jsonl`. It supports topic filtering,
mastery-based difficulty ranking, source details, and formatted citations.

**The current tutor node does not call `CourseRetriever` or insert retrieved chunks
into its prompt.** The corpus and retriever are present, but the current tutor responds
from the model prompt and the student's context rather than using an active RAG
pipeline. The architecture image depicts a RAG connection; that connection is not
wired into the current runtime path.

## Tool calling and model usage

“Tool calling” can refer to model-issued function calls or to application-level
orchestration. This project uses the latter:

- The **LangGraph workflow** selects and runs Python nodes.
- The **orchestrator** returns `Command(goto=...)` to route to a graph node.
- `interrupt()` pauses for a student response; `resume()` provides the submitted
  answers.
- Local Python functions perform topic matching, question selection, grading,
  persistence, error analysis, pedagogy selection, and root-cause scoring.

There is **no model-bound function/tool-calling interface** in the current code:
there are no `bind_tools`/tool schemas or model-directed external tool executions.
The LLM receives messages and returns text. The application decides which Python
component to call.

### Local model and profiles

The application client in [agents/llm_factory.py](./agents/llm_factory.py) targets
`Qwen/Qwen3-4B-MLX-4bit` through MLX-LM's OpenAI-compatible server. The model runs
separately from Streamlit, by default at `http://127.0.0.1:8080/v1`. The client does
not load the model weights itself.

| Profile | Purpose | Max new tokens | Temperature | Top-p |
|---|---|---:|---:|---:|
| `router` | Intent and topic routing | 1,024 | 0.6 | 0.95 |
| `tutor` | Explanations and lesson dialogue | 2,048 | 0.7 | 0.8 |
| `judge` | Turn classification and optional labeling tasks | 512 | 0.0 | 1.0 |
| `quiz` | Quiz introduction | 2,048 | 0.0 | 1.0 |

The factory uses Qwen's `/think` or `/no_think` directive as configured by each call
site, retries a response up to two times if generation ends at its token limit, and
combines continuation text and usage metadata.

Configuration variables:

| Variable | Default | Purpose |
|---|---|---|
| `TUTOR_MODEL_NAME` | `Qwen/Qwen3-4B-MLX-4bit` | Model identifier or local model directory; use the same value for server and app |
| `TUTOR_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible server URL |
| `TUTOR_LLM_API_KEY` | `local` | Client credential value; a local server normally does not require a real secret |
| `TUTOR_LLM_JUDGE` | `1` | Set to `0` to use only rule-based turn judging |
| `TOPIC_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model; `none` selects TF-IDF instead |
| `TOPIC_EMBED_DOWNLOAD` | unset | Set to `1` to permit downloading the topic-embedding model |

## Adaptive pedagogy: behavior and examples

The first strategy for a topic is based on the student's topic mastery:

- Below `0.4`: `step_scaffolding`
- From `0.4` up to (but not including) `0.8`: `explicit_explanation`
- At least `0.8`: `self_explanation`

For example, for a student with a `0.32` mastery score on loops, the tutor could
introduce a loop by tracing one iteration at a time rather than starting with a dense
general explanation. At `0.55`, the initial response is a direct explanation with a
short example. At `0.86`, it can ask the student to predict a loop's output before
showing the answer.

There are six strategies:

| Strategy | Example on the Python list question “Why does `my_list.sort()` print `None`?” |
|---|---|
| **Hint** | “Check the return value of an in-place operation. What does `my_list` contain after the call, even if the call itself returns nothing?” |
| **Step scaffolding** | “First, note that `sort()` changes the existing list. Next, inspect the method's return value. Finally, compare it with `sorted(my_list)`, which creates a new list.” |
| **Explicit explanation** | Explain that `list.sort()` sorts the original list in place and returns `None`; show a short example, then contrast it with `sorted()` returning a list. |
| **Worked example** | Start with `values = [3, 1, 2]`, execute `result = values.sort()`, and walk through both `values` and `result`; then invite the student to predict `sorted(values)`. |
| **Self-explanation** | Ask, “Before running it, what do you expect `values` and `result` to contain after `result = values.sort()`? Why?” Give a focused clue if needed. |
| **Contrastive feedback** | Compare `values.sort()` (mutates `values`, returns `None`) with `sorted(values)` (returns a new sorted list), and explain the different behaviors directly. |

Strategies are stored in lesson history and kept consistent during a follow-up exchange.
The controller changes strategy only when the evidence supports it:

1. If an error signal repeats in at least two of the latest three assessments for a
   topic, it prioritizes contrastive feedback unless that was already the latest
   lesson's strategy.
2. If at least two assessments associated with the current strategy average below
   `60%`, it advances to the next strategy in the configured six-strategy cycle.
3. Otherwise, it keeps the current strategy.

These are rule-based adaptation thresholds, not a claim that the strategy is
guaranteed to improve learning. The re-test flow provides a way to measure the
student's observed before/after accuracy.

## Learner data and persistence

### Student Knowledge Base

[models/skb_schema.py](./models/skb_schema.py) defines the Pydantic schema. The
`data/skb_database.json` file stores a JSON list of student records. Important fields
include:

- `topics`: topic ID to mastery score.
- `skills`: subtopic ID to mastery score.
- `strengths` and `weaknesses`: topic IDs classified at the `0.7` threshold.
- `active_misconceptions`: misconception descriptions and counts from tagged answers.
- `diagnosis_responses`: answers to the initial diagnostic.
- `pedagogy_history`: lesson strategy events and topic-aligned assessment events.
- `learning_signals` and `learning_signal_descriptions`: recurring error signal counts
  and their descriptions.
- `learning_conversations`: bounded student turns, struggle flags, and answer
  verdicts.
- `root_cause_history`: saved quiz root-cause selections.
- `retests`: original and re-test quiz IDs, before/after accuracy, gain, and review
  turns.
- `chat_memory`: compact summaries of the latest exchanges.
- timestamps for record creation and updates.

Topic mastery after an aligned assessment uses an 80/20 update:

```text
new_topic_mastery = 0.8 * old_topic_mastery + 0.2 * quiz_accuracy
```

Subtopic mastery is updated per question using a learning rate of `0.3`:

```text
new_subtopic_mastery = old_mastery + 0.3 * (answer_outcome - old_mastery)
```

If a topic has no previous score, the assessment accuracy initializes it. Untested
subtopics use the topic score or a default score of `0.5`.

### Other stored data

- `data/evaluation/consent.jsonl`: consent records.
- `data/evaluation/labels.jsonl`: student-provided explanations of quiz mistakes.
- `data/evaluation/feedback.jsonl`: session ratings and optional comments.
- `data/evaluation/expert_labels.jsonl`: expert labels imported for evaluation.
- `data/traces/`: JSONL agent-step traces, one file per graph thread.
- `data/backups/`: backups of the SKB, traces, and evaluation data.

The evaluation store appends one JSON record per line. The app backs up study data at
startup and retains the newest 30 backup directories. The project `.gitignore`
excludes traces, evaluation data, backups, Python caches, and virtual environments.

## Data assets

| Path | Contents and use |
|---|---|
| `data/common/topics.json` | 13 topic records, including descriptions, prerequisites, subtopics, and misconception catalog entries |
| `data/common/mcqs_list.json` | 455 practice MCQs with options, correct option, difficulty, topic/subtopic tags, and explanations |
| `data/diagnosis/static_diagnosis_mcq_list.json` | 26 fixed onboarding questions |
| `data/common/corpus.jsonl` | 430 labeled course-content chunks across the 13 topics; currently not wired into the tutor prompt |
| `data/skb_database.json` | Persisted student SKBs; treat as study data, not source code |
| `data/skb_database.backup.json` | Additional SKB backup file present in the repository |
| `data/evaluation/` | Study records and generated evaluation outputs |
| `data/traces/` | Runtime agent traces, normally ignored by Git |
| `data/backups/` | Timestamped app-start and other backups, normally ignored by Git |

The current quiz data generally does not include option-level misconception tags.
When absent, quiz error signals use related subtopics/topics as proxies; they are not
proof that a student holds a specific misconception. The tagger script can add
option-level annotations, but those annotations should be reviewed before being used
as ground truth.

## File-by-file guide

### Application and workflow

| File | Responsibility |
|---|---|
| [app.py](./app.py) | Streamlit UI: student profile and study controls, consent, chat, diagnostic/quiz forms, results, root-cause comparison, remediation, re-test, and feedback |
| [agents/workflow.py](./agents/workflow.py) | Defines LangGraph state and graph nodes; runs diagnosis, tutoring, quiz grading, error analysis, root-cause comparison, persistence, interrupts/resume, and trace output |
| [agents/orchestrator.py](./agents/orchestrator.py) | Reads SKBs, normalizes route labels, detects follow-ups, selects topics, applies forced routes, and routes requests to graph nodes |
| [agents/llm_factory.py](./agents/llm_factory.py) | Configures the local OpenAI-compatible MLX client and per-task generation profiles |

### Learning components

| File | Responsibility |
|---|---|
| [agents/diagnosis_agent.py](./agents/diagnosis_agent.py) | Creates an SKB from the fixed initial diagnostic |
| [agents/tutor_agent.py](./agents/tutor_agent.py) | Builds personalized teaching prompts, handles tutor-question follow-ups, and records lessons/learning turns |
| [agents/assessment_agent.py](./agents/assessment_agent.py) | Chooses and introduces adaptive MCQs; workflow.py grades submitted answers |
| [agents/pedagogy_controller.py](./agents/pedagogy_controller.py) | Selects strategies, records lesson/assessment events, tracks signals, and updates topic/subtopic mastery |
| [agents/error_analysis_agent.py](./agents/error_analysis_agent.py) | Calculates accuracy and emits misconception/subtopic error signals |
| [agents/root_cause_agent.py](./agents/root_cause_agent.py) | Implements quiz root-cause methods, candidate evidence, configurable weights, and comparisons |
| [agents/turn_judge.py](./agents/turn_judge.py) | Classifies answer/new-question/other, verdict, confusion, and hint request; has a rule fallback |
| [agents/struggle_tracker.py](./agents/struggle_tracker.py) | Keyword-based topic/subtopic matching and learning-struggle evidence |
| [agents/topic_classifier.py](./agents/topic_classifier.py) | Optional semantic topic/subtopic classification using local embeddings or TF-IDF |
| [agents/chat_memory.py](./agents/chat_memory.py) | Compact summaries of recent student/tutor exchanges for subsequent prompts |
| [agents/course_retriever.py](./agents/course_retriever.py) | Standalone TF-IDF corpus retriever; not currently called by the tutor |
| [models/skb_schema.py](./models/skb_schema.py) | Pydantic model and defaults for Student Knowledge Base records |

### Study and evaluation utilities

| File | Responsibility |
|---|---|
| [agents/evaluation_store.py](./agents/evaluation_store.py) | Appends and reads study JSONL records, finds IDs, and makes bounded backups |
| [scripts/eval_data.py](./scripts/eval_data.py) | Loads student records, merges per-topic assessment events into quiz records, and combines student/expert labels |
| [scripts/evaluate.py](./scripts/evaluate.py) | Generates a Markdown report and CSV tables for root-cause accuracy/agreement, misconception checks, learning gain, judge-vs-rules, latency, and feedback |
| [scripts/fit_weights.py](./scripts/fit_weights.py) | Searches a grid of hypothesis weights, evaluates leave-one-student-out accuracy, and optionally writes `learned_weights.json` |
| [scripts/tag_mcq_misconceptions.py](./scripts/tag_mcq_misconceptions.py) | Tags wrong answer options using the local LLM or a weaker lexical method and writes a review CSV |
| [tests/test_llm_factory.py](./tests/test_llm_factory.py) | Unit test for generation continuation and combined usage metadata |

### Project documentation and assets

| File | Responsibility |
|---|---|
| [README.md](./README.md) | Existing setup, launch, usage, and study instructions |
| [project_details.md](./project_details.md) | This detailed architectural and implementation guide |
| [requirements.txt](./requirements.txt) | Python package requirements |
| [part1_flow.jpeg](./part1_flow.jpeg) | High-level architecture illustration; some pictured RAG behavior is not yet wired into the tutor |
| [mid_report.pdf](./mid_report.pdf) | Project report document |
| [.gitignore](./.gitignore) | Excludes local environments, caches, traces, evaluation data, and backups |

## Study and evaluation workflow

### Collecting a study session

1. Enable Study mode and create an anonymous student ID.
2. Obtain consent through the UI.
3. Complete diagnosis, tutoring, and quizzes.
4. After a quiz, optionally fill in the “why did you get these wrong?” form. This
   provides student-reported labels for evaluating the system's error analysis.
5. Use **Review** to study the predicted root cause, then **Re-test** to measure
   accuracy on that subtopic after remediation.
6. Submit optional end-of-session ratings.

The app's A/B setting can assign one of the three root-cause methods by student ID.
Method, preset, weights, candidate features, and comparisons are stored with quiz
assessment events so that results can be revisited.

### Analysis scripts

Run these from the project root after collecting study records:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --export-labeling
python scripts/evaluate.py --import-labels data/evaluation/labeling_sheet.csv
python scripts/fit_weights.py
python scripts/fit_weights.py --save
```

`evaluate.py` writes a Markdown report plus CSV tables under
`data/evaluation/report/`. It prefers expert labels when present and otherwise uses
student-provided labels. `fit_weights.py --save` writes fitted weights for the app to
load after restart; its leave-one-student-out score is more informative than training
accuracy when the dataset is small.

The misconception tagger supports a dry run and review-before-use workflow:

```bash
python scripts/tag_mcq_misconceptions.py --limit 10 --dry-run
python scripts/tag_mcq_misconceptions.py --method lexical --dry-run
```

The LLM tagging method uses the local `judge` profile. The lexical method is weaker
and should not be treated as an authoritative label. A normal tagging run backs up
the question bank before writing tags.

## Setup and running

The supported local backend is MLX-LM on Apple Silicon. Use Python 3.11 or 3.12,
install requirements in a virtual environment, and run the inference server separately
from Streamlit:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the model is not fully cached, download it while online:

```bash
hf download Qwen/Qwen3-4B-MLX-4bit
```

Start the model server in one terminal:

```bash
source .venv/bin/activate
export TUTOR_MODEL_NAME="Qwen/Qwen3-4B-MLX-4bit"
HF_HUB_OFFLINE=1 mlx_lm.server --model "$TUTOR_MODEL_NAME" --host 127.0.0.1 --port 8080
```

Then start the UI in another terminal:

```bash
cd /path/to/LMA_Group_Project
source .venv/bin/activate
python -m streamlit run app.py
```

Open the Streamlit URL it prints, usually `http://localhost:8501`. For the complete
installation steps and local troubleshooting, see [README.md](./README.md).

### Current-machine readiness note

During the recent local inspection, the available system interpreter was Python 3.9.6,
and the MLX model cache contained repository metadata but not the referenced model
weight blob. This machine therefore needs Python 3.11/3.12 and a complete model
download before it can serve the model fully offline.

## Known limitations and implementation notes

- The tutor's course-corpus retriever is implemented but not currently called by the
  tutor node; the present runtime is not an active RAG pipeline.
- There is no model-issued tool/function calling. Agent coordination is application-
  controlled through LangGraph nodes and commands.
- Quiz question selection and grading are local code over a fixed bank; the LLM
  generates the quiz introduction, not the bank questions.
- When option-level misconception annotations are absent, related topics/subtopics
  are proxies for error signals. Do not describe those proxies as confirmed student
  beliefs.
- The topic classifier can use TF-IDF when the optional embedding model is unavailable.
  Without scikit-learn, classification may be disabled and local keyword rules still
  provide a fallback for some routing decisions.
- Graph checkpoints use in-memory storage and do not persist across app-process
  restarts. The SKB and JSONL files persist learner/study records separately.
- The app stores study messages, quiz answers, and analysis under anonymous IDs.
  Avoid entering personally identifying information; handle the SKB, traces, consent,
  labels, and feedback files as study data.
