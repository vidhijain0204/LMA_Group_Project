# Adaptive Python Tutor Agent — Synthetic Dataset Generator

A modular, deterministic Python pipeline for generating Python programming questions, a prerequisite skill graph, synthetic student profiles, chronological interactions, and hidden mastery trajectories. It does **not** implement the Diagnosis, Tutor, or Assessment agents.

> **Research caveat:** learners, scores, misconceptions, learning events, and mastery states are simulated. The dataset can test an agent against the simulator's assumptions; it cannot establish real-world educational effectiveness.

## 1. Quick start

Python 3.10+ is recommended. Template generation and simulation use the standard library; PyYAML is required only to read YAML configuration files.

```bash
cd adaptive_python_tutor
python -m pip install -r requirements.txt
python build_dataset.py --output-dir data/generated --num-students 20 --num-questions 100 --interactions-per-student 50 --seed 42
python -m unittest discover -s tests -v
```

A small generated example is committed under `data/example/`. It contains 81 accepted questions and 120 interaction events for 6 synthetic learners. The machine-readable report is `data/example/validation_report.json`.

### Larger dataset

```bash
python build_dataset.py \
  --output-dir data/run_001 \
  --num-students 100 \
  --num-questions 250 \
  --interactions-per-student 75 \
  --max-attempts 2 \
  --seed 42
```

The template bank is finite: a larger request can yield fewer accepted questions when candidate bodies repeat. This is not hidden—the CLI logs a warning and `question_generation_report.json` records requested, generated, rejected, accepted, and shortfall counts. Use LLM mode with a configured endpoint for broader novelty, then inspect and validate every generated item.

Output paths are resolved relative to the current working directory. Non-empty output directories are not overwritten unless `--overwrite` is passed, and even then the directory must contain this tool's `dataset_metadata.json` marker. Prefer a new output directory for each experiment.

### YAML configuration

```bash
python build_dataset.py --config configs/default.yaml
python build_dataset.py --config configs/default.yaml --num-students 100 --output-dir data/run_002
```

Command-line values override the configuration file. `--help` lists the available options. You can restrict coverage by comma-separated skill IDs, categories, or exact skill names and tune the difficulty weights:

```bash
python build_dataset.py \
  --output-dir data/control_flow_run \
  --skill-coverage SK001,SK002,SK004,SK005,SK008,SK009 \
  --difficulty-distribution beginner:0.3,intermediate:0.5,advanced:0.2 \
  --minimum-prerequisite-coverage 0.5 \
  --num-students 40 --num-questions 120 --interactions-per-student 40
```

A question's difficulty cannot be below its skill's curriculum level. Therefore the reported `difficulty_counts` may differ from the requested weights when the requested distribution is infeasible for the selected skills; prerequisite coverage is also checked explicitly.

## 2. Optional LLM question generation

The LLM backend uses an OpenAI-compatible `/chat/completions` endpoint via the standard library; there is no provider SDK dependency. Configure the URL, model, timeout, retry limit, batch size, and the **name** of an environment variable holding the API key. Do not write secret keys to config files.

```bash
export PYTUTOR_LLM_API_KEY="<your-key>"
python build_dataset.py \
  --generation-mode llm \
  --llm-base-url "https://your-compatible-endpoint/v1" \
  --llm-model "your-configured-model" \
  --llm-api-key-env PYTUTOR_LLM_API_KEY \
  --num-questions 100 \
  --output-dir data/llm_run
```

The model must return the requested structured fields. Questions pass schema checks and executable-answer checks before acceptance. Missing credentials, invalid records, duplicated question bodies, failed batches, and code-validation failures are reported; LLM mode does **not** silently fall back to templates. Template mode is recommended for repeatable tests and can run with no API key or network.

The code-runner subprocess timeout limits accidental hangs but is **not a security sandbox**. Only use an LLM endpoint and execution environment you trust when enabling LLM-generated code. Do not run generated source with elevated privileges.

## 3. Output files and visibility boundaries

| File | Purpose | Feed directly to an agent? |
|---|---|---|
| `AgentVisibleProblems.csv` | Student-facing question text, options, difficulty, and skill references | Yes, for the current question only |
| `AgentVisibleInteractions.csv` | Student responses, objectively scored past outcomes, hints, timestamps, and response times | Yes, after filtering to the current decision time |
| `PythonSkills.csv` | Skill descriptions and prerequisite IDs | Yes |
| `SkillPrerequisites.csv` | Canonical prerequisite edge list | Yes |
| `PythonProblems.csv` | Full question bank, answer keys, reference solutions, tests, and response templates | **No — answer-key leakage** |
| `PythonInteractions.csv` | Research interactions with simulator error/misconception annotations | **No — label leakage** |
| `Students.csv` | Hidden starting mastery and true learner parameters | **No — ground-truth leakage** |
| `SkillMasteryHistory.csv` | Hidden skill mastery snapshots after each event | **No — evaluation labels** |
| `LearningSequence.csv` | When a question was first introduced to a learner | Evaluation/simulation metadata; don't reveal future rows |
| `rejected_questions.csv` | Candidate IDs and reasons for rejection | Build-time audit artifact |
| `question_generation_report.json` | Requested/generated/accepted/rejected counts and type distribution | Build-time audit artifact |
| `validation_report.json` | Machine-readable pass/fail checks, counts, and errors | Build-time audit artifact |
| `dataset_metadata.json` | Seed, configuration, artifact boundaries, and dataset counts | Experiment metadata |

`AgentVisibleProblems.csv` excludes reference solutions, answer keys, fill-in answers, response templates, and misconception labels. `AgentVisibleInteractions.csv` excludes the simulator's `error_type`, `misconception_tag`, and `scoring_method` labels. The full interaction table can be used for offline analysis only.

Use `utils.validation.agent_history_as_of(rows, user_id, timestamp)` to create a user-specific history containing only events whose `end_time` is at or before the decision timestamp. For a decision immediately before a question, use that question's `start_time` as the cutoff. Never provide the entire chronological file to an agent that is supposed to act at an earlier time.

## 4. Schemas

### Question bank

`PythonProblems.csv` keeps the compatibility fields `problem_id`, `Problem Body`, `Problem Type`, `Multiple Choice Options`, `Multiple Choice Answers`, and `Fill-in Answers`, and adds `difficulty`, skill references, misconception tags, reference solutions, explanations, starter code, executable tests, expected output, answer key JSON, timing estimates, generation source, trace code, function name, and deterministic response templates.

Supported `Problem Type` values:

- `mcq_single`: exactly one correct option ID.
- `mcq_multi`: two or more correct option IDs.
- `code_trace`: exact standard-output prediction.
- `fill_blank`: normalized exact answer among accepted strings.
- `debugging`: submit the corrected line.
- `short_answer`: normalized exact match to a canonical response.
- `coding`: callable Python function evaluated against executable test cases.

CSV fields that hold lists or objects use JSON strings. Multiple-choice options are JSON arrays of `{"id":"A","text":"..."}` objects. Correct IDs are stored separately. Reference solutions and test cases are not placed into the agent-visible problem bank.

### Skills

`PythonSkills.csv` holds skill metadata and a JSON array of prerequisite IDs. `SkillPrerequisites.csv` is the canonical directed edge table. The 27-skill graph is checked for missing nodes, self-links, and cycles.

### Interaction history

`PythonInteractions.csv` preserves the compatibility columns `id`, `problem_id`, `hint_count`, `answer_text`, `saw_answer`, `discrete_score`, `end_time`, and `user_id`. It adds start time, attempt number, code submission, objective test pass rate, response time, hint status, selected option IDs, independent-attempt flag, sequence index, introduction time, and simulator annotations.

`discrete_score` is `0` or `1` for the supported objective scorers; empty is reserved for a genuinely unscored response. Multiple-choice responses are compared with answer IDs, trace output is compared with expected stdout, short answers use normalized exact matching, and coding submissions run through the item test cases. `test_pass_rate` records partial coding performance; `discrete_score` for coding is 1 only if all tests pass.

### Hidden mastery history

`SkillMasteryHistory.csv` stores one snapshot per user, skill, and event time:

- `mastery_probability`: continuous latent mastery in `[0, 1]`.
- `true_mastery`: binary label computed as `mastery_probability >= 0.70`.
- `observation_count`: number of direct scored observations of that skill so far.
- `learning_events_count`: number of direct/indirect learning updates applied to that skill.

`Students.csv` stores the starting mastery vector and simulator parameters. Both it and the mastery history are evaluation-only. No agent-visible table carries these hidden fields.

## 5. Simulation assumptions

The learner simulator is local and seeded. A question response probability combines 72% primary-skill mastery and 28% mean secondary-skill mastery. Difficulty, relevant misconception tendency, code/syntax susceptibility, and transfer difficulty reduce the ability term. Response consistency mixes this ability term with a question-type-specific guessing baseline. Hints, currently viewed answers, and prior worked-example exposure can raise response probability.

For reproducibility, the high-level response formula is:

```text
combined_mastery = 0.72 * primary_mastery + 0.28 * mean(secondary_mastery)
ability = clip(0.12 + 0.84 * combined_mastery
               - difficulty_penalty - misconception_penalty
               - syntax_penalty - transfer_penalty, 0.02, 0.98)
p_correct = clip(response_consistency * ability
                  + (1 - response_consistency) * guessing_baseline
                  + hint_effect + answer_exposure_effect + prior_example_effect,
                  0.02, 0.98)
```

The exact constants and scoring/update routines are in `generation/simulate_students.py` and `generation/update_mastery.py`. Mastery updates are incremental, bounded below 1, and affected by retention decay. A correct answer after seeing the solution gets a much smaller mastery increment than an independent correct response. Incorrect answers do not reset mastery. Secondary skills receive a smaller update than the primary skill.

Student archetypes cover broad prerequisite gaps, strong fundamentals with weaker control-flow/functions, syntax-sensitive responses, transfer gaps, uneven profiles, and advanced learners. These are designed populations, not calibrated representations of actual students. The simulator uses templates for common mistakes rather than arbitrary fabricated code. Question selection combines skill weakness, uncertainty, and exploration; it is not a learned tutor policy.

## 6. Validation and tests

The final validation report checks file/schema presence, unique IDs, skill graph structure, skill and problem references, MCQ answer-key shape, timestamps, discrete scores, question-introduction chronology, hidden mastery bounds, and leakage-safe agent-visible column sets. Question validation additionally compiles or executes reference solutions and coding test cases where applicable. Rejected candidates remain in `rejected_questions.csv` with a reason.

Run tests with:

```bash
python -m unittest discover -s tests -v
```

The test suite uses only the local template generator and needs neither GPU access nor an LLM key.

## 7. Reference dataset note

The specification referenced `Problems.csv`, `Skills.csv`, and `Interactions.csv` as schema examples. Those three files were not attached to this task or discoverable as exact file names in the available library, so the implementation preserves the interaction fields and compatibility-style problem fields stated in the specification rather than claiming to have inspected unseen original columns. If the reference files are supplied later, the column mapping can be checked against their actual headers and types without modifying those source files.

## 8. Limitations to keep in mind

- Template questions emphasize answerable snippets and compact coding tasks; expert review is still needed before use in a real course.
- Deduplication catches normalized exact question bodies within each question type, not semantic paraphrases or near-duplicates.
- `short_answer` and `fill_blank` scoring currently use normalized exact matching, not semantic equivalence.
- Misconception labels and mastery probabilities come from the simulator design and should be treated as synthetic targets.
- The simulator's probabilities and learning increments are engineering assumptions, not fitted parameters from learning-science data.
- The provided evaluation tables do not prove causal tutoring gains; a later experiment should use held-out questions, appropriate baselines, and explicit leakage controls.


## 9. Troubleshooting

- **YAML configuration fails to load:** install the optional dependency with `python -m pip install PyYAML`, or pass flat JSON configuration instead.
- **LLM mode fails before generation:** confirm the environment variable named by `--llm-api-key-env` is exported, and supply both `--llm-base-url` and `--llm-model`. The code never falls back to templates in LLM mode.
- **Fewer questions are accepted than requested:** read `question_generation_report.json` and `rejected_questions.csv`. The offline template bank uses finite patterns; for a larger varied bank, enable a configured LLM backend and review its generated questions.
- **A run produces no accepted questions:** the generator saves the rejection CSV, metadata, and a `FAIL` validation report before returning an error.
- **Output directory is not empty:** choose a new output path or use `--overwrite` only for a directory created by this tool and containing `dataset_metadata.json`.
