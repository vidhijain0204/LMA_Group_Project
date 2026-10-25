# Generated Dataset Guide

This folder contains one reproducible, synthetic dataset for developing and
evaluating a multi-agent Python programming tutor. The companion
[project README](../README.md) documents how to generate datasets and
describes the generator in more detail.

> **Important:** students, responses, misconceptions, and mastery values in
> this dataset are simulated. They are useful for development and controlled
> experiments, but are not evidence of real learner behavior or tutoring
> effectiveness.

## What's in this generated run

The dataset was generated in template mode with seed `42`. It contains:

- **100 accepted Python questions**, covering all 27 skills.
- **20 synthetic learner profiles**.
- **1,000 interaction attempts** across those learners.
- A **27-skill prerequisite graph** with 51 prerequisite edges.
- **885 learning-sequence rows**, which track question introductions rather
  than attempts.
- No rejected questions and no shortfall from the requested question count.

Question types include single- and multiple-answer multiple choice, code
tracing, fill-in-the-blank, debugging, short answer, and coding. The
`dataset_metadata.json`, `question_generation_report.json`, and
`validation_report.json` files are the source of truth for this run's
configuration, counts, and validation results.

## Files and intended use

| File | Contents | Intended use |
|---|---|---|
| `AgentVisibleProblems.csv` | Student-facing prompt, type, options, difficulty, skill links, and estimated time | Provide the current question to tutor/assessment components. |
| `AgentVisibleInteractions.csv` | Learner responses, scored outcomes, hints, timing, and attempt metadata | Build a learner's past interaction history after filtering it to the current decision time. |
| `PythonSkills.csv` | Skill names, descriptions, categories, levels, and prerequisite IDs | Explain skills and interpret evidence at the skill level. |
| `SkillPrerequisites.csv` | Directed prerequisite edges | Reason about prerequisite gaps and order learning goals. |
| `LearningSequence.csv` | Per-learner question introductions, sequence index, skill, and introduction time | Scheduling and offline analysis; do not expose future rows to an agent. |
| `PythonProblems.csv` | Full question records, including keys, reference solutions, explanations, tests, and misconception tags | Trusted assessment service and offline evaluation only. Never put this file in an LLM agent's context. |
| `PythonInteractions.csv` | Full interaction records, including simulator error and misconception labels | Offline evaluation only. Use the agent-visible version for agent context. |
| `Students.csv` | Starting mastery vectors and simulator parameters | Simulator analysis only; these values reveal synthetic ground truth. |
| `SkillMasteryHistory.csv` | Per-learner, per-skill hidden mastery snapshots | Offline diagnosis/assessment evaluation only; never use as an agent input or a prediction feature. |
| `rejected_questions.csv` | Candidate question IDs and rejection reasons | Generation audit. It is empty for this run. |
| `question_generation_report.json` | Question generation counts, type/difficulty distribution, and coverage | Dataset audit and experiment reporting. |
| `validation_report.json` | Schema, reference, chronology, and leakage validation results | Check dataset integrity before use. |
| `dataset_metadata.json` | Dataset version, seed, configuration, file boundaries, and overall counts | Reproduce and identify the run. |

CSV columns containing arrays or objects are JSON-encoded strings; parse them
as JSON rather than treating them as Python literals. CSV booleans and numeric
values are serialized text and should be converted to the appropriate types
when loaded.

## Recommended multi-agent workflow

The dataset supplies agent inputs and evaluation targets; it does **not**
implement the agents. Keep the agents coordinated through a shared,
time-bounded learner state rather than giving every agent every file.

### 1. Diagnosis agent

**Purpose:** infer likely skill strengths and gaps from observed work, and
identify which skills or misconceptions need further evidence.

**Agent context:**

- The learner's past rows from `AgentVisibleInteractions.csv`.
- The problems associated with those rows from `AgentVisibleProblems.csv`.
- Skill descriptions and the prerequisite graph from `PythonSkills.csv` and
  `SkillPrerequisites.csv`.
- The current question and its skill references, if diagnosis is part of an
  in-progress turn.

**Expected output:** evidence-linked estimates, confidence/uncertainty, and
one or more skills to investigate or practice. Treat a wrong answer as
evidence—not proof—of a misconception. Do not read simulator labels or hidden
mastery values.

### 2. Tutor agent

**Purpose:** teach the current concept, select an appropriate next activity,
and respond to the learner's work with useful hints or explanations.

**Agent context:**

- The current student-facing problem from `AgentVisibleProblems.csv`.
- The skills and prerequisites relevant to that problem.
- The Diagnosis agent's evidence-based summary.
- A filtered history of previous attempts, if needed for personalization.

**Expected output:** a supportive, targeted hint or explanation, or a
recommended next problem/skill. Adapt difficulty and revisit prerequisites
when the diagnosis is uncertain or indicates a gap. Do not expose answer keys
or reference solutions as part of ordinary tutoring; use a trusted grading
service for correctness checks.

### 3. Assessment agent

**Purpose:** evaluate a submitted response, summarize demonstrated skills,
and measure progress without confusing coached performance with independent
mastery.

**Agent context:**

- The submitted response and current problem.
- Objective scoring results returned by a trusted evaluator.
- The relevant skill definitions and prerequisite relationships.
- Past agent-visible attempts filtered to the assessment time.

**Expected output:** a score or rubric result, a concise explanation of the
evidence, and skill-level estimates with appropriate uncertainty. Distinguish
independent attempts from responses after hints or answer exposure using
`independent_attempt`, `used_hint`, `hint_count`, and `saw_answer`. A correct
response after seeing an answer is not equivalent evidence to an independent
correct response.

For objective grading, the assessment service may look up the correct answer
or run the hidden test cases from `PythonProblems.csv` in a trusted,
non-agent-facing process. Return only the necessary result to the agents; do
not send answer keys, reference solutions, or hidden tests to the diagnosis or
tutor agent.

## Keep each decision leak-free

The agent-visible files are safer for prompts, but their full CSVs still
contain future events. For every decision:

1. Select only the current learner's interaction rows.
2. Keep only events whose `end_time` is at or before the decision timestamp.
3. Join only the problem and skill information needed for those events and
   the current question.
4. Do not include future learning-sequence rows, hidden mastery snapshots,
   simulator profiles, or simulator-generated labels.

The project provides `utils.validation.agent_history_as_of(rows, user_id,
timestamp)` for filtering the interaction history to one learner and cutoff.
For a decision immediately before a question begins, use that question's
`start_time` as the cutoff. The helper sorts the retained rows chronologically
and removes simulator-only annotation columns. Apply the same cutoff rule to
any separately loaded sequence or other event data.

In particular:

- `PythonInteractions.csv` contains `error_type`, `misconception_tag`, and
  `scoring_method`; these are simulator annotations and must not be agent
  evidence.
- `PythonProblems.csv` contains answer keys, reference solutions, expected
  outputs, and test cases; exposing these lets an agent bypass assessment.
- `Students.csv` and `SkillMasteryHistory.csv` expose the simulator's latent
  learner state and must be reserved for offline evaluation.
- `discrete_score` in agent-visible history is the observable scored outcome
  of a past attempt, not the hidden continuous mastery target.

Use separate storage permissions or service boundaries if the agents, grader,
and evaluation code share a database; separating filenames alone is not an
access-control boundary.

## Interpreting the data

- `problem_id` links a response to a question; `user_id` identifies a
  simulated learner; and `id` identifies an interaction attempt.
- `primary_skill_id` is the main skill targeted. `secondary_skill_ids` may
  identify additional skills involved. Prerequisite edges point from a skill
  to one of its prerequisites.
- `start_time` and `end_time` delimit an attempt. `introduced_at` and
  `sequence_index` describe when an item entered the learner's sequence.
- `discrete_score` is the objective outcome for supported question types;
  `test_pass_rate`, `tests_passed`, and `tests_total` provide more detail for
  coding attempts. Empty values can mean a field does not apply.
- `independent_attempt`, hint fields, and `saw_answer` help distinguish
  unaided evidence from coached or answer-exposed performance.
- In `SkillMasteryHistory.csv`, `mastery_probability` and `true_mastery` are
  simulator-generated evaluation targets. The binary label uses the
  `0.70` threshold recorded in `dataset_metadata.json`.

The learner behavior and mastery updates follow the simulator's assumptions,
not a model fitted to real students. Compare agent performance against
appropriate baselines, keep hidden labels out of agent inputs, and avoid
claiming causal learning gains from this synthetic dataset alone.

## Regenerate or validate

From the `adaptive_python_tutor/` directory, regenerate the dataset with its
current run settings:

```bash
python build_dataset.py \
  --output-dir data/generated \
  --num-students 20 \
  --num-questions 100 \
  --interactions-per-student 50 \
  --seed 42
```

Run the dataset pipeline tests from the same directory with:

```bash
python -m unittest discover -s tests -v
```

See the [project README](../README.md) for configuration options, LLM
question generation, validation details, and simulator limitations.
