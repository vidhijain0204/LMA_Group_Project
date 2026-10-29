# Python Tutor MCQ Test Set

This package contains **455 Python-only multiple-choice questions** for testing and evaluating the adaptive Python tutor.

## Files

- `python_tutor_mcqs.json` — the question bank (a JSON array with one question per object).
- `metadata.json` — schema notes, taxonomy alignment, difficulty/type distributions, topic coverage, and validation summary.
- `README.md` — this guide.

## Question schema

Every question contains:

- `prob_id`: unique identifier such as `PY0001`.
- `prob_body`: the question prompt, sometimes with a short Python snippet.
- `type`: item format, such as `output_prediction`, `debugging`, `code_completion`, `conceptual`, `best_practice`, `application`, `error_identification`, or `edge_case`.
- `options`: four choices in an object labelled `A`, `B`, `C`, and `D`.
- `correct_option`: the label of the correct choice.
- `difficulty`: one of `0`, `0.2`, `0.4`, `0.6`, `0.8`, or `1`.
- `related_topics`: at least two topic IDs. The first entry is the primary topic; the next entry links a prerequisite or related skill for SKB-aware recommendation.
- `related_subtopics`: subconcept IDs aligned with the supplied `topics.json` file. The first entry is the primary subconcept.
- `explanations`: a short rationale for the correct answer.

## Difficulty scale

| Value | Intended level | Typical item |
|---:|---|---|
| `0` | Very easy | Direct recall of one foundational fact |
| `0.2` | Easy | One-step application |
| `0.4` | Lower intermediate | Short code tracing or selecting a correct construct |
| `0.6` | Intermediate | Multi-step tracing or debugging |
| `0.8` | Hard | Subtle semantics and edge cases |
| `1` | Very hard | More involved reasoning across interacting Python features |

Treat these values as an initial calibration, not a validated psychometric measure. You can update them using student performance and item analysis.

## Topic coverage

The set has 35 questions for each of the 13 primary topics in `topics.json`, covering all 74 subconcepts in the taxonomy. Questions are cross-tagged with a second related topic to support prerequisite-aware teaching and SKB-based routing. The source instructional corpus has uneven numbers of chunks by topic, so the MCQs are balanced to the taxonomy rather than to the corpus chunk frequency.

## Example record

```json
{
  "prob_id": "PY0001",
  "prob_body": "Which component executes Python source code statement by statement?",
  "type": "conceptual",
  "options": {
    "A": "The Python interpreter",
    "B": "The package installer",
    "C": "The database engine",
    "D": "The HTML renderer"
  },
  "correct_option": "A",
  "difficulty": 0.0,
  "related_topics": [
    "T01",
    "T02"
  ],
  "related_subtopics": [
    "T01.1",
    "T02.1"
  ],
  "explanations": "The interpreter reads and executes Python code; a package installer manages installation of packages."
}
```

This is the first record from the question file, with its actual assigned ID and option order.

## Using with the SKB

A simple initial policy could use the student's mastery estimate for a question's primary topic and relevant prerequisites to choose a target difficulty. For example, low mastery can start around `0`–`0.4`, while consistently high mastery can unlock `0.6`–`1`. Consider using both topic-level mastery and recent correctness rather than a single overall score. Make the thresholds configurable and validate them using held-out evaluation.

A question's `related_topics` is a list of IDs, not mastery values. Resolve IDs through `topics.json`. The first topic is primary; subsequent entries are contextual links. `related_subtopics` stores subconcept IDs for finer-grained analysis.

## Validation and limitations

The generation script checks that the JSON parses, every ID is unique, every question has four distinct options, the answer label exists, difficulty values are from the allowed scale, and topic/subtopic tags reference the supplied taxonomy. The item bank is intended as a **development/testing set**, not a standardized exam. Run a human review pass before grading students, examine item difficulty after collecting response data, and avoid using the same questions for both policy tuning and final evaluation if an unbiased final score matters.

The questions were newly authored to align with the supplied topic taxonomy; the instructional corpus was used for topic alignment and coverage planning, not copied verbatim into quiz prompts.
