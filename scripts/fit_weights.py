"""Learn the hypothesis weights (LS, CE, RP, PW, PM) from labelled quizzes.

Why it exists: the hypothesis method of the root-cause agent scores each candidate subtopic
with 5 evidence values. By default every value gets the same weight (a plain average). This
script checks which weights would have predicted the true root cause best, using the quizzes
students labelled in the "why did you get these wrong?" form (or imported expert labels).

When to run it: after collecting a good number of labelled quizzes. It only reads data; it does
not need the LLM.

How it works: for every labelled quiz the stored candidates (their LS, CE, RP, PW, PM' values)
are re-scored with every weight combination on a grid (each weight a multiple of --step, all
summing to 1). The best combination is the one whose top candidate matches the true subtopic
most often (ties: higher average P(true subtopic)).

Because there are only a few students, the result is checked with leave-one-student-out
cross-validation: weights are fitted without one student and tested on that student's quizzes.
Report the cross-validated accuracy, not the training accuracy.

    python scripts/fit_weights.py            # fit and compare with the existing weight presets
    python scripts/fit_weights.py --save     # also save them (data/evaluation/learned_weights.json);
                                             # after a restart the app's "Custom weights" sliders
                                             # start from these values
"""

import argparse
import itertools
import json
import os
from datetime import datetime

import numpy as np

# eval_data must be imported first: it adds the project root to sys.path for the agents imports.
from eval_data import labels, quizzes, students, true_subtopic
from agents.root_cause_agent import LEARNED_WEIGHTS_FILE, PARAMS, SIGNS, WEIGHT_PRESETS


def dataset():
    """Turn the labelled quizzes into arrays for fast scoring.

    Returns (student id per quiz, params [quizzes, candidates, 5], index of the true candidate
    per quiz or -1 if the truth is not among them, quiz ids), or None if there is no data.
    Quizzes with fewer candidates are padded with NaN.
    """
    label_by_quiz = labels()
    rows = []
    for quiz in quizzes(students()):
        truth = true_subtopic(label_by_quiz.get(quiz["quiz_id"]))
        if not truth or not quiz.get("candidates"):
            continue
        rows.append((quiz["student_id"], quiz["quiz_id"], quiz["candidates"], truth))
    if not rows:
        return None
    width = max(len(r[2]) for r in rows)
    params = np.full((len(rows), width, len(PARAMS)), np.nan)
    truth_index = np.full(len(rows), -1)
    for q, (_, _, candidates, truth) in enumerate(rows):
        for c, cand in enumerate(candidates):
            params[q, c] = [cand["params"][p] for p in PARAMS]
            if cand["subtopic_id"] == truth:
                truth_index[q] = c
    return [r[0] for r in rows], params, truth_index, [r[1] for r in rows]


def weight_grid(step: float) -> np.ndarray:
    """Every set of 5 weights that are multiples of `step` and sum to 1 (one row per set)."""
    units = int(round(1 / step))
    grid = [
        [n * step for n in combo]
        for combo in itertools.product(range(units + 1), repeat=len(PARAMS)) if sum(combo) == units
    ]
    return np.array(grid)


def evaluate(weights: np.ndarray, params: np.ndarray, truth: np.ndarray):
    """Score every quiz with every weight set.

    Args: weights [W, 5], params [Q, C, 5], truth [Q] (index of the true candidate or -1).
    Returns: (top-1 accuracy per weight set, mean P(true candidate) per weight set).
    P is each candidate's positive score divided by the sum (equal split if all are <= 0).
    """
    signs = np.array([SIGNS[p] for p in PARAMS])
    scores = np.einsum("qcp,wp->wqc", np.nan_to_num(params, nan=0.0), weights * signs)
    scores = np.where(np.isnan(params[..., 0])[None], -np.inf, scores)      # padding never wins
    top = scores.argmax(axis=2)
    hits = (top == truth[None]) & (truth[None] >= 0)
    positive = np.clip(np.where(np.isfinite(scores), scores, 0), 0, None)
    totals = positive.sum(axis=2, keepdims=True)
    count = np.isfinite(scores).sum(axis=2, keepdims=True)
    probs = np.where(totals > 0, positive / np.where(totals > 0, totals, 1), 1 / count)
    p_truth = np.where(truth[None] >= 0, np.take_along_axis(probs, np.clip(truth, 0, None)[None, :, None], 2)[..., 0], 0)
    return hits.mean(axis=1), p_truth.mean(axis=1)


def best_weights(grid, params, truth):
    """The weight set with the best accuracy (ties: higher mean P(truth)), with both scores."""
    accuracy, p_truth = evaluate(grid, params, truth)
    best = np.lexsort((-p_truth, -accuracy))[0]
    return grid[best], accuracy[best], p_truth[best]


def main() -> None:
    """Fit on all data, cross-validate, compare with the presets and optionally save."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data = dataset()
    if not data:
        print("No labelled quizzes with stored candidates yet. Collect labels in the app (the 'why did you get these wrong?' form) "
              "or import expert labels with scripts/evaluate.py --import-labels.")
        return
    student_ids, params, truth, _ = data
    coverage = float((truth >= 0).mean())
    print(f"Labelled quizzes: {len(truth)} from {len(set(student_ids))} students. "
          f"True subtopic among the 3 candidates: {coverage:.0%} (upper bound for any weights).")

    grid = weight_grid(args.step)
    weights, accuracy, p_truth = best_weights(grid, params, truth)
    print(f"\nBest weights on all data: { {p: round(float(w), 2) for p, w in zip(PARAMS, weights)} }  "
          f"accuracy {accuracy:.0%}, mean P(true) {p_truth:.2f}")

    # Leave-one-student-out cross-validation.
    ids = np.array(student_ids)
    held_out_hits = []
    for student in sorted(set(student_ids)):
        train, test = ids != student, ids == student
        if not train.any():
            continue
        fold_weights, _, _ = best_weights(grid, params[train], truth[train])
        fold_accuracy, _ = evaluate(fold_weights[None], params[test], truth[test])
        held_out_hits.extend([fold_accuracy[0]] * int(test.sum()))
    cv_accuracy = float(np.mean(held_out_hits)) if held_out_hits else None
    if cv_accuracy is not None:
        print(f"Leave-one-student-out accuracy: {cv_accuracy:.0%}   <- report this one")

    print("\nOther weight sets on the same quizzes:")
    for name, preset in WEIGHT_PRESETS.items():
        acc, pt = evaluate(np.array([[preset[p] for p in PARAMS]]), params, truth)
        print(f"  {name:15s} accuracy {acc[0]:.0%}   mean P(true) {pt[0]:.2f}")

    if args.save:
        os.makedirs(os.path.dirname(LEARNED_WEIGHTS_FILE), exist_ok=True)
        with open(LEARNED_WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "weights": {p: round(float(w), 3) for p, w in zip(PARAMS, weights)},
                "train_accuracy": round(float(accuracy), 3),
                "loso_cv_accuracy": None if cv_accuracy is None else round(cv_accuracy, 3),
                "quizzes": int(len(truth)), "students": len(set(student_ids)),
                "step": args.step, "created_at": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
        print(f"\nSaved to {LEARNED_WEIGHTS_FILE}; restart the app; 'Custom weights' now starts from them.")


if __name__ == "__main__":
    main()
