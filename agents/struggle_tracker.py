"""Track how much a student struggled while learning a topic with the tutor.

Every student message sent to the tutor is logged in the knowledge base as a "learning turn",
tagged with the topic, the subtopics it mentions and two struggle flags (confusion, hint request).
`learning_struggle()` later turns those turns into the LS (Learning Struggle) parameter used by
the root-cause agent:

    LS(H) = average(CF, RA, TR, HR, WS, SW)          each scaled to 0..1

    CF  Confusion phrases     share of turns that say "I don't get it", "still not working", ...
    RA  Re-asking             the student asked about this subtopic 2+ times
    TR  Turns to resolve      many back-and-forth turns were needed on the topic
    HR  Hint requests         share of turns asking for a hint or the answer
    WS  Wrong self-explanation   after a hints/self-explanation lesson the student was still
                                 confused or asked for the answer (a rule-based proxy)
    SW  Strategy switches     the pedagogy controller had to change strategy on the topic

The detection is rule-based (regular expressions and keyword overlap) so it is fast,
deterministic and testable without the LLM.
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

TOPICS_FILEPATH = "data/common/topics.json"
MAX_LOGGED_TURNS = 200

CONFUSION_PATTERN = re.compile(
    r"\b(confus\w*|don'?t (get|understand)|do not (get|understand)|didn'?t (get|understand)|"
    r"not sure|i'?m lost|stuck|makes? no sense|doesn'?t make sense|can'?t understand|"
    r"cannot understand|not following|what do you mean|still (not|don'?t|confused|wrong|"
    r"getting|failing)|not working|doesn'?t work|isn'?t working|why (is|does|do|did) .{0,40}"
    r"(empty|error|wrong|fail|crash|not|none|erased|missing))\b",
    re.IGNORECASE,
)
HINT_PATTERN = re.compile(
    r"\b(hint|just (tell|give|show) me|(tell|give|show) me the (answer|solution|code)|"
    r"what'?s the answer|what is the answer|solve it for me|do it for me)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "and", "the", "for", "with", "how", "what", "are", "vs", "its", "into", "from", "that",
    "this", "their", "basic", "using", "use", "common", "python",
}
# Strategies that ask the student to explain or reason first; a confused reply after one of
# them counts as a failed self-explanation (WS).
SELF_EXPLANATION_STRATEGIES = {"self_explanation", "hints"}


def stem(word: str) -> str:
    """A tiny stemmer so that "files"/"file", "modes"/"mode" and "closing"/"close" match."""
    for suffix in ("ing", "ed", "s"):
        if len(word) > 4 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> set:
    words = re.findall(r"[a-z_]+", str(text or "").lower())
    return {stem(word) for word in words if len(word) >= 3 and word not in STOPWORDS}


@lru_cache(maxsize=4)
def load_catalog(topics_filepath: str = TOPICS_FILEPATH) -> dict:
    """Subtopic names, the topic of every subtopic, topic prerequisites and per-topic keywords."""
    with open(topics_filepath, "r", encoding="utf-8") as topic_file:
        topics = json.load(topic_file)

    subtopic_names, subtopic_topic, prerequisites, topic_subtopics = {}, {}, {}, {}
    for topic in topics:
        topic_id = topic["topic_id"]
        prerequisites[topic_id] = list(topic.get("prerequisites", []))
        topic_subtopics[topic_id] = []
        for subtopic in topic.get("subconcepts", []):
            subtopic_names[subtopic["id"]] = subtopic["name"]
            subtopic_topic[subtopic["id"]] = topic_id
            topic_subtopics[topic_id].append(subtopic["id"])

    # Keywords that tell the subtopics of one topic apart: drop words shared by most of them
    # (e.g. "file" in every File Handling subtopic).
    distinctive_keywords = {}
    for topic_id, subtopic_ids in topic_subtopics.items():
        token_sets = {sid: tokenize(subtopic_names[sid]) for sid in subtopic_ids}
        for sid, tokens in token_sets.items():
            distinctive_keywords[sid] = {
                token for token in tokens
                if sum(token in other for other in token_sets.values()) <= max(1, (len(token_sets) - 1) // 2)
            }

    return {
        "subtopic_names": subtopic_names,
        "subtopic_topic": subtopic_topic,
        "prerequisites": prerequisites,
        "topic_subtopics": topic_subtopics,
        "subtopic_keywords": {sid: tokenize(name) for sid, name in subtopic_names.items()},
        "distinctive_keywords": distinctive_keywords,
    }


def detect_struggle(text: str) -> Dict[str, bool]:
    return {
        "confusion": bool(CONFUSION_PATTERN.search(text or "")),
        "hint_request": bool(HINT_PATTERN.search(text or "")),
    }


def map_subtopics(text: str, topic_id: str, topics_filepath: str = TOPICS_FILEPATH) -> List[str]:
    """Subtopics of `topic_id` whose distinctive keywords appear in the message.
    An empty list means the message is about the topic in general."""
    catalog = load_catalog(topics_filepath)
    tokens = tokenize(text)
    return [
        sid for sid in catalog["topic_subtopics"].get(topic_id, [])
        if tokens & catalog["distinctive_keywords"].get(sid, set())
    ]


def record_learning_turn(
    student_kb: dict,
    topic_id: str,
    student_text: str,
    previous_strategy: Optional[str] = None,
    topics_filepath: str = TOPICS_FILEPATH,
) -> dict:
    """Log one student message sent to the tutor, with its struggle flags."""
    updated = deepcopy(student_kb)
    if not str(student_text or "").strip():
        return updated
    flags = detect_struggle(student_text)
    turns = updated.setdefault("learning_conversations", [])
    turns.append({
        "topic_id": topic_id,
        "subtopic_ids": map_subtopics(student_text, topic_id, topics_filepath),
        "text": str(student_text).strip()[:300],
        "confusion": flags["confusion"],
        "hint_request": flags["hint_request"],
        "after_strategy": previous_strategy,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    updated["learning_conversations"] = turns[-MAX_LOGGED_TURNS:]
    return updated


def strategy_switches(student_kb: Optional[dict], topic_id: str) -> int:
    strategies = [
        event.get("strategy")
        for event in (student_kb or {}).get("pedagogy_history", [])
        if event.get("event") == "lesson" and event.get("topic_id") == topic_id
    ]
    return sum(1 for before, after in zip(strategies, strategies[1:]) if before != after)


def learning_struggle(
    student_kb: Optional[dict],
    subtopic_id: str,
    topics_filepath: str = TOPICS_FILEPATH,
) -> dict:
    """Compute the LS parameter and its six components for one subtopic."""
    catalog = load_catalog(topics_filepath)
    topic_id = catalog["subtopic_topic"].get(subtopic_id, subtopic_id.split(".")[0])
    turns = [
        turn for turn in (student_kb or {}).get("learning_conversations", [])
        if turn.get("topic_id") == topic_id
        and (not turn.get("subtopic_ids") or subtopic_id in turn["subtopic_ids"])
    ]
    explicit_turns = [turn for turn in turns if subtopic_id in turn.get("subtopic_ids", [])]
    eligible = [turn for turn in turns if turn.get("after_strategy") in SELF_EXPLANATION_STRATEGIES]

    count = len(turns)
    components = {
        "CF": sum(turn.get("confusion", False) for turn in turns) / count if count else 0.0,
        "RA": min(1.0, max(0, len(explicit_turns) - 1) / 2),
        "TR": min(1.0, max(0, count - 2) / 4),
        "HR": sum(turn.get("hint_request", False) for turn in turns) / count if count else 0.0,
        "WS": (
            sum(turn.get("confusion", False) or turn.get("hint_request", False) for turn in eligible)
            / len(eligible)
            if eligible else 0.0
        ),
        "SW": min(1.0, strategy_switches(student_kb, topic_id) / 2),
    }
    components = {key: round(value, 2) for key, value in components.items()}
    return {
        "LS": round(sum(components.values()) / len(components), 2),
        "components": components,
        "turns": count,
        "examples": [turn["text"] for turn in turns if turn.get("confusion") or turn.get("hint_request")][-2:],
    }
