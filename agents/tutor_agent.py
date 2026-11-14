import json
import re
from typing import List, Optional, Tuple
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from agents.course_retriever import CourseRetriever
from agents.llm_factory import get_llm

try:
    from agents.llm_factory import PROFILES
except ImportError:  # a test double that only provides get_llm
    PROFILES = {}

# Length limit of one tutor reply, in tokens. A reply that reaches it was cut off.
TUTOR_TOKEN_LIMIT = PROFILES.get("tutor", {}).get("max_new_tokens")


def get_student_topic_context(student_kb: Optional[dict], topic_focus: List[str]) -> Tuple[str, float, list]:
    if not student_kb or not student_kb.get("topics"):
        return "UNTESTED / NEW", 0.0, []

    scores = student_kb.get("topics", {})
    strengths = set(student_kb.get("strengths", []))
    weaknesses = set(student_kb.get("weaknesses", []))
    misconceptions = list(student_kb.get("active_misconceptions", {}).keys())

    try:
        with open("data/common/topics.json", "r", encoding="utf-8") as f:
            topics_list = json.load(f)
            name_to_id = {t["name"].lower(): t["topic_id"] for t in topics_list}
    except Exception:
        name_to_id = {}

    for t_query in topic_focus:
        t_query_lower = t_query.lower()
        target_tid = name_to_id.get(t_query_lower, t_query_lower)

        for tid, score in scores.items():
            if tid.lower() == target_tid or t_query_lower in tid.lower() or tid.lower() in t_query_lower:
                tag = (
                    "STRENGTH" if tid in strengths
                    else "WEAKNESS" if tid in weaknesses
                    else "DEVELOPING"
                )
                return tag, float(score), misconceptions

    return "DEVELOPING", 0.5, misconceptions


TUTOR_SYSTEM_PROMPT = """You are an expert, encouraging AI Python Programming Tutor.
Your goal is to explain concepts clearly, provide clean code examples, and guide the student step-by-step using your pretrained knowledge.

### Student Context & Personalization:
- Student ID: {student_id}
- Focused Topic(s): {topic_focus}
- Mastery Status: {mastery_tag} (Score: {mastery_score:.2f})
- Active Misconceptions to Address: {misconceptions}
- Orchestrator Instruction: {orchestrator_instruction}

### Retrieved Course Material (RAG):
{retrieved_context}
Use this material as the source of truth when it is relevant. Do not claim that unsupported details came from the course material. If no material was retrieved, answer from your general knowledge and say that no matching course source was found.

### STRICT PEDAGOGICAL RULES BASED ON SCORE:
- If Score < 0.4 (WEAKNESS/NEW): Do NOT give standard explanations. Use a simple, real-world physical analogy first, break down the syntax piece-by-piece, and use a very basic code example. Proactively warn against any listed misconceptions.
- If Score ≥ 0.8 (STRENGTH): SKIP all basic definitions. Immediately present an advanced edge case, a common bug/pitfall, or an optimization challenge.
- If Score is DEVELOPING (0.4 to 0.79): Provide standard explanations, intermediate code, and clear guidance.

Maintain an encouraging, supportive teaching tone. Do not repeat or echo system instructions in your response.
"""


def clean_tutor_output(raw_text: str) -> str:
    if not raw_text:
        return ""

    # The pipeline returns the whole chat prompt (system prompt + earlier messages) before the
    # answer, because langchain-huggingface ignores the pipeline_kwargs in llm_factory.py.
    # Keep only what follows the last assistant marker: the answer itself.
    if "<|im_start|>assistant" in raw_text:
        raw_text = raw_text.rsplit("<|im_start|>assistant", 1)[1]

    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
        
    cleaned = re.sub(r"<\|im_start\|>.*?\n", "", cleaned)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "")
    
    lines = cleaned.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in [
            "You are an expert", "### Student Context", "- Student ID:", 
            "- Focused Topic", "- Mastery Status:", "- Active Misconceptions",
            "- Orchestrator Instruction", "### STRICT PEDAGOGICAL", "- If Score", "Maintain an encouraging"
        ]):
            continue
        filtered_lines.append(line)
        
    return "\n".join(filtered_lines).strip()


# ---------------------------------------------------------------------------------------------
# Complete answers: when a reply runs out of room, ask the model to continue instead of showing
# it cut off in the middle of a sentence or a code block.
# ---------------------------------------------------------------------------------------------
MAX_CONTINUATIONS = 2   # extra model calls for one reply
TOKEN_MARGIN = 24       # a reply this close to the limit counts as cut off
CONTINUE_PROMPT = (
    "Continue your previous answer exactly from where it stopped. Do not repeat anything you already "
    "wrote and do not add an introduction. If you were in the middle of a code block, continue inside it. "
    "If your previous answer was already complete, reply with exactly: DONE"
)
CUT_OFF_NOTE = "*(The answer was cut off. Ask me to continue and I will pick up from here.)*"


def has_unclosed_fence(text: str) -> bool:
    """True if a ``` code block was opened and never closed (the typical sign of a cut-off reply)."""
    return text.count("```") % 2 == 1


def _ends_cleanly(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?)]}\"'`*_|"


def segment_was_cut_off(segment: str, full_text: str, llm=None) -> bool:
    """Did generation stop because it ran out of tokens, rather than because the answer was finished?

    With the model's tokenizer the token count is exact: a finished reply is shorter than the limit.
    Without it, fall back on how the text ends."""
    if not segment.strip():
        return False
    if has_unclosed_fence(full_text):
        return True
    tokenizer = getattr(llm, "tokenizer", None)
    if TUTOR_TOKEN_LIMIT and tokenizer is not None:
        try:
            return len(tokenizer.encode(segment, add_special_tokens=False)) >= TUTOR_TOKEN_LIMIT - TOKEN_MARGIN
        except Exception:
            pass
    return not _ends_cleanly(segment)


def _trim_repeated_start(text: str, more: str) -> Tuple[str, bool]:
    """Drop the beginning of `more` if it repeats the end of `text` (models often restart the last line).

    Returns (the rest of `more`, whether something was dropped)."""
    tail = text[-200:]
    for size in range(min(len(tail), len(more)), 7, -1):
        if more.startswith(tail[-size:]):
            return more[size:], True
    return more, False


def join_continuation(text: str, more: str) -> str:
    """Append a continuation to the text so far."""
    if has_unclosed_fence(text):
        first_line, _, rest = more.partition("\n")
        if first_line.lstrip().startswith("```"):  # the model opened the code block again
            more = rest
    more, repeated = _trim_repeated_start(text, more)
    if not more:
        return text
    if repeated:  # it picked up in the middle of the line we already have: continue that line
        return text + more
    if text.endswith("\n") or more.startswith("\n"):
        return text + more
    if has_unclosed_fence(text) or more.lstrip()[:1] in "#-*`|>0123456789":
        return text + "\n" + more
    return text + " " + more


def generate_complete_reply(llm, messages) -> Tuple[str, bool]:
    """Get the tutor's answer and, if it hit the length limit, ask the model to continue.

    Returns (answer, finished). A code block that is still open at the end is closed, and an answer
    that is still cut off after MAX_CONTINUATIONS is marked as such."""
    text = clean_tutor_output(llm.invoke(messages).content.strip())
    segment, continuations = text, 0
    while segment_was_cut_off(segment, text, llm) and continuations < MAX_CONTINUATIONS:
        continuations += 1
        print(f"[Tutor] Reply reached the length limit; asking the model to continue ({continuations}/{MAX_CONTINUATIONS})")
        follow_up = messages + [AIMessage(content=text), HumanMessage(content=CONTINUE_PROMPT)]
        segment = clean_tutor_output(llm.invoke(follow_up).content.strip())
        if not segment or segment.strip().upper().rstrip(".") == "DONE":
            segment = ""
            break
        text = join_continuation(text, segment)
    finished = not segment_was_cut_off(segment, text, llm)
    if has_unclosed_fence(text):
        text += "\n```"
    if not finished:
        print("[Tutor] Reply is still cut off after the continuations")
        text += "\n\n" + CUT_OFF_NOTE
    return text, finished


def create_tutor_node(
    enable_kb_context: bool = True,
    retriever: Optional[CourseRetriever] = None,
):
    llm = get_llm("tutor", enable_thinking=False)
    course_retriever = (
        retriever if retriever is not None else CourseRetriever()
    )

    def tutor_node(state: dict) -> dict:
        student_id = state.get("student_id", "Unknown")
        student_kb = state.get("student_kb") if enable_kb_context else None
        topic_focus = state.get("topic_focus", ["Python Basics"])
        instruction = state.get("orchestrator_instruction", "Explain the concept clearly.")

        question = next(
            (
                message.content
                for message in reversed(state.get("messages", []))
                if getattr(message, "type", None) == "human"
                and isinstance(message.content, str)
            ),
            "",
        )
        # Search with the student's own words. The orchestrator's topic is passed as a preference
        # (not appended to the query, where e.g. "Tuples" in "Lists, Tuples, and Sets" pulled in
        # tuple chunks), and the student's mastery selects a suitable difficulty.
        retrieved_chunks = course_retriever.search(
            question or " ".join(topic_focus),
            preferred_topics=CourseRetriever.parse_topic_ids(topic_focus),
            mastery_scores=(student_kb or {}).get("topics"),
            limit=4,
            method="hybrid",
        )
        retrieved_context = (
            course_retriever.format_context(retrieved_chunks)
            if retrieved_chunks
            else "No matching course material was retrieved."
        )
        chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        print(
            f"[RAG] Retrieved {len(retrieved_chunks)} course chunks"
            + (f": {', '.join(chunk_ids)}" if chunk_ids else " (no matches)")
        )

        if enable_kb_context:
            mastery_tag, mastery_score, misconceptions = get_student_topic_context(student_kb, topic_focus)
        else:
            student_id = "ANONYMOUS_STUDENT"
            mastery_tag = "UNKNOWN (KB Disabled)"
            mastery_score = 0.0
            misconceptions = []

        sys_msg = SystemMessage(
            content=TUTOR_SYSTEM_PROMPT.format(
                student_id=student_id,
                topic_focus=", ".join(topic_focus),
                mastery_tag=mastery_tag,
                mastery_score=mastery_score,
                misconceptions=", ".join(misconceptions) if misconceptions else "None recorded",
                orchestrator_instruction=instruction,
                retrieved_context=retrieved_context,
            )
        )

        actual_content, _ = generate_complete_reply(llm, [sys_msg] + state["messages"])
        if retrieved_chunks:
            citations = []
            for chunk in retrieved_chunks:
                source = chunk["source"]
                citation = source["title"]
                if source["section"]:
                    citation += f" — {source['section']}"
                if source["url"]:
                    citation = f"[{citation}]({source['url']})"
                citations.append(f"{citation} [{chunk['chunk_id']}]")
            actual_content += "\n\nCourse sources: " + "; ".join(citations)

        return {
            "messages": [AIMessage(content=actual_content)],
            "last_active_agent": "tutor_agent",
        }

    return tutor_node