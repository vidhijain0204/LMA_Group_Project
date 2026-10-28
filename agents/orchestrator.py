import os
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration & File Path Constants ---
DEFAULT_SKB_DB_PATH = "data/skb_database.json"
HISTORY_TURNS = 4          # how many recent messages the router sees
HISTORY_CHAR_LIMIT = 300   # truncate long turns so the prompt stays small

# Strong, unambiguous assessment phrases. These skip the LLM entirely.
ASSESSMENT_PATTERN = re.compile(
    r"\b(quiz me|give me a quiz|take a quiz|take a test|test me|test my|"
    r"mcqs?|practice questions?|check my knowledge|evaluate me|assess me)\b",
    re.IGNORECASE,
)

ROUTER_SYSTEM_PROMPT = """You are an intent classification router for an adaptive Python tutoring platform.
Classify the student's LATEST message into exactly one category:

TUTOR: the student wants a concept explained, asks a programming question, or needs debugging help.
ASSESSMENT: the student wants to take a quiz or test, or asks to be questioned on what they know.

Use the recent conversation only to resolve short replies. For example, "yes, go ahead" after the tutor offered a quiz means ASSESSMENT.
Output ONLY one word: TUTOR or ASSESSMENT."""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human",
     "Recent conversation:\n{history}\n\n"
     "Latest student message: {user_input}\n\n"
     "Category:"),
])


class Orchestrator:
    def __init__(self, db_filepath: str = DEFAULT_SKB_DB_PATH, llm=None):
        self.db_filepath = db_filepath
        if llm is None:
            from agents.llm_factory import get_llm      # lazy import: keeps unit tests light
            llm = get_llm("router")
        self.router_chain = ROUTER_PROMPT | llm | StrOutputParser()

    # ------------------------------------------------------------ KB lookup
    def check_student_exists(self, student_id: str) -> bool:
        if not os.path.exists(self.db_filepath):
            return False
        with open(self.db_filepath, "r") as file:
            try:
                db_data = json.load(file)
                return any(entry.get("student_id") == student_id for entry in db_data)
            except json.JSONDecodeError:
                return False

    # --------------------------------------------------------- chat history
    @staticmethod
    def _format_history(chat_history) -> str:
        """chat_history = [{"role": "user" | "assistant", "content": "..."}, ...]"""
        if not chat_history:
            return "(no previous messages)"
        lines = []
        for msg in chat_history[-HISTORY_TURNS:]:
            role = "Student" if msg.get("role") == "user" else "Tutor"
            lines.append(f"{role}: {str(msg.get('content', ''))[:HISTORY_CHAR_LIMIT]}")
        return "\n".join(lines)

    # ------------------------------------------------------ intent routing
    @staticmethod
    def _parse_label(raw: str) -> str:
        text = raw.upper()
        # if the model says both words, trust whichever appears first
        a, t = text.find("ASSESSMENT"), text.find("TUTOR")
        if a != -1 and (t == -1 or a < t):
            return "ASSESSMENT"
        return "TUTOR"

    def classify_intent(self, user_input: str, chat_history: list = None) -> str:
        # 1) cheap deterministic path
        if ASSESSMENT_PATTERN.search(user_input):
            return "ASSESSMENT"

        # 2) LLM path for ambiguous messages
        try:
            raw = self.router_chain.invoke({
                "history": self._format_history(chat_history),
                "user_input": user_input,
            })
            return self._parse_label(raw)
        except Exception as e:
            print(f"[Orchestrator] Router LLM failed ({e}). Defaulting to TUTOR.")
            return "TUTOR"  