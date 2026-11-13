"""Data model of a student's knowledge base (the "SKB"), one per student.

Where it fits: diagnosis_agent.py creates a StudentKnowledgeBase after the diagnosis test and
saves it (as JSON) in data/skb_database.json. After that, the other agents read and update the
stored record as a plain dict (mastery after quizzes, chat memory, root causes, ...), so this
class mainly documents which fields a record has and gives them their defaults. Some keys are
added later without being listed here, e.g. "chat_memory" (see agents/chat_memory.py).
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List
from datetime import datetime

DEFAULT_THRESHOLD = 0.7   # topic mastery at or above this counts as a strength

def get_current_time() -> str:
    """The current date and time as "YYYY-MM-DD HH:MM:SS"."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class StudentKnowledgeBase(BaseModel):
    """Everything the system knows about one student. Mastery values are between 0 and 1."""

    student_id: str
    
    # Topic id -> mastery (e.g. {"T01": 0.89, "T06": 0.33})
    topics: Dict[str, float] = Field(default_factory=dict, description="Topic ID to mastery probability mapping")
      
    # Subtopic id -> mastery (e.g. {"T06.3": 0.5}); first filled by the diagnosis test
    skills: Dict[str, float] = Field(default_factory=dict, description="Subtopic ID to mastery probability mapping")

    strengths: List[str] = Field(default_factory=list, description="Topics with probability >= threshold")
    weaknesses: List[str] = Field(default_factory=list, description="Topics with probability < threshold")

    # Misconceptions seen in wrong answers: description -> number of times observed
    active_misconceptions: Dict[str, int] = Field(default_factory=dict, description="Observed misconceptions and their counts")

    # Lessons and quizzes in order (with the teaching strategy used and the quiz outcome),
    # plus how often each error signal from error_analysis_agent.py was seen, and its text.
    pedagogy_history: List[Dict[str, Any]] = Field(default_factory=list)
    learning_signals: Dict[str, int] = Field(default_factory=dict)
    learning_signal_descriptions: Dict[str, str] = Field(default_factory=dict)

    # Student messages sent to the tutor, tagged with topic, subtopics and struggle flags
    # (see agents/struggle_tracker.py), and the root cause selected after each quiz.
    learning_conversations: List[Dict[str, Any]] = Field(default_factory=list)
    root_cause_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Re-tests after a remediation lesson: accuracy on the root-cause subtopic before and after.
    retests: List[Dict[str, Any]] = Field(default_factory=list)

    # One record per diagnosis question ({prob_id, is_correct, subtopics}); the root-cause agent
    # treats the diagnosis as the student's first quiz (counter-evidence and repeats).
    diagnosis_responses: List[Dict[str, Any]] = Field(default_factory=list)
    
    first_entry_time: str = Field(default_factory=get_current_time, description="Time when the SKB was first created")
    latest_update_time: str = Field(default_factory=get_current_time, description="Time of the most recent SKB update")

    def update_categorization(self, threshold: float = DEFAULT_THRESHOLD):
        """Sort topics into strengths (mastery >= threshold) and weaknesses, and update the time stamp."""
        self.strengths = [topic for topic, prob in self.topics.items() if prob >= threshold]
        self.weaknesses = [topic for topic, prob in self.topics.items() if prob < threshold]
        self.latest_update_time = get_current_time()