from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime

# --- Configuration Constants ---
DEFAULT_THRESHOLD = 0.7

def get_current_time() -> str:
    """Returns the current date and time in a simple YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class StudentKnowledgeBase(BaseModel):
    student_id: str
    
    # Topic-level probability mapping (e.g., {"T01": 0.89, "T06": 0.33})
    topics: Dict[str, float] = Field(default_factory=dict, description="Topic ID to mastery probability mapping")
      
    # Subtopic-level probability mapping (e.g., {"T06.3": 0.5}); filled by the diagnosis test
    skills: Dict[str, float] = Field(default_factory=dict, description="Subtopic ID to mastery probability mapping")

    strengths: List[str] = Field(default_factory=list, description="Topics with probability >= threshold")
    weaknesses: List[str] = Field(default_factory=list, description="Topics with probability < threshold")

    # Misconceptions seen in wrong answers: description -> number of times observed
    active_misconceptions: Dict[str, int] = Field(default_factory=dict, description="Observed misconceptions and their counts")
    
    # Timestamps
    first_entry_time: str = Field(default_factory=get_current_time, description="Time when the SKB was first created")
    latest_update_time: str = Field(default_factory=get_current_time, description="Time of the most recent SKB update")

    def update_categorization(self, threshold: float = DEFAULT_THRESHOLD):
        """Automatically sorts topics into strengths and weaknesses based on probability and updates timestamp."""
        self.strengths = [topic for topic, prob in self.topics.items() if prob >= threshold]
        self.weaknesses = [topic for topic, prob in self.topics.items() if prob < threshold]
        self.latest_update_time = get_current_time()