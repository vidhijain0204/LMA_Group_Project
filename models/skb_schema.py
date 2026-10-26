from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime

def get_current_time() -> str:
    """Returns the current date and time in a simple YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class StudentKnowledgeBase(BaseModel):
    student_id: str
    topics: Dict[str, float] = Field(default_factory=dict, description="Topic ID to probability mapping")
    strengths: List[str] = Field(default_factory=list, description="Topics with probability >= 0.7")
    weaknesses: List[str] = Field(default_factory=list, description="Topics with probability < 0.7")
    
    # Timestamp fields updated to string type
    first_entry_time: str = Field(default_factory=get_current_time, description="Time when the SKB was first created")
    latest_update_time: str = Field(default_factory=get_current_time, description="Time of the most recent SKB update")

    def update_categorization(self, threshold: float = 0.7):
        """Automatically sorts topics into strengths and weaknesses based on probability and updates the modified time."""
        self.strengths = [topic for topic, prob in self.topics.items() if prob >= threshold]
        self.weaknesses = [topic for topic, prob in self.topics.items() if prob < threshold]
        self.latest_update_time = get_current_time()