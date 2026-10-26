import json
import os
from models.skb_schema import StudentKnowledgeBase

# --- Configuration & File Path Constants ---
DEFAULT_TOPICS_FILEPATH = "data/common/topics.json"
DEFAULT_MCQS_FILEPATH = "data/diagnosis/static_diagnosis_mcq_list.json"
DEFAULT_DB_FILEPATH = "data/skb_database.json"
DEFAULT_THRESHOLD = 0.7

class DiagnosisAgent:
    def __init__(
        self, 
        topics_filepath: str = DEFAULT_TOPICS_FILEPATH, 
        mcqs_filepath: str = DEFAULT_MCQS_FILEPATH, 
        db_filepath: str = DEFAULT_DB_FILEPATH
    ):
        self.topics_filepath = topics_filepath
        self.mcqs_filepath = mcqs_filepath
        self.db_filepath = db_filepath
        self.threshold = DEFAULT_THRESHOLD

        self.topics = self._load_json(self.topics_filepath)
        self.mcqs = self._load_json(self.mcqs_filepath)

    def _load_json(self, filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as file:
            return json.load(file)

    def administer_test(self, student_id: str, student_answers: dict) -> StudentKnowledgeBase:
        # 1. Calculate raw scores per topic based on MCQs
        topic_scores = {topic['topic_id']: {'correct': 0, 'total': 0} for topic in self.topics}
        
        for q in self.mcqs:
            topic_id = q['topic_id']
            if topic_id in topic_scores:
                topic_scores[topic_id]['total'] += 1
                
                student_choice = student_answers.get(q['question_id'])
                if student_choice == q['correct_option']:
                    topic_scores[topic_id]['correct'] += 1

        # 2. Calculate probabilities
        topic_probabilities = {}
        for topic_id, stats in topic_scores.items():
            if stats['total'] > 0:
                probability = round(stats['correct'] / stats['total'], 2)
            else:
                probability = 0.0
            
            topic_probabilities[topic_id] = probability

        # 3. Create SKB entry
        skb_entry = StudentKnowledgeBase(
            student_id=student_id,
            topics=topic_probabilities
        )
        
        skb_entry.update_categorization(threshold=self.threshold)
        
        # 4. Write to database 
        self._write_to_skb_database(skb_entry)
        
        return skb_entry

    def _write_to_skb_database(self, skb_entry: StudentKnowledgeBase):
        """Writes the SKB entry to a local JSON file acting as a database."""
        db_data = []
        
        if os.path.exists(self.db_filepath):
            with open(self.db_filepath, 'r') as file:
                try:
                    db_data = json.load(file)
                except json.JSONDecodeError:
                    db_data = []

        db_data = [entry for entry in db_data if entry.get("student_id") != skb_entry.student_id]
        db_data.append(skb_entry.model_dump(mode='json'))

        with open(self.db_filepath, 'w') as file:
            json.dump(db_data, file, indent=2)
            
        print(f"Success: Diagnosis Agent saved SKB for {skb_entry.student_id} to {self.db_filepath}")