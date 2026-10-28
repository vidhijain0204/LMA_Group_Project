import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.skb_schema import StudentKnowledgeBase

# --- Configuration, Model & File Path Constants ---
DEFAULT_SKB_DB_PATH = "data/skb_database.json"
DEFAULT_MCQS_FILEPATH = "data/static_diagnosis_mcq_list.json"
SLM_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_DEVICE = "cuda"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3

class AssessmentAgent:
    def __init__(
        self,
        db_filepath: str = DEFAULT_SKB_DB_PATH,
        mcqs_filepath: str = DEFAULT_MCQS_FILEPATH,
        shared_tokenizer = None,
        shared_model = None
    ):
        self.db_filepath = db_filepath
        self.mcqs_filepath = mcqs_filepath
        self.tokenizer = shared_tokenizer
        self.model = shared_model

        self.mcqs = self._load_json(self.mcqs_filepath)

        if self.tokenizer is None or self.model is None:
            print(f"[AssessmentAgent] Loading standalone SLM ({SLM_MODEL_NAME})...")
            self.tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                SLM_MODEL_NAME, device_map="auto", torch_dtype="auto"
            )
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.max_length = None

        self.assessment_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _load_json(self, filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as file:
            return json.load(file)

    def _get_student_skb(self, student_id: str) -> dict:
        """Retrieves the student's enriched knowledge base profile."""
        if not os.path.exists(self.db_filepath):
            return {}
        with open(self.db_filepath, 'r') as file:
            try:
                db_data = json.load(file)
                for entry in db_data:
                    if entry.get("student_id") == student_id:
                        return entry
            except json.JSONDecodeError:
                pass
        return {}

    def get_targeted_quiz(self, student_id: str, limit: int = 3) -> list:
        """
        Selects quiz questions targeted to the student's weaknesses or active misconceptions.
        If no profile exists, returns a random sample.
        """
        skb = self._get_student_skb(student_id)
        weaknesses = skb.get("weaknesses", [])
        
        # Filter questions matching student's weak topics
        targeted_questions = [q for q in self.mcqs if q.get("topic_id") in weaknesses]
        
        # Fallback to general pool if no specific weaknesses found
        if not targeted_questions:
            targeted_questions = self.mcqs

        # Return a limited subset for the quiz session
        return targeted_questions[:limit]

    def evaluate_quiz(self, student_id: str, student_answers: dict) -> dict:
        """
        Grades a submitted quiz, updates the student's SKB probabilities, 
        adjusts active misconceptions, and persists changes to the database.
        """
        skb_data = self._get_student_skb(student_id)
        if not skb_data:
            return {"error": "Student SKB profile not found. Run diagnosis first."}

        # Load existing topics, skills, and misconception counters
        topics = skb_data.get("topics", {})
        skills = skb_data.get("skills", {})
        active_misconceptions = skb_data.get("active_misconceptions", {})

        correct_count = 0
        total_count = len(student_answers)
        evaluation_details = []

        for q in self.mcqs:
            q_id = q['problem_id']
            if q_id not in student_answers:
                continue

            topic_id = q['topic_id']
            skill_id = q.get('primary_skill_id')
            student_choice = student_answers[q_id]
            
            answer_key = json.loads(q['answer_key_json'])
            correct_options = answer_key.get('correct_option_ids', [])
            is_correct = student_choice in correct_options

            if is_correct:
                correct_count += 1
                # Increment topic probability slightly towards mastery (moving average or step boost)
                current_prob = topics.get(topic_id, 0.0)
                topics[topic_id] = round(min(1.0, current_prob + 0.15), 2)
                
                if skill_id:
                    curr_skill_prob = skills.get(skill_id, 0.0)
                    skills[skill_id] = round(min(1.0, curr_skill_prob + 0.15), 2)
            else:
                # Decrement probability and log misconception frequency
                current_prob = topics.get(topic_id, 0.0)
                topics[topic_id] = round(max(0.0, current_prob - 0.15), 2)
                
                misconceptions = q.get('misconception_tags', [])
                for tag in misconceptions:
                    active_misconceptions[tag] = active_misconceptions.get(tag, 0) + 1

            evaluation_details.append({
                "problem_id": q_id,
                "topic_id": topic_id,
                "is_correct": is_correct,
                "explanation": q.get("explanation", "No explanation provided.")
            })

        # Re-initialize Pydantic model to recalculate strengths/weaknesses and timestamps
        updated_skb = StudentKnowledgeBase(
            student_id=student_id,
            topics=topics,
            skills=skills,
            active_misconceptions=active_misconceptions,
            strengths=skb_data.get("strengths", []),
            weaknesses=skb_data.get("weaknesses", []),
            first_entry_time=skb_data.get("first_entry_time"),
            latest_update_time=skb_data.get("latest_update_time")
        )
        updated_skb.update_categorization()

        # Save back to database
        self._update_skb_database(updated_skb)

        return {
            "score": f"{correct_count}/{total_count}",
            "evaluation_details": evaluation_details,
            "updated_skb": updated_skb.model_dump(mode='json')
        }

    def _update_skb_database(self, skb_entry: StudentKnowledgeBase):
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
            
        print(f"Success: Assessment Agent updated SKB for {skb_entry.student_id} in {self.db_filepath}")