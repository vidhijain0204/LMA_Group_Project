import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from agents.diagnosis_agent import DiagnosisAgent

# --- Configuration, Model & File Path Constants ---
DEFAULT_SKB_DB_PATH = "data/skb_database.json"
SLM_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_DEVICE = "cuda"  # Set to "cpu" if running without GPU
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.1

class Orchestrator:
    def __init__(
        self, 
        db_filepath: str = DEFAULT_SKB_DB_PATH,
        model_name: str = SLM_MODEL_NAME
    ):
        self.db_filepath = db_filepath
        self.model_name = model_name
        self.diagnosis_agent = DiagnosisAgent()
        
        # Initialize the SLM pipeline for intent classification
        print(f"[Orchestrator] Loading SLM ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Clear default max_length to prevent conflict with max_new_tokens
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None

        # Initialize pipeline cleanly WITHOUT generation arguments
        self.intent_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def _check_student_exists(self, student_id: str) -> bool:
        """Checks if a student already has an entry in the SKB database."""
        if not os.path.exists(self.db_filepath):
            return False
        
        with open(self.db_filepath, 'r') as file:
            try:
                db_data = json.load(file)
                return any(entry.get("student_id") == student_id for entry in db_data)
            except json.JSONDecodeError:
                return False

    def _classify_intent_with_slm(self, user_input: str) -> str:
        """
        Uses Qwen/Qwen3-4B to classify the student's message intent.
        Expected outputs: 'TUTOR' or 'ASSESSMENT'
        """
        prompt = f"""<|im_start|>system
You are an intent classification router for an adaptive Python programming tutoring platform. 
Analyze the student message and classify their intent into one of two categories:
- TUTOR: The student wants a concept explained, is asking a programming question, or needs debugging help.
- ASSESSMENT: The student wants to take a test, quiz, or check their knowledge via questions.

Output ONLY the category name (`TUTOR` or `ASSESSMENT`). Do not add any extra text or punctuation.<|im_end|>
<|im_start|>user
Student message: "{user_input}"<|im_end|>
<|im_start|>assistant
"""
        # Pass generation parameters directly inside the pipeline call
        response = self.intent_pipeline(
            prompt, 
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]['generated_text']
        
        # Extract assistant response after the prompt
        output_text = response.split("<|im_start|>assistant")[-1].strip().upper()
        
        if "ASSESSMENT" in output_text:
            return "ASSESSMENT"
        return "TUTOR"  # Default fallback to Tutor

    def route_request(self, student_id: str, user_input: str, student_answers: dict = None):
        """
        Core decision-making method for the Orchestrator.
        """
        # Step 1: Cold-Start Check (New vs Existing Student)
        is_new_student = not self._check_student_exists(student_id)

        if is_new_student:
            print(f"[Orchestrator] Student {student_id} is new. Routing to -> Diagnosis Agent.")
            
            if student_answers:
                skb = self.diagnosis_agent.administer_test(student_id, student_answers)
                return {"status": "diagnosed", "routed_to": "DiagnosisAgent", "skb": skb}
            else:
                return {
                    "status": "pending_diagnosis", 
                    "routed_to": "DiagnosisAgent", 
                    "message": "New student detected. Please complete the static diagnostic MCQ test."
                }

        # Step 2: Intent Classification via Qwen/Qwen3-4B for existing students
        intent = self._classify_intent_with_slm(user_input)

        if intent == "ASSESSMENT":
            print(f"[Orchestrator] Intent classified as ASSESSMENT. Routing to -> Assessment Agent.")
            return {"status": "active", "routed_to": "AssessmentAgent"}
        else:
            print(f"[Orchestrator] Intent classified as TUTOR. Routing to -> Tutor Agent.")
            return {"status": "active", "routed_to": "TutorAgent"}