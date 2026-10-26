import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration, Model & File Path Constants ---
DEFAULT_SKB_DB_PATH = "data/skb_database.json"
DEFAULT_COURSE_MATERIAL_PATH = "data/course_material/python_handbook.txt"
SLM_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_DEVICE = "cuda"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

class TutorAgent:
    def __init__(
        self,
        db_filepath: str = DEFAULT_SKB_DB_PATH,
        course_material_path: str = DEFAULT_COURSE_MATERIAL_PATH,
        model_name: str = SLM_MODEL_NAME
    ):
        self.db_filepath = db_filepath
        self.course_material_path = course_material_path
        self.model_name = model_name

        print(f"[TutorAgent] Loading SLM ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # to prevent conflict with max_new_tokens.
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None

        self.tutor_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def _get_student_skb(self, student_id: str) -> dict:
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

    def _retrieve_domain_knowledge(self, user_query: str) -> str:
        if not os.path.exists(self.course_material_path):
            return "[Domain dataset not found. Relying on model internal knowledge.]"

        try:
            with open(self.course_material_path, 'r') as file:
                content = file.read()
                return f"[Retrieved Course Context]: {content[:1000]}"
        except Exception:
            return "[Error reading domain dataset. Relying on model internal knowledge.]"

    def explain(self, student_id: str, user_query: str) -> str:
        skb_profile = self._get_student_skb(student_id)
        weaknesses = skb_profile.get("weaknesses", [])
        strengths = skb_profile.get("strengths", [])

        rag_context = self._retrieve_domain_knowledge(user_query)

        system_prompt = (
            "You are an expert, encouraging Python programming tutor agent. "
            "Explain concepts clearly with code examples. "
            "Adapt your explanation tone based on the student's profile: "
            f"The student's current identified weaknesses are topics: {weaknesses}, and strengths are: {strengths}. "
            "If the requested topic is in their weaknesses, provide a more foundational, step-by-step explanation with clear analogies."
        )

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Context Information:
{rag_context}

Student Profile Data:
SKB ID: {student_id}
Weaknesses: {weaknesses}

Student Question: "{user_query}"<|im_end|>
<|im_start|>assistant
"""

        print("*"*30)
        print("Prompt is ready."+ prompt)
        print("*"*30)
        
        # directly inside the generate call instead of the pipeline constructor.
        response = self.tutor_pipeline(
            prompt, 
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]['generated_text']
        
        explanation = response.split("<|im_start|>assistant")[-1].strip()

        return explanation