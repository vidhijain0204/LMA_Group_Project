import os
from functools import lru_cache, partial

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

SLM_MODEL_NAME = "Qwen/Qwen3-8B"

SLM_MODEL_PATH = (
    "/home/jenil/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)

HF_CACHE_DIR = "/home/.cache"

PROFILES = {
    "router": dict(max_new_tokens=5000, do_sample=False, enable_thinking=True),
    "tutor": dict(max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9),
    "judge": dict(max_new_tokens=256, do_sample=False),
    "quiz": dict(max_new_tokens=512, do_sample=False),
}


@lru_cache(maxsize=1)
def load_base_model():
    print(f"[llm_factory] Loading local model from {SLM_MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
    )

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    return model


@lru_cache(maxsize=2)
def load_tokenizer(enable_thinking: bool = False):
    print("[llm_factory] Loading local tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        SLM_MODEL_PATH,
        local_files_only=True,
    )

    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template,
        enable_thinking=enable_thinking,
    )

    return tokenizer


@lru_cache(maxsize=None)
def get_llm(profile: str = "tutor", enable_thinking: bool = False):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from {list(PROFILES)}")

    model = load_base_model()
    tokenizer = load_tokenizer(enable_thinking=enable_thinking)

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    pipe_llm = HuggingFacePipeline(
        pipeline=hf_pipe,
        pipeline_kwargs={
            **PROFILES[profile],
            "return_full_text": False,
            "pad_token_id": tokenizer.eos_token_id,
        },
    )

    return ChatHuggingFace(llm=pipe_llm, tokenizer=tokenizer)