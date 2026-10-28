from functools import lru_cache, partial

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

SLM_MODEL_NAME = "Qwen/Qwen3-4B"

# One profile per job. Same weights, different decoding settings.
PROFILES = {
    "router": dict(max_new_tokens=32,  do_sample=False),
    "tutor":  dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9),
    "judge":  dict(max_new_tokens=256, do_sample=False),
    "quiz":   dict(max_new_tokens=512, do_sample=False),
}


@lru_cache(maxsize=1)
def load_base_model():
    """Loads Qwen3-4B once per process. lru_cache makes every later call free."""
    print(f"[llm_factory] Loading {SLM_MODEL_NAME} (one time)...")
    tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL_NAME, device_map="auto", torch_dtype="auto"
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None   # same fix you had before

    # Qwen3 "thinking" off for every call. ChatHuggingFace calls
    # tokenizer.apply_chat_template internally and gives no way to pass
    # enable_thinking, so we bake it in once here.
    tokenizer.apply_chat_template = partial(
        tokenizer.apply_chat_template, enable_thinking=False
    )
    return tokenizer, model


@lru_cache(maxsize=None)
def get_llm(profile: str = "tutor") -> ChatHuggingFace:
    """Returns a LangChain chat model for the given role (cached per role)."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from {list(PROFILES)}")

    tokenizer, model = load_base_model()
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    pipe_llm = HuggingFacePipeline(
        pipeline=hf_pipe,
        pipeline_kwargs={
            **PROFILES[profile],
            "return_full_text": False,                 # only the new tokens
            "pad_token_id": tokenizer.eos_token_id,
        },
    )
    return ChatHuggingFace(llm=pipe_llm, tokenizer=tokenizer)