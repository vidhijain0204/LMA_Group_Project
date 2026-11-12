from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableLambda


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B"

PROFILES = {
    "router": {
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
    "tutor": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
    },
    "judge": {
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "quiz": {
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}

# Number of additional generation requests allowed.
# Total generation attempts = 1 + MAX_CONTINUATIONS.
MAX_CONTINUATIONS = 2


# --------------------------------------------------
# Model & Tokenizer Loader (Cached in NVIDIA GPU)
# --------------------------------------------------

@lru_cache(maxsize=1)
def _get_hf_model_and_tokenizer(model_path):
    print(f"\n[LLM] Loading Hugging Face model directly onto NVIDIA GPU: {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model onto GPU with bfloat16/float16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    print("[LLM] Model loaded successfully onto GPU.")
    return model, tokenizer


# --------------------------------------------------
# Message preparation
# --------------------------------------------------

def _apply_thinking_mode(messages, enable_thinking=False):
    """Apply Qwen3's thinking directive to the system message."""

    prepared_messages = list(messages)
    directive = "/think" if enable_thinking else "/no_think"

    for index, message in enumerate(prepared_messages):
        if isinstance(message, SystemMessage):
            content = message.content
            if isinstance(content, str):
                if directive not in content:
                    prepared_messages[index] = SystemMessage(
                        content=f"{content.rstrip()}\n\n{directive}"
                    )
                return prepared_messages

    prepared_messages.insert(0, SystemMessage(content=directive))
    return prepared_messages


def _normalize_messages(inputs):
    """Normalize supported LangChain inputs into messages."""

    if hasattr(inputs, "to_messages"):
        return list(inputs.to_messages())

    if isinstance(inputs, BaseMessage):
        return [inputs]

    if isinstance(inputs, str):
        return [HumanMessage(content=inputs)]

    if isinstance(inputs, (list, tuple)):
        return list(inputs)

    raise TypeError(
        f"Unsupported LLM input type: {type(inputs).__name__}"
    )


def _visible_text(chunk):
    """Drop Qwen3 reasoning; return "" if the model was still thinking at the token limit."""
    if "</think>" in chunk:
        return chunk.rsplit("</think>", 1)[1]
    if "<think>" in chunk:
        return ""
    return chunk


# --------------------------------------------------
# Iterative response generation (Direct Transformers)
# --------------------------------------------------

def _invoke_with_continuation(
    inputs,
    profile_config,
    enable_thinking=False,
):
    """
    Generate a response directly via Hugging Face and continue when 
    token length hits max_tokens limit.
    """
    model, tokenizer = _get_hf_model_and_tokenizer(MODEL_NAME)

    conversation = _normalize_messages(inputs)
    conversation = _apply_thinking_mode(conversation, enable_thinking)

    output_parts = []
    responses_meta = []

    max_attempts = 1 + MAX_CONTINUATIONS
    max_tokens = profile_config["max_tokens"]
    temperature = profile_config["temperature"]
    top_p = profile_config["top_p"]

    print(f"\n[LLM] Maximum generation attempts: {max_attempts}")

    for attempt in range(max_attempts):
        # Convert LangChain messages to chat format dictionaries
        chat_messages = []
        for msg in conversation:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            chat_messages.append({"role": role, "content": msg.content})

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize and move to GPU
        inputs_tensor = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_len = inputs_tensor.input_ids.shape[1]

        # Generate text directly using PyTorch/Transformers.
        # Sampling settings are passed only when sampling; greedy decoding (temperature 0)
        # would otherwise print a warning on every call.
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            outputs = model.generate(**inputs_tensor, **generation_kwargs)

        # Slice off the prompt tokens to get only the newly generated text
        generated_ids = outputs[0][input_len:]
        chunk = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Keep only user-visible answer text; Qwen3's <think> reasoning stays out of the
        # output and out of the conversation fed back for continuation.
        visible = _visible_text(chunk)

        output_parts.append(visible)

        # Determine finish reason based on whether max_new_tokens was hit
        generated_tokens = len(generated_ids)
        finish_reason = "length" if generated_tokens >= max_tokens else "stop"
        
        responses_meta.append({
            "finish_reason": finish_reason,
            "prompt_tokens": input_len,
            "completion_tokens": generated_tokens
        })

        generation_number = attempt + 1
        print(
            f"[LLM] Generation {generation_number}/{max_attempts} "
            f"| Finish reason: {finish_reason} "
            f"| Visible tokens: {len(visible.split())} words"
        )

        # Stop when the model finishes naturally
        if finish_reason != "length":
            break

        # No more continuation attempts are available
        if generation_number >= max_attempts:
            print("[LLM] Maximum continuation limit reached.")
            break

        # Append assistant chunk and continuation prompt for the next turn
        conversation.append(AIMessage(content=visible))

        if visible.strip():
            continuation_prompt = (
                "/no_think\n"
                "Continue exactly where your previous response "
                "ended. Do not repeat, restart, or summarize "
                "previous content. Continue the unfinished "
                "explanation naturally. Maintain the same "
                "structure, formatting, and level of detail."
            )
        else:
            print(
                "[LLM] No visible answer text in this chunk. "
                "Retrying in non-thinking mode."
            )
            continuation_prompt = (
                "/no_think\n"
                "Your previous generation reached its token limit "
                "before producing user-facing answer text. "
                "Begin writing the requested answer directly."
            )

        conversation.append(HumanMessage(content=continuation_prompt))

    final_content = "".join(output_parts)
    continuation_count = len(responses_meta) - 1

    metadata = {
        "continuation_count": continuation_count,
        "continuation_truncated": responses_meta[-1]["finish_reason"] == "length",
        "empty_final_response": not bool(final_content.strip()),
        "finish_reason": responses_meta[-1]["finish_reason"]
    }

    # LangChain's usage_metadata requires input_tokens, output_tokens and total_tokens;
    # leaving input_tokens out raises "usage_metadata.input_tokens Field required".
    total_input_tokens = sum(r["prompt_tokens"] for r in responses_meta)
    total_output_tokens = sum(r["completion_tokens"] for r in responses_meta)
    total_usage = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }

    final_response = AIMessage(
        content=final_content,
        response_metadata=metadata,
        usage_metadata=total_usage,
    )

    print(
        f"\n[LLM] Completed | "
        f"Continuations: {continuation_count} | "
        f"Truncated: {metadata['continuation_truncated']} | "
        f"Empty response: {metadata['empty_final_response']}"
    )

    return final_response


# --------------------------------------------------
# LLM factory
# --------------------------------------------------

@lru_cache(maxsize=None)
def get_llm(profile="tutor", enable_thinking=False):
    """Return a LangChain-compatible local NVIDIA-backed LLM with automatic continuation."""

    if profile not in PROFILES:
        raise ValueError(
            f"Unknown LLM profile '{profile}'. "
            f"Available profiles: {list(PROFILES)}"
        )

    config = PROFILES[profile]

    return RunnableLambda(
        lambda messages: _invoke_with_continuation(
            messages,
            config,
            enable_thinking,
        )
    )
