"""Load the local Qwen LLM and hand it out to the agents.

Every agent that needs a language model (orchestrator, tutor, assessment, workflow, some
scripts) calls `get_llm(profile)` from this file. It returns a LangChain runnable, so callers
just do `llm.invoke(messages)` and get back an `AIMessage`.

How it works:
    - The model (MODEL_NAME, Qwen3-4B) is loaded once with Hugging Face `transformers`
      onto the GPU (`device_map="auto"`) and cached, so every agent shares the same copy.
    - A "profile" sets the generation settings for one kind of job (PROFILES):
      router (routing decisions), tutor (long explanations), judge (short, deterministic
      verdicts) and quiz (deterministic quiz generation).
    - Qwen3 can "think" (write hidden reasoning inside <think>...</think>) before answering.
      `enable_thinking` switches this on or off with the /think or /no_think directive, and
      the reasoning is always removed from the text returned to the caller.
    - If an answer is cut off at the token limit, the model is asked to continue, up to
      MAX_CONTINUATIONS extra times, and the pieces are joined into one answer.

Main parts: PROFILES, get_llm() (public entry point), _invoke_with_continuation() (the
generation loop) and a few small message helpers.
"""

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

# Generation settings per profile. max_tokens = new tokens per generation attempt;
# temperature 0.0 means greedy (deterministic) decoding.
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

# How many times the model may be asked to continue a cut-off answer.
# Total generation attempts = 1 + MAX_CONTINUATIONS.
MAX_CONTINUATIONS = 2


# --------------------------------------------------
# Model and tokenizer loader (loaded once, then cached)
# --------------------------------------------------

@lru_cache(maxsize=1)
def _get_hf_model_and_tokenizer(model_path):
    """Load the Hugging Face model and tokenizer once and reuse them on later calls.

    `device_map="auto"` lets transformers place the model on the available GPU(s).
    Returns (model, tokenizer).
    """
    print(f"\n[LLM] Loading Hugging Face model: {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Half precision saves memory: bfloat16 when the GPU supports it, else float16.
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
    """Add Qwen3's "/think" or "/no_think" directive to the conversation.

    The directive is appended to the first system message (if not already there); when there
    is no system message, a new one holding only the directive is inserted at the front.
    Returns a new list; the input list is not changed.
    """

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
    """Turn any supported LangChain input into a list of messages.

    Accepts a prompt value (anything with `to_messages()`), a single message, a plain string
    (becomes a HumanMessage) or a list/tuple of messages. Raises TypeError otherwise.
    """

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
    """Remove Qwen3's <think> reasoning and return only the answer text.

    Keeps the text after the last "</think>". If "<think>" was opened but never closed, the
    model was still thinking when it hit the token limit, so there is no answer yet: returns "".
    """
    if "</think>" in chunk:
        return chunk.rsplit("</think>", 1)[1]
    if "<think>" in chunk:
        return ""
    return chunk


# --------------------------------------------------
# Response generation with automatic continuation
# --------------------------------------------------

def _invoke_with_continuation(
    inputs,
    profile_config,
    enable_thinking=False,
):
    """Generate an answer with the local model, continuing it if it gets cut off.

    Each attempt generates up to `max_tokens` new tokens. If the limit is hit, the visible
    text so far is added to the conversation as an assistant message, followed by a user
    message asking the model to continue (or, if it produced no answer text yet, to start
    answering directly). This repeats at most MAX_CONTINUATIONS times.

    Args:
        inputs: messages in any form accepted by _normalize_messages().
        profile_config: one entry of PROFILES (max_tokens, temperature, top_p).
        enable_thinking: let Qwen3 reason in <think> tags before answering.

    Returns:
        An AIMessage with the joined answer text, `response_metadata` (continuation count,
        whether it was still truncated or empty, last finish reason) and `usage_metadata`
        (token counts summed over all attempts).
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
        # Convert LangChain messages to the {"role", "content"} dicts the chat template expects.
        # Unknown message types are treated as user messages.
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

        # Build Qwen's prompt text; add_generation_prompt=True ends it with the assistant's turn.
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs_tensor = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_len = inputs_tensor.input_ids.shape[1]

        # Sampling settings are passed only when sampling; with greedy decoding (temperature 0)
        # transformers would otherwise print a warning on every call.
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

        # The output starts with the prompt tokens; keep only the newly generated ones.
        generated_ids = outputs[0][input_len:]
        chunk = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Keep only user-visible answer text; Qwen3's <think> reasoning stays out of the
        # output and out of the conversation fed back for continuation.
        visible = _visible_text(chunk)

        output_parts.append(visible)

        # "length" = cut off at the token limit, "stop" = the model finished by itself.
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

        if finish_reason != "length":
            break

        if generation_number >= max_attempts:
            print("[LLM] Maximum continuation limit reached.")
            break

        # Show the model what it wrote so far, then ask it to continue from there.
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
    """Return the local Qwen model as a LangChain runnable for one profile.

    Args:
        profile: a key of PROFILES ("router", "tutor", "judge" or "quiz").
        enable_thinking: let Qwen3 reason before answering (the reasoning is not returned).

    Returns a RunnableLambda: `.invoke(messages)` returns an AIMessage (see
    _invoke_with_continuation). Results are cached, so each profile/thinking pair is built once.
    Raises ValueError for an unknown profile.
    """

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
