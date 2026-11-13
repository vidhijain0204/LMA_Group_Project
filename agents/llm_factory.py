
from functools import lru_cache

from langchain_openai import ChatOpenAI
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

MODEL_NAME = "Qwen/Qwen3-4B-MLX-4bit"
BASE_URL = "http://127.0.0.1:8080/v1"

PROFILES = {
    "router": {
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
    "tutor": {
        "max_tokens": 1024,
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

    prepared_messages.insert(
        0,
        SystemMessage(content=directive)
    )

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


def _extract_text(content):
    """Extract readable text from a LangChain message."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") in ("text", "output_text"):
                    parts.append(block.get("text", ""))

        return "".join(parts)

    return ""


# --------------------------------------------------
# Iterative response generation
# --------------------------------------------------


def _invoke_with_continuation(
    inputs,
    chat_model,
    enable_thinking=False,
):
    """
    Generate a response and continue when MLX reports
    finish_reason == 'length'.

    Also handles reasoning-only responses where the model
    has not yet produced user-visible answer text.
    """

    conversation = _normalize_messages(inputs)

    conversation = _apply_thinking_mode(
        conversation,
        enable_thinking,
    )

    output_parts = []
    responses = []

    max_attempts = 1 + MAX_CONTINUATIONS

    print(f"\n[LLM] Maximum generation attempts: {max_attempts}")

    for attempt in range(max_attempts):

        response = chat_model.invoke(conversation)
        responses.append(response)

        chunk = _extract_text(response.content)

        if chunk:
            output_parts.append(chunk)

        finish_reason = response.response_metadata.get(
            "finish_reason"
        )

        generation_number = attempt + 1

        print(
            f"[LLM] Generation {generation_number}/{max_attempts} "
            f"| Finish reason: {finish_reason} "
            f"| Visible tokens: {len(chunk.split())} words"
        )

        # Stop when the model finishes naturally.
        if finish_reason != "length":
            break

        # No more continuation attempts are available.
        if generation_number >= max_attempts:
            print("[LLM] Maximum continuation limit reached.")
            break

        # Preserve only user-visible assistant content.
        # Do not feed hidden reasoning back as answer text.
        conversation.append(
            AIMessage(content=chunk)
        )

        if chunk:
            continuation_prompt = (
                "/no_think\n"
                "Continue exactly where your previous response "
                "ended. Do not repeat, restart, or summarize "
                "previous content. Continue the unfinished "
                "explanation naturally. Maintain the same "
                "structure, formatting, and level of detail."
            )
        else:
            # Recover when the previous generation produced
            # reasoning but no user-visible answer text.
            print(
                "[LLM] No visible answer text in this chunk. "
                "Retrying in non-thinking mode."
            )

            continuation_prompt = (
                "/no_think\n"
                "Your previous generation reached its token limit "
                "before producing user-facing answer text. "
                "Do not generate or explain internal reasoning. "
                "Begin writing the requested answer directly. "
                "Provide the actual explanation or content requested."
            )

        conversation.append(
            HumanMessage(content=continuation_prompt)
        )

    # Combine all user-visible response chunks.
    final_content = "".join(output_parts)

    last_response = responses[-1]

    metadata = dict(
        last_response.response_metadata or {}
    )

    continuation_count = len(responses) - 1

    metadata["continuation_count"] = continuation_count

    metadata["continuation_truncated"] = (
        metadata.get("finish_reason") == "length"
    )

    metadata["empty_final_response"] = (
        not bool(final_content.strip())
    )

    # Aggregate token usage across all requests.
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    has_usage = False

    for item in responses:

        usage = item.usage_metadata or {}

        for key in total_usage:

            value = usage.get(key)

            if isinstance(value, (int, float)):
                total_usage[key] += value
                has_usage = True

    result_kwargs = {
        "content": final_content,
        "response_metadata": metadata,
    }

    if has_usage:
        result_kwargs["usage_metadata"] = total_usage

    final_response = AIMessage(**result_kwargs)

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
    """Return a LangChain-compatible LLM with automatic continuation."""

    if profile not in PROFILES:
        raise ValueError(
            f"Unknown LLM profile '{profile}'. "
            f"Available profiles: {list(PROFILES)}"
        )

    config = PROFILES[profile]

    chat_model = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key="mlx",
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        timeout=300,
        max_retries=1,
    )

    return RunnableLambda(
        lambda messages: _invoke_with_continuation(
            messages,
            chat_model,
            enable_thinking,
        )
    )
