import os
from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


MODEL_NAME = os.getenv("TUTOR_MODEL_NAME", "Qwen/Qwen3-4B-MLX-4bit")
BASE_URL = os.getenv("TUTOR_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.getenv("TUTOR_LLM_API_KEY", "local")

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

MAX_CONTINUATIONS = 2


@lru_cache(maxsize=None)
def _get_chat_model(profile):
    config = PROFILES[profile]
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=config["temperature"],
        top_p=config["top_p"],
        extra_body={"max_tokens": config["max_tokens"]},
        timeout=120,
        max_retries=0,
    )


def _apply_thinking_mode(messages, enable_thinking=False):
    prepared_messages = list(messages)
    directive = "/think" if enable_thinking else "/no_think"

    for index, message in enumerate(prepared_messages):
        if isinstance(message, SystemMessage):
            content = message.content
            if isinstance(content, str) and directive not in content:
                prepared_messages[index] = SystemMessage(
                    content=f"{content.rstrip()}\n\n{directive}"
                )
            return prepared_messages

    prepared_messages.insert(0, SystemMessage(content=directive))
    return prepared_messages


def _normalize_messages(inputs):
    if hasattr(inputs, "to_messages"):
        return list(inputs.to_messages())
    if isinstance(inputs, BaseMessage):
        return [inputs]
    if isinstance(inputs, str):
        return [HumanMessage(content=inputs)]
    if isinstance(inputs, (list, tuple)):
        return list(inputs)
    raise TypeError(f"Unsupported LLM input type: {type(inputs).__name__}")


def _visible_text(chunk):
    if "</think>" in chunk:
        return chunk.rsplit("</think>", 1)[1]
    if "<think>" in chunk:
        return ""
    return chunk


def _invoke_with_continuation(inputs, profile, enable_thinking=False):
    conversation = _apply_thinking_mode(
        _normalize_messages(inputs), enable_thinking
    )
    model = _get_chat_model(profile)
    output_parts = []
    responses = []

    for attempt in range(1 + MAX_CONTINUATIONS):
        response = model.invoke(conversation)
        if not isinstance(response.content, str):
            raise TypeError("The local LLM returned non-text message content")

        visible = _visible_text(response.content)
        output_parts.append(visible)
        responses.append(response)

        if response.response_metadata.get("finish_reason") != "length":
            break
        if attempt + 1 >= 1 + MAX_CONTINUATIONS:
            print("[LLM] Maximum continuation limit reached.")
            break

        conversation.append(AIMessage(content=visible))
        continuation_prompt = (
            "Continue exactly where your previous response ended. Do not repeat, "
            "restart, or summarize it. Continue the unfinished explanation naturally."
            if visible.strip()
            else "Begin writing the requested answer directly; the previous response "
            "did not contain user-facing text."
        )
        conversation.append(
            HumanMessage(content=f"/no_think\n{continuation_prompt}")
        )

    final_content = "".join(output_parts)
    usage = [response.usage_metadata or {} for response in responses]
    input_tokens = sum(item.get("input_tokens", 0) for item in usage)
    output_tokens = sum(item.get("output_tokens", 0) for item in usage)
    last_response = responses[-1]
    metadata = {
        **last_response.response_metadata,
        "continuation_count": len(responses) - 1,
        "continuation_truncated": (
            last_response.response_metadata.get("finish_reason") == "length"
        ),
        "empty_final_response": not bool(final_content.strip()),
    }

    return AIMessage(
        content=final_content,
        response_metadata=metadata,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


@lru_cache(maxsize=None)
def get_llm(profile="tutor", enable_thinking=False):
    """Return a LangChain runnable backed by the local MLX-LM server."""
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown LLM profile '{profile}'. Available profiles: {list(PROFILES)}"
        )

    return RunnableLambda(
        lambda messages: _invoke_with_continuation(
            messages, profile, enable_thinking
        )
    )
