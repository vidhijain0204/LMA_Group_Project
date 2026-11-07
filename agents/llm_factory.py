
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda


MODEL_NAME = "Qwen/Qwen3-4B-MLX-4bit"
BASE_URL = "http://127.0.0.1:8080/v1"

PROFILES = {
    "router": {
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
    },
    "tutor": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
    },
    "judge": {
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "quiz": {
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}


def _apply_thinking_mode(messages, enable_thinking=False):
    """Apply Qwen3's thinking directive to the system message."""

    if not isinstance(messages, (list, tuple)):
        return messages

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


@lru_cache(maxsize=None)
def get_llm(profile="tutor", enable_thinking=False):
    """Return a LangChain-compatible client for the local MLX model."""

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

    prepare_messages = RunnableLambda(
        lambda messages: _apply_thinking_mode(
            messages,
            enable_thinking
        )
    )

    return prepare_messages | chat_model
