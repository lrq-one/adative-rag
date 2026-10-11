from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ai_agent_template.core.settings import Settings


def build_chat_model(
    settings: Settings,
    *,
    model_name: str | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    name = model_name or settings.model_name
    output_limit = max_tokens or settings.model_max_tokens
    match settings.model_provider:
        case "groq":
            return ChatGroq(
                api_key=settings.groq_api_key,
                model_name=name,
                temperature=settings.model_temperature,
                timeout=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
                max_tokens=output_limit,
            )
        case "openai":
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model=name,
                base_url=settings.model_base_url,
                temperature=settings.model_temperature,
                timeout=settings.model_timeout_seconds,
                max_retries=settings.model_max_retries,
                max_tokens=output_limit,
                extra_body={"enable_thinking": False} if name.lower().startswith("qwen") else None,
            )
