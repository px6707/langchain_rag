from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import settings


def get_llm(*, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_api_base,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        temperature=temperature,
    )


def get_small_llm(*, temperature: float | None = None) -> ChatOpenAI:
    """辅助任务用小模型：检索路由、query 改写、grounding 校验等。"""
    temp = settings.small_llm_temperature if temperature is None else temperature
    model = settings.small_llm_model or settings.llm_model
    api_base = settings.small_llm_api_base or settings.llm_api_base
    api_key = settings.small_llm_api_key or settings.llm_api_key
    return ChatOpenAI(
        model=model,
        base_url=api_base,
        api_key=SecretStr(api_key) if api_key else None,
        temperature=temp,
    )
