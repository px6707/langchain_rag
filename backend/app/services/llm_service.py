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


def get_router_llm(*, temperature: float | None = None) -> ChatOpenAI:
    temp = settings.retrieval_router_temperature if temperature is None else temperature
    model = settings.router_llm_model or settings.llm_model
    api_base = settings.router_llm_api_base or settings.llm_api_base
    api_key = settings.router_llm_api_key or settings.llm_api_key
    return ChatOpenAI(
        model=model,
        base_url=api_base,
        api_key=SecretStr(api_key) if api_key else None,
        temperature=temp,
    )
