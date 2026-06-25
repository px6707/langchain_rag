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
