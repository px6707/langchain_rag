from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_api_base,
        api_key=SecretStr(settings.embedding_api_key) if settings.embedding_api_key else None,
        # DashScope 等 OpenAI 兼容 API 不接受 token 数组，需传原始文本
        check_embedding_ctx_length=False,
    )
