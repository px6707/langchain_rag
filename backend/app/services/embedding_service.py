from langchain_openai import OpenAIEmbeddings

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_base=settings.embedding_api_base,
        openai_api_key=settings.embedding_api_key,
    )
