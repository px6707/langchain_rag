from langchain_elasticsearch import ElasticsearchStore

from app.config import settings
from app.services.embedding_service import get_embeddings


def get_vector_store() -> ElasticsearchStore:
    return ElasticsearchStore(
        es_url=settings.es_url,
        index_name=settings.es_index,
        embedding=get_embeddings(),
    )
