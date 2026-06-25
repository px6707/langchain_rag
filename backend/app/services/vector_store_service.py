from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore

from app.config import settings
from app.services.embedding_service import get_embeddings

_store_cache: dict[bool, ElasticsearchStore] = {}


def get_vector_store(*, use_hybrid: bool | None = None) -> ElasticsearchStore:
    hybrid = settings.retrieval_hybrid_enabled if use_hybrid is None else use_hybrid
    if hybrid not in _store_cache:
        strategy = DenseVectorStrategy(
            hybrid=hybrid,
            rrf=settings.retrieval_rrf_enabled if hybrid else False,
        )
        _store_cache[hybrid] = ElasticsearchStore(
            es_url=settings.es_url,
            index_name=settings.es_index,
            embedding=get_embeddings(),
            strategy=strategy,
        )
    return _store_cache[hybrid]


def clear_vector_store_cache() -> None:
    _store_cache.clear()
