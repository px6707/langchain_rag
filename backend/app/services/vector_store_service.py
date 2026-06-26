from langchain_core.documents import Document
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


def _hit_to_document(hit: dict) -> Document:
    source = hit.get("_source", {})
    text = source.get("text") or source.get("page_content") or ""
    metadata = source.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    return Document(page_content=text, metadata=metadata)


def get_chunk_by_ref(document_id: str, chunk_index: int) -> Document | None:
    store = get_vector_store(use_hybrid=False)
    client = store.client
    response = client.search(
        index=settings.es_index,
        query={
            "bool": {
                "filter": [
                    {"term": {"metadata.document_id.keyword": document_id}},
                    {"term": {"metadata.chunk_index": chunk_index}},
                ]
            }
        },
        size=1,
    )
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    return _hit_to_document(hits[0])


def bm25_search(query: str, *, k: int) -> list[Document]:
    store = get_vector_store(use_hybrid=False)
    client = store.client
    response = client.search(
        index=settings.es_index,
        query={"match": {"text": query}},
        size=k,
    )
    hits = response.get("hits", {}).get("hits", [])
    return [_hit_to_document(hit) for hit in hits]
