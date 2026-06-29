import logging
import time

from langchain_core.documents import Document
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore

from app.config import settings
from app.services.embedding_service import get_embeddings
from app.services.retrieval_context import get_retrieval_user_filter

logger = logging.getLogger(__name__)

_store_cache: dict[bool, ElasticsearchStore] = {}


def user_es_term(user_id: str) -> dict:
    return {"term": {"metadata.user_id.keyword": user_id}}


def append_es_user_filters(filters: list[dict]) -> list[dict]:
    user_id = get_retrieval_user_filter()
    if user_id is None:
        return filters
    return [*filters, user_es_term(user_id)]


def filter_documents_by_user(docs: list[Document]) -> list[Document]:
    user_id = get_retrieval_user_filter()
    if user_id is None:
        return docs
    return [doc for doc in docs if str(doc.metadata.get("user_id", "")) == user_id]


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


def delete_document_vectors(document_id: str) -> None:
    store = get_vector_store()
    store.delete(filter={"term": {"metadata.document_id.keyword": document_id}})


def delete_document_vectors_except_batch(document_id: str, keep_batch: str) -> None:
    store = get_vector_store()
    store.delete(
        filter={
            "bool": {
                "filter": [{"term": {"metadata.document_id.keyword": document_id}}],
                "must_not": [{"term": {"metadata.index_batch.keyword": keep_batch}}],
            }
        }
    )


def delete_document_vectors_before_generation(document_id: str, keep_generation: int) -> None:
    store = get_vector_store()
    store.delete(
        filter={
            "bool": {
                "filter": [{"term": {"metadata.document_id.keyword": document_id}}],
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "metadata.parse_generation"}}}},
                    {"range": {"metadata.parse_generation": {"lt": keep_generation}}},
                ],
                "minimum_should_match": 1,
            }
        }
    )


def delete_document_vectors_except_batch_and_generation(
    document_id: str,
    keep_batch: str,
    keep_generation: int,
) -> None:
    store = get_vector_store()
    store.delete(
        filter={
            "bool": {
                "filter": [
                    {"term": {"metadata.document_id.keyword": document_id}},
                    {"term": {"metadata.parse_generation": keep_generation}},
                ],
                "must_not": [{"term": {"metadata.index_batch.keyword": keep_batch}}],
            }
        }
    )


def delete_document_vectors_except_batch_and_generation_with_retry(
    document_id: str,
    keep_batch: str,
    keep_generation: int,
    *,
    max_attempts: int | None = None,
) -> None:
    attempts = max_attempts or settings.index_delete_retry_attempts
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            delete_document_vectors_except_batch_and_generation(
                document_id,
                keep_batch,
                keep_generation,
            )
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Delete old index batches failed (attempt %s/%s): document_id=%s generation=%s",
                attempt + 1,
                attempts,
                document_id,
                keep_generation,
            )
            if attempt < attempts - 1:
                time.sleep(0.5 * (2**attempt))
    if last_exc is not None:
        logger.exception(
            "Failed to delete old index batches after retries: document_id=%s generation=%s",
            document_id,
            keep_generation,
        )
        raise last_exc


def delete_document_vectors_except_batch_with_retry(
    document_id: str,
    keep_batch: str,
    *,
    max_attempts: int | None = None,
) -> None:
    attempts = max_attempts or settings.index_delete_retry_attempts
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            delete_document_vectors_except_batch(document_id, keep_batch)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Delete old index batches failed (attempt %s/%s): document_id=%s",
                attempt + 1,
                attempts,
                document_id,
            )
            if attempt < attempts - 1:
                time.sleep(0.5 * (2**attempt))
    if last_exc is not None:
        logger.exception("Failed to delete old index batches after retries: document_id=%s", document_id)
        raise last_exc


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
                "filter": append_es_user_filters(
                    [
                        {"term": {"metadata.document_id.keyword": document_id}},
                        {"term": {"metadata.chunk_index": chunk_index}},
                    ]
                )
            }
        },
        size=1,
    )
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    return _hit_to_document(hits[0])


def fetch_chunks_by_page(
    document_id: str,
    page_number: int,
    *,
    max_chunks: int | None = None,
) -> list[Document]:
    limit = max_chunks or settings.retrieval_page_expand_max_chunks
    store = get_vector_store(use_hybrid=False)
    client = store.client
    response = client.search(
        index=settings.es_index,
        query={
            "bool": {
                "filter": append_es_user_filters(
                    [
                        {"term": {"metadata.document_id.keyword": document_id}},
                        {"term": {"metadata.page_number": page_number}},
                    ]
                )
            }
        },
        size=limit,
        sort=[{"metadata.chunk_index": {"order": "asc", "unmapped_type": "long"}}],
    )
    hits = response.get("hits", {}).get("hits", [])
    return [_hit_to_document(hit) for hit in hits]


def fetch_chunks_by_asr_segment(
    document_id: str,
    segment_index: int,
    *,
    max_chunks: int | None = None,
) -> list[Document]:
    limit = max_chunks or settings.retrieval_asr_segment_expand_max_chunks
    store = get_vector_store(use_hybrid=False)
    client = store.client
    response = client.search(
        index=settings.es_index,
        query={
            "bool": {
                "filter": append_es_user_filters(
                    [{"term": {"metadata.document_id.keyword": document_id}}]
                ),
                "should": [
                    {"term": {"metadata.asr_segment_index": segment_index}},
                    {"term": {"metadata.segment_index": segment_index}},
                ],
                "minimum_should_match": 1,
            }
        },
        size=limit,
        sort=[{"metadata.chunk_index": {"order": "asc", "unmapped_type": "long"}}],
    )
    hits = response.get("hits", {}).get("hits", [])
    return [_hit_to_document(hit) for hit in hits]


def bm25_search(query: str, *, k: int) -> list[Document]:
    store = get_vector_store(use_hybrid=False)
    client = store.client
    user_id = get_retrieval_user_filter()
    if user_id is None:
        es_query: dict = {"match": {"text": query}}
    else:
        es_query = {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [user_es_term(user_id)],
            }
        }
    response = client.search(
        index=settings.es_index,
        query=es_query,
        size=k,
    )
    hits = response.get("hits", {}).get("hits", [])
    return filter_documents_by_user([_hit_to_document(hit) for hit in hits])
