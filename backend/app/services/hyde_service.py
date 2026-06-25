import math

from app.config import settings
from app.services.embedding_service import get_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_hyde_quality(hyde_doc: str, query: str) -> float:
    hyde = hyde_doc.strip()
    q = query.strip()
    if not hyde or not q:
        return 0.0
    embeddings = get_embeddings()
    hyde_vec = embeddings.embed_query(hyde)
    query_vec = embeddings.embed_query(q)
    return _cosine_similarity(hyde_vec, query_vec)


def should_use_hyde(hyde_doc: str, query: str) -> bool:
    return score_hyde_quality(hyde_doc, query) >= settings.retrieval_hyde_min_score
