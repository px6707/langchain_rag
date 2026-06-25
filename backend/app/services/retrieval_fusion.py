from langchain_core.documents import Document

from app.config import settings


def _doc_key(doc: Document) -> tuple[str, str]:
    return (doc.page_content, str(doc.metadata.get("filename", "")))


def rrf_fuse(
    lists: list[list[Document]],
    *,
    rrf_k: int | None = None,
    list_weights: list[float] | None = None,
) -> list[Document]:
    if not lists:
        return []

    k = rrf_k if rrf_k is not None else settings.retrieval_fusion_rrf_k
    weights = list_weights or [1.0] * len(lists)

    scores: dict[tuple[str, str], float] = {}
    doc_map: dict[tuple[str, str], Document] = {}

    for list_idx, doc_list in enumerate(lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, doc in enumerate(doc_list):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            doc_map[key] = doc

    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_map[key] for key in ordered_keys]
