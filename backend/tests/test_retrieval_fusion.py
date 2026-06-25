from langchain_core.documents import Document

from app.services.retrieval_fusion import rrf_fuse


def test_rrf_fuse_promotes_docs_in_multiple_lists():
    doc_a = Document("shared", metadata={"filename": "a.pdf"})
    doc_b = Document("only-b", metadata={"filename": "b.pdf"})
    doc_c = Document("only-c", metadata={"filename": "c.pdf"})

    list1 = [doc_a, doc_b]
    list2 = [doc_a, doc_c]

    fused = rrf_fuse([list1, list2], rrf_k=60, list_weights=[2.0, 1.0])

    assert fused[0].page_content == "shared"
    assert len(fused) == 3


def test_rrf_fuse_respects_list_weights():
    doc_a = Document("a", metadata={"filename": "a.pdf"})
    doc_b = Document("b", metadata={"filename": "b.pdf"})

    fused = rrf_fuse([[doc_a], [doc_b]], rrf_k=60, list_weights=[3.0, 1.0])
    assert fused[0].page_content == "a"
