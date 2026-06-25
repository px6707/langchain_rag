from unittest.mock import MagicMock, patch

from app.services.hyde_service import score_hyde_quality, should_use_hyde


def test_should_use_hyde_by_score():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.side_effect = [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    with (
        patch("app.services.hyde_service.get_embeddings", return_value=mock_embeddings),
        patch("app.services.hyde_service.settings.retrieval_hyde_min_score", 0.5),
    ):
        assert should_use_hyde("hyde answer about rag systems", "what is rag") is True


def test_score_hyde_quality_orthogonal_vectors():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.side_effect = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    with patch("app.services.hyde_service.get_embeddings", return_value=mock_embeddings):
        assert score_hyde_quality("a", "b") == 0.0
