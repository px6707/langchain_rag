import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.models import Document as DocumentModel
from app.parsing.pipeline import run_parse_pipeline


def test_run_parse_pipeline_python_strategy():
    doc_id = uuid.uuid4()
    document = DocumentModel(
        id=doc_id,
        filename="notes.txt",
        file_path="/tmp/notes.txt",
        file_type="txt",
    )
    session = AsyncMock()
    raw_docs = [Document(page_content="hello", metadata={})]

    with (
        patch("app.parsing.pipeline.load_with_python", return_value=raw_docs),
        patch("app.parsing.pipeline.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)),
        patch("app.parsing.pipeline.index_documents", new_callable=AsyncMock, return_value=1) as mock_index,
    ):
        count = asyncio.run(run_parse_pipeline(session, document))

    assert count == 1
    mock_index.assert_awaited_once()
    assert session.commit.await_count >= 2
