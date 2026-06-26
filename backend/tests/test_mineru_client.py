from unittest.mock import MagicMock

from app.parsing.mineru_client import _upload_file


def test_upload_file_uses_streaming_body(tmp_path):
    file_path = tmp_path / "a.pdf"
    file_path.write_bytes(b"%PDF-1.4 content")

    put_kwargs = {}

    def mock_put(url, **kwargs):
        put_kwargs.update(kwargs)
        response = MagicMock()
        response.status_code = 200
        return response

    client = MagicMock()
    client.put.side_effect = mock_put

    _upload_file(client, "https://upload.example/1", str(file_path))
    content = put_kwargs.get("content")
    assert hasattr(content, "read")
