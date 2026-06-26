from app.parsing.chunking_config import get_strategy, load_chunking_config


def test_load_chunking_config_defaults():
    config = load_chunking_config()
    assert config["strategies"]["table"] == "row_split_header_repeat"
    assert config["overrides"]["chunk_size"] == 500


def test_get_strategy_for_block_types():
    assert get_strategy("table") == "row_split_header_repeat"
    assert get_strategy("html_table") == "html_row_split"
    assert get_strategy("audio_transcript") == "segment_preserve"
