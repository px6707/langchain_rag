from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    checkpointer_database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    es_url: str = "http://localhost:9200"
    es_index: str = "rag_documents"

    upload_dir: str = "./uploads"

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4  # legacy; final k uses rerank_top_n
    retrieval_score_threshold: float = 0.7
    retrieval_hybrid_enabled: bool = True
    retrieval_rrf_enabled: bool = True
    retrieval_fetch_k: int = 20
    rerank_enabled: bool = True
    rerank_api_base: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    rerank_top_n: int = 4

    summarization_trigger_messages: int = 30
    summarization_keep_messages: int = 15

    pii_enabled: bool = True
    pii_types: str = "email,credit_card,ip"
    pii_strategy: str = "redact"

    hitl_enabled: bool = True
    hitl_tools: str = "get_current_time,send_email,run_skill_script"

    smtp_host: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_use_tls: bool = True

    tool_retry_max_retries: int = 2
    tool_retry_initial_delay: float = 1.0

    todo_list_enabled: bool = True

    skills_enabled: bool = True
    skills_dir: str = "./skills"
    skills_allowlist: str = ""

    skill_script_enabled: bool = True
    skill_script_timeout_seconds: int = 30
    skill_script_max_timeout_seconds: int = 300
    skill_script_max_output_bytes: int = 512_000
    skill_script_max_file_bytes: int = 1_048_576
    skill_script_allowed_extensions: str = ".py,.sh"

    mcp_enabled: bool = True
    mcp_servers_file: str = "./mcp_servers.json"
    mcp_tool_allowlist: str = ""

    langsmith_tracing_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "langchain-rag"
    langsmith_endpoint: str = ""

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    admin_username: str = "admin"
    admin_password: str = "admin123"

    openviking_enabled: bool = False
    openviking_path: str = "./openviking_data"
    openviking_config_file: str = ""
    openviking_account_id: str = "default"
    openviking_find_limit: int = 5
    openviking_find_score_threshold: float = 0.3
    openviking_commit_every_messages: int = 20


settings = Settings()
