from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 从 .env 加载环境变量；未识别的键忽略
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM（对话 Agent 主模型）---
    llm_api_base: str = "https://api.openai.com/v1"  # Chat 模型 API 根地址（兼容 OpenAI 协议的服务）
    llm_api_key: str = ""  # Chat 模型 API 密钥
    llm_model: str = "gpt-4o-mini"  # Agent 对话使用的模型名称

    # --- 小模型（检索路由、query 改写、grounding 校验等辅助任务）---
    small_llm_api_base: str = ""  # 小模型 API 地址；空则回退 llm_api_base
    small_llm_api_key: str = ""  # 小模型 API 密钥；空则回退 llm_api_key
    small_llm_model: str = ""  # 小模型名称；空则回退 llm_model
    small_llm_temperature: float = 0.0  # 小模型默认温度（建议 0 以保证稳定输出）

    # --- Embedding（文档向量化与检索）---
    embedding_api_base: str = "https://api.openai.com/v1"  # Embedding API 根地址
    embedding_api_key: str = ""  # Embedding API 密钥
    embedding_model: str = "text-embedding-3-small"  # 文档切块与检索用的向量模型

    # --- 数据库 ---
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"  # 业务库连接串（asyncpg，用户/文档等）
    checkpointer_database_url: str = "postgresql://rag:rag@localhost:5433/rag"  # LangGraph 会话 checkpoint 库（同步驱动）
    auto_db_migrate: bool = True  # 启动时自动执行 Alembic upgrade head
    cors_origins: str = (
        "http://localhost:5170,http://127.0.0.1:5170,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # --- Elasticsearch（向量 + 全文检索）---
    es_url: str = "http://localhost:9200"  # Elasticsearch 集群地址
    es_index: str = "rag_documents"  # 存放文档切片的索引名

    # --- 文档上传与切块 ---
    upload_dir: str = "./uploads"  # 用户上传文件的本地存储目录
    chunk_size: int = 500  # 文档切块时的目标字符数
    chunk_overlap: int = 50  # 相邻切块之间的重叠字符数，减少边界信息丢失
    parse_max_file_mb: int = 200  # 单文件上传上限（MB，对齐 MinerU 云端）
    parse_worker_poll_sec: float = 2.0  # Worker 拉取队列间隔（秒）
    parse_max_attempts: int = 3  # 解析失败最大重试次数
    parse_job_stale_timeout_sec: int = 7200  # running 超时回收（秒），应 >= MINERU_POLL_TIMEOUT_SEC；主要用于告警/日志，回收触发以 lease_expires_at 为主
    parse_job_lease_ttl_sec: int = 120  # Worker claim 后租约有效期（秒），超时未续租可被 reclaim
    parse_job_heartbeat_sec: int = 30  # Worker / pipeline 续租心跳间隔（秒）
    parse_job_stale_grace_sec: int = 60  # 租约过期后再等待的宽限期（秒），过后才 reclaim stale job
    parse_job_stale_auto_retry: bool = True  # stale 回收后是否自动新建 pending job 重新解析
    table_chunk_rows: int = 8  # 表格分块时每 chunk 数据行数
    chunking_config_path: str = "./chunking_config.yaml"
    index_delete_retry_attempts: int = 3

    # --- MinerU 云端解析 ---
    mineru_api_base: str = "https://mineru.net"
    mineru_api_token: str = ""
    mineru_model_version: str = "vlm"  # pipeline / vlm / MinerU-HTML
    mineru_poll_interval_sec: float = 5.0
    mineru_poll_timeout_sec: int = 3600

    # --- ASR（OpenAI 兼容 /v1/audio/transcriptions）---
    asr_api_base: str = ""
    asr_api_key: str = ""
    asr_model: str = "whisper-1"
    asr_use_verbose_json: bool = True
    asr_fallback_segment_sec: int = 120
    asr_proactive_split_enabled: bool = True
    asr_proactive_split_min_duration_sec: int = 600
    asr_proactive_segment_sec: int = 600
    asr_proactive_max_file_mb: int = 25

    # --- VLM 视觉摘要 ---
    vlm_api_base: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""

    # --- 视频处理 ---
    video_frame_mode: str = "scene"  # legacy; planner 主路径不再依赖
    video_scene_threshold: float = 0.3
    video_frame_interval_sec: float = 30.0  # legacy
    video_max_frames: int = 60  # legacy
    video_frame_budget: int = 96
    video_min_interval_sec: float = 60.0
    video_asr_anchor_enabled: bool = True
    video_asr_merge_gap_sec: float = 90.0
    video_scene_gap_sec: float = 120.0
    video_scene_extra_max: int = 24
    video_timestamp_merge_sec: float = 3.0
    video_vlm_enabled: bool = False
    video_vlm_min_ocr_chars: int = 80
    video_frame_concurrency: int = 4
    video_dedupe_enabled: bool = True
    video_dedupe_hamming_threshold: int = 5
    video_extract_batch_window_sec: float = 120.0
    video_extract_select_margin_sec: float = 0.5
    video_extract_ffmpeg_threads: int = 1

    # --- 检索与 Rerank ---
    retrieval_k: int = 4  # 遗留项；实际最终返回条数由 rerank_top_n 控制
    retrieval_score_threshold: float = 0.7  # Rerank 分数阈值，低于此值的文档会被过滤
    retrieval_hybrid_enabled: bool = True  # 是否启用 ES 混合检索（向量 + BM25）
    retrieval_rrf_enabled: bool = True  # 混合检索时是否用 RRF 融合向量与 BM25 结果
    retrieval_fetch_k: int = 20  # 启用 Rerank 时 ES 初召回条数（粗召回后再精排）
    rerank_enabled: bool = True  # 是否启用 HTTP Rerank 精排
    rerank_api_base: str = ""  # Rerank 服务地址；为空则回退 embedding_api_base
    rerank_api_key: str = ""  # Rerank 服务密钥；为空则回退 embedding_api_key
    rerank_model: str = ""  # Rerank 模型名；为空则跳过精排
    rerank_top_n: int = 4  # Rerank 后保留的文档数；未启用 Rerank 时作为 ES 的 k

    # --- 检索路由与 Query 变换 ---
    retrieval_routing_enabled: bool = True  # 是否启用 LLM 检索路由（闲聊/工具调用可跳过 ES）
    retrieval_history_messages: int = 6  # 路由与多轮改写时可见的最近消息条数
    retrieval_multi_query_count: int = 3  # multi_query 策略下额外生成的同义 query 数量上限
    retrieval_max_sub_questions: int = 4  # decompose 策略下子问题数量上限
    retrieval_router_tool_names: str = (  # 写入路由 prompt 的工具名列表，用于识别纯工具意图
        "get_current_time,send_email,run_skill_script,load_skill,write_todos"
    )
    retrieval_empty_fallback_enabled: bool = True  # none 策略零结果时自动升级为 multi_query 重试
    retrieval_tool_context_max_chars: int = 500  # 路由历史中 ToolMessage 内容截断长度
    retrieval_per_query_k_min: int = 5  # 多 query 时每路召回 k 的下限
    retrieval_fusion_rrf_k: int = 60  # 多路召回 RRF 融合的 k 常数
    retrieval_rrf_parent_weight: float = 1.5  # standalone_query 来源 list 的 RRF 加权系数
    retrieval_history_turns: int = 4  # 路由/改写可见的最近对话轮数（Human 消息计轮）
    retrieval_doc_intent_keywords: str = (  # postcheck 强制 retrieve 的文档意图关键词，逗号分隔
        "文档,文件,条款,合同,报告,pdf,上传,资料,检索,章节,附件"
    )
    retrieval_fallback_threshold_ratio: float = 0.5  # 分级 fallback 降阈值比例
    retrieval_fallback_k_multiplier: float = 2.0  # 分级 fallback 最后一级 k 放大倍数
    retrieval_fallback_max_tiers: int = 4  # 分级 fallback 最大 tier 数
    retrieval_hyde_min_score: float = 0.3  # HyDE 假设文档与 query 的 embedding 相似度下限
    retrieval_page_expand_enabled: bool = True  # 粗召回后扩展同 document_id + page_number 的 sibling chunk
    retrieval_page_expand_max_chunks: int = 32  # 每页扩展最多拉取的 chunk 数
    retrieval_asr_segment_expand_enabled: bool = True
    retrieval_asr_segment_expand_max_chunks: int = 32

    # --- 对话历史压缩（SummarizationMiddleware）---
    summarization_trigger_messages: int = 30  # 消息数超过此值时触发历史摘要
    summarization_keep_messages: int = 15  # 摘要后保留的最近消息条数

    # --- PII 脱敏（PIIMiddleware）---
    pii_enabled: bool = True  # 是否启用输入/输出 PII 检测与处理
    pii_types: str = "email,credit_card,ip"  # 需检测的 PII 类型，逗号分隔
    pii_strategy: str = "redact"  # PII 处理策略，如 redact（替换/遮盖）

    # --- 人工审批（HumanInTheLoopMiddleware）---
    hitl_enabled: bool = True  # 是否对敏感工具调用启用人工审批
    hitl_tools: str = "get_current_time,send_email,run_skill_script"  # 需要审批的工具名，逗号分隔

    # --- 邮件发送（send_email 工具）---
    smtp_host: str = "smtp.example.com"  # SMTP 服务器主机名
    smtp_port: int = 587  # SMTP 端口
    smtp_use_tls: bool = True  # 是否使用 STARTTLS 加密连接

    # --- 工具调用重试（ToolRetryMiddleware）---
    tool_retry_max_retries: int = 2  # 工具失败后的最大重试次数
    tool_retry_initial_delay: float = 1.0  # 首次重试前的等待秒数（指数退避起点）

    # --- 任务规划（TodoListMiddleware）---
    todo_list_enabled: bool = True  # 是否启用 write_todos 多步骤任务规划

    # --- Skills ---
    skills_enabled: bool = True  # 是否启用 Skills 中间件与 load_skill 工具
    skills_dir: str = "./skills"  # Skills 定义文件所在目录
    skills_allowlist: str = ""  # 允许加载的 skill 名白名单，逗号分隔；空表示不限制

    # --- Skill 脚本执行（run_skill_script 工具）---
    skill_script_enabled: bool = True  # 是否在 prompt 中暴露脚本执行能力
    skill_script_timeout_seconds: int = 30  # 单次脚本执行的默认超时（秒）
    skill_script_max_timeout_seconds: int = 300  # 允许请求的最大超时上限（秒）
    skill_script_max_output_bytes: int = 512_000  # 脚本 stdout/stderr 最大捕获字节数
    skill_script_max_file_bytes: int = 1_048_576  # 可读取的 skill 文件最大字节数
    skill_script_allowed_extensions: str = ".py,.sh"  # 允许执行的脚本扩展名，逗号分隔

    # --- MCP 外部工具 ---
    mcp_enabled: bool = True  # 是否加载 MCP 服务器提供的工具
    mcp_servers_file: str = "./mcp_servers.json"  # MCP 服务器配置文件路径
    mcp_tool_allowlist: str = ""  # MCP 工具名白名单，逗号分隔；空表示加载全部

    # --- 运行环境 ---
    app_env: Literal["dev", "prod", "test"] = "dev"  # APP_ENV；LangSmith project 默认为 rag-{app_env}

    # --- LangSmith 可观测性 ---
    langsmith_tracing_enabled: bool | None = None  # None = 有 API key 时自动开启
    langsmith_api_key: str = ""  # LangSmith API 密钥
    langsmith_project: str = ""  # 空则 rag-{app_env}
    langsmith_endpoint: str = ""  # 自定义 LangSmith API 端点；空则使用官方默认
    langsmith_metadata_max_chunks: int = 20  # turn metadata 中 chunk_refs 条数上限
    langsmith_metadata_snippet_chars: int = 300  # chunk_snippets 单条最大字符
    langsmith_stage_tracing_enabled: bool = True  # Phase B 命名 span；关闭则 stage no-op

    # --- JWT 认证 ---
    jwt_secret: str = "change-me-in-production"  # JWT 签名密钥（生产环境务必修改）
    jwt_algorithm: str = "HS256"  # JWT 签名算法
    jwt_expire_minutes: int = 60 * 24  # 访问令牌有效期（分钟）

    # --- 初始管理员账号（首次启动/bootstrap 用）---
    admin_username: str = "admin"  # 默认管理员用户名
    admin_password: str = "admin123"  # 默认管理员密码

    # --- OpenViking 长期记忆 ---
    openviking_enabled: bool = False  # 是否启用 OpenViking 用户记忆检索与归档
    openviking_path: str = "./openviking_data"  # OpenViking 本地数据目录
    openviking_config_file: str = ""  # 可选 OpenViking 配置文件；空则仅用 path 初始化
    openviking_account_id: str = "default"  # OpenViking 账户标识前缀
    openviking_find_limit: int = 5  # 每次记忆检索返回条数上限
    openviking_find_score_threshold: float = 0.3  # 记忆相似度分数阈值
    openviking_commit_every_messages: int = 20  # 会话消息数达到此值时归档到长期记忆

    # --- 答案 Grounding 校验 ---
    grounding_enabled: bool = True  # 是否对生成答案做检索支撑校验
    grounding_min_supported_ratio: float = 0.8  # 支持率低于此值标为 partial
    grounding_fail_ratio: float = 0.5  # 支持率低于此值标为 not_supported
    grounding_max_claims: int = 8  # 最多抽取 claim 数

    @model_validator(mode="after")
    def _apply_langsmith_defaults(self) -> Self:
        if self.langsmith_tracing_enabled is None:
            object.__setattr__(
                self,
                "langsmith_tracing_enabled",
                bool(self.langsmith_api_key.strip()),
            )
        if not self.langsmith_project.strip():
            object.__setattr__(self, "langsmith_project", f"rag-{self.app_env}")
        return self


settings = Settings()
