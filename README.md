# LangChain RAG 项目

前后端分离的 RAG（检索增强生成）项目，支持文档上传、向量化存储与智能问答。

## 技术栈

- **前端**: Vue 3 + Element Plus + Tailwind CSS
- **后端**: FastAPI + LangChain
- **数据库**: PostgreSQL（元数据）+ Elasticsearch（向量存储）
- **LLM**: OpenAI 兼容 API

## 快速开始

### 方式 A：Docker（推荐）

```bash
cp backend/.env.docker.example backend/.env
# 编辑 backend/.env，填入 LLM / Embedding API 密钥

docker compose up -d --build
```

| 服务 | 地址 |
|------|------|
| 前端 | http://localhost:5170 |
| 后端 API | http://localhost:8000 |
| PostgreSQL | localhost:5433 |
| Elasticsearch | http://localhost:9200 |
| Mailpit（SMTP 测试） | http://localhost:8025 |

**分目录启动**：

```bash
# 仅后端（含 DB / ES / worker / Mailpit）
cd backend && docker compose up -d --build

# 仅前端（需 backend 已创建 rag-network）
cd frontend && docker compose up -d --build
```

LLM、MinerU、ASR 等仍为外部 API，在 `backend/.env` 中配置。视频抽帧由 parse-worker 容器内 **ffmpeg** 完成（非独立服务）。

**Docker 镜像拉取**：默认经 DaoCloud 镜像站拉取基础镜像；Mailpit 用 `ghcr.io`。Apple Silicon 上 backend 使用 `platform: linux/amd64`（`office-oxide` 无 arm64 Linux wheel）。海外网络可在 `.env.docker.example` 中改回官方镜像名。

### 方式 B：本地开发

#### 1. 启动基础设施

```bash
docker compose up -d postgres elasticsearch mailpit
```

> 也可 `cd backend && docker compose up -d postgres elasticsearch mailpit`。PostgreSQL 映射到本机 **5433** 端口。

#### 2. 配置环境变量

```bash
cp .env.example backend/.env
# 编辑 backend/.env，填入 LLM 和 Embedding API 配置
```

#### 3. 启动后端

需要 **Python 3.12**（与 parse-worker Docker 镜像一致，`office-oxide` 需此版本）。

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5170 ，使用 `.env` 中配置的管理员账号登录（默认 `admin` / `admin123`）。

## 用户与权限

- **登录页** `/login`：用户名 + 密码，无自助注册
- **全站保护**：对话、文档上传等业务页需登录
- **管理中心** `/admin/users`：管理员可增删改用户、禁用/启用、重置密码
- **种子管理员**：首次启动且数据库无用户时，按 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 自动创建

## 数据库迁移

业务库 schema 由 [Alembic](https://alembic.sqlalchemy.org/) 管理。API 与 parse worker 启动时会自动执行 `upgrade head`（可通过 `AUTO_DB_MIGRATE=false` 关闭）。

**修改 ORM 模型后**（在 `backend/` 目录）：

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

**从旧版升级**（schema 已是最新、但从未跑过 Alembic）：

```bash
cd backend && alembic stamp head
```

详见 [`backend/PARSING.md`](backend/PARSING.md) 中的部署说明。

## 可观测性（LangSmith）

配置 `LANGSMITH_API_KEY` 后自动开启 tracing，project 默认为 `rag-{APP_ENV}`（如 `rag-dev` / `rag-prod`）。支持 turn 级 RAG metadata、结构化用户反馈（👍/👎 + 原因）、Online Eval 与告警配置，详见 [`backend/OBSERVABILITY.md`](backend/OBSERVABILITY.md)（含 Phase A 运维清单）。

## 架构说明

- **Agent**: LangChain `create_agent` + `AsyncPostgresSaver` 持久化多轮对话
- **检索**: ES 混合检索（BM25 + 向量 RRF）+ 云 rerank 精排；详见 [`backend/RETRIEVAL.md`](backend/RETRIEVAL.md)
- **Tools**: 在 `backend/app/tools/` 下新增 `.py` 文件并用 `@tool` 装饰，重启后自动加载
- **MCP**: 在 `backend/mcp_servers.json` 配置 MCP Server，启动时自动加载外部工具（见 [`backend/MCP.md`](backend/MCP.md)）

### 新增 Tool

在 [`backend/app/tools/`](backend/app/tools/) 创建新文件即可，例如：

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """工具描述。"""
    return "result"
```


- **对话页** (`/`): 基于已上传文档的 RAG 问答
- **上传页** (`/upload`): 上传 PDF/TXT/MD/DOCX 文件，自动解析并存储到向量数据库

## API 端点

| 方法 | 路径 | 说明 | 鉴权 |
|------|------|------|------|
| POST | `/api/auth/login` | 登录，返回 JWT | 公开 |
| GET | `/api/auth/me` | 当前用户信息 | 需登录 |
| GET | `/api/admin/users` | 用户列表 | 管理员 |
| POST | `/api/admin/users` | 创建用户 | 管理员 |
| PATCH | `/api/admin/users/{id}` | 更新用户 | 管理员 |
| DELETE | `/api/admin/users/{id}` | 删除用户 | 管理员 |
| POST | `/api/documents/upload` | 上传文件 | 需登录 |
| GET | `/api/documents` | 文档列表 | 需登录 |
| DELETE | `/api/documents/{id}` | 删除文档 | 需登录 |
| POST | `/api/chat` | RAG 对话 | 需登录 |
| GET | `/api/chat/history` | 对话历史 | 需登录 |
| GET | `/api/chat/tools` | 已注册工具列表（本地 + MCP） | 需登录 |
| GET | `/health` | 健康检查（含 Postgres / Elasticsearch / MCP 状态） | 公开 |
