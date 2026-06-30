from contextlib import asynccontextmanager
import logging
import time

from app.observability.langsmith import configure_langsmith, is_langsmith_enabled
from app.observability.log_context import TraceContextFilter

configure_langsmith()

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s [%(name)s] "
        "trace_id=%(trace_id)s session_id=%(session_id)s user_id=%(user_id)s "
        "%(message)s"
    ),
)
logging.getLogger().addFilter(TraceContextFilter())

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.factory import build_agent
from app.auth.seed import ensure_seed_admin
from app.config import settings
from app.db.migrate import run_migrations_async
from app.database import get_db
from app import models as _models  # noqa: F401 — register ORM tables
from app.health import build_health_payload
from app.mcp.loader import close_mcp, get_mcp_status, init_mcp
from app.openviking.client import close_openviking, get_openviking_status, init_openviking
from app.routers import admin, auth, chat, documents, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_migrations_async()

    async for db in get_db():
        await ensure_seed_admin(db)
        break

    await init_checkpointer()
    await init_openviking()
    await init_mcp()
    app.state.agent = build_agent()

    yield

    await close_mcp()
    await close_openviking()
    await close_checkpointer()


app = FastAPI(title="LangChain RAG API", lifespan=lifespan)

logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(feedback.router)


@app.get("/health")
async def health():
    extra = {
        "langsmith_tracing": is_langsmith_enabled(),
        **get_mcp_status(),
        **get_openviking_status(),
    }
    body, status_code = await build_health_payload(extra=extra)
    return JSONResponse(content=body, status_code=status_code)
