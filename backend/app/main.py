from contextlib import asynccontextmanager
import logging
import time

from app.observability.langsmith import configure_langsmith, is_langsmith_enabled

configure_langsmith()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.factory import build_agent
from app.auth.seed import ensure_seed_admin
from app.database import Base, engine, get_db
from app.mcp.loader import close_mcp, get_mcp_status, init_mcp
from app.openviking.client import close_openviking, get_openviking_status, init_openviking
from app.routers import admin, auth, chat, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
    allow_origins=[
        "http://localhost:5170",
        "http://127.0.0.1:5170",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    mcp_status = get_mcp_status()
    ov_status = get_openviking_status()
    return {
        "status": "ok",
        "langsmith_tracing": is_langsmith_enabled(),
        **mcp_status,
        **ov_status,
    }
