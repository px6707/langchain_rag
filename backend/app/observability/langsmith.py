import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

_enabled = False


def configure_langsmith() -> bool:
    """Enable LangSmith tracing for LangChain / LangGraph via environment variables."""
    global _enabled

    if _enabled:
        return True

    if not settings.langsmith_tracing_enabled:
        if settings.langsmith_api_key:
            logger.info(
                "LangSmith tracing disabled (LANGSMITH_TRACING_ENABLED=false); app_env=%s project=%s",
                settings.app_env,
                settings.langsmith_project,
            )
        return False

    if not settings.langsmith_api_key:
        logger.warning("LangSmith tracing requested but LANGSMITH_API_KEY is empty; tracing disabled.")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    _enabled = True
    logger.info(
        "LangSmith tracing enabled: project=%s app_env=%s",
        settings.langsmith_project,
        settings.app_env,
    )
    return True


def is_langsmith_enabled() -> bool:
    return _enabled
