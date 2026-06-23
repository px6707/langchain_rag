from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from langchain.agents import create_agent

from app.agent.checkpointer import get_checkpointer
from app.agent.middleware.retrieval import BASE_SYSTEM_APPENDIX, RAGAgentState
from app.agent.middleware.stack import build_middleware_stack
from app.config import settings
from app.mcp.loader import reload_mcp_sync
from app.skills.loader import load_all_skills, reload_skills
from app.tools.loader import load_all_tools, reload_tools

_agent: CompiledStateGraph | None = None


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_api_base,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        temperature=0.7,
    )


def build_agent() -> CompiledStateGraph:
    global _agent
    load_all_skills()
    tools = load_all_tools()
    llm = _get_llm()
    _agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=BASE_SYSTEM_APPENDIX,
        checkpointer=get_checkpointer(),
        middleware=build_middleware_stack(llm),
        state_schema=RAGAgentState,
    )
    return _agent


def get_agent() -> CompiledStateGraph:
    if _agent is None:
        return build_agent()
    return _agent


def rebuild_agent() -> CompiledStateGraph:
    global _agent
    _agent = None
    reload_skills()
    reload_mcp_sync()
    reload_tools()
    return build_agent()
