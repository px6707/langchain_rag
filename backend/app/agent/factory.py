from langgraph.graph.state import CompiledStateGraph

from langchain.agents import create_agent

from app.agent.checkpointer import get_checkpointer
from app.agent.middleware.retrieval import BASE_SYSTEM_APPENDIX, RAGAgentState
from app.agent.middleware.stack import build_middleware_stack
from app.mcp.loader import reload_mcp_sync
from app.services.llm_service import get_llm
from app.skills.loader import load_all_skills, reload_skills
from app.tools.loader import load_all_tools, reload_tools

_agent: CompiledStateGraph | None = None


def _get_llm():
    return get_llm()


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
