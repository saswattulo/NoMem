"""Adapters that plug NoMem into agent frameworks.

Each framework lives in its own module and needs its own extra:

==========================================  =====================================
``nomem.integrations.anthropic``            Claude API (``nomem-ai[anthropic]``)
``nomem.integrations.openai_agents``        OpenAI Agents SDK (``[openai-agents]``)
``nomem.integrations.langchain``            LangChain (``[langchain]``)
``nomem.integrations.langgraph``            LangGraph (``[langgraph]``)
``nomem.integrations.llamaindex``           LlamaIndex (``[llamaindex]``)
``nomem.integrations.crewai``               CrewAI (``[crewai]``)
==========================================  =====================================

They all build on :class:`MemoryToolkit`, which is framework-free and can
be used directly with any tool-calling API.
"""

from ._toolkit import DEFAULT_TOOLS, TOOL_NAMES, MemoryToolkit, ToolSpec

__all__ = ["MemoryToolkit", "ToolSpec", "TOOL_NAMES", "DEFAULT_TOOLS"]
