import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import get_settings
from app.llm import LLMError, resolve_model
from app.rag import build_context
from app.tools import TOOLS

settings = get_settings()

MAX_TOOL_CALLS = 4

TOOL_MAP = {tool.name: tool for tool in TOOLS}


class AgentState(MessagesState):
    tool_calls: int
    rag_context: str


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that preserves `reasoning_content` from OpenAI-compatible servers.

    langchain-openai deliberately drops non-standard fields like
    `reasoning_content`, so we re-attach it to the chunk's additional_kwargs.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> Any:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if reasoning:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    if not text or "{" not in text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str):
        return None
    arguments = obj.get("arguments") or obj.get("args") or {}
    return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}


def _system_prompt(rag_context: str = "") -> str:
    lines = [
        "You are a helpful assistant with access to tools.",
        "",
        "Available tools:",
    ]
    for tool in TOOLS:
        lines.append(f"- {tool.name}: {tool.description}")
    lines += [
        "",
        "To use a tool, reply with ONLY a JSON object in this exact format:",
        '{"tool": "<tool_name>", "arguments": {<arguments as JSON>}}',
        "",
        "Example:",
        "User: What is the weather in Paris?",
        'Assistant: {"tool": "get_weather", "arguments": {"city": "Paris"}}',
        "",
        'After a "Tool result" message, use the result to answer the user.',
        "If no tool is needed, answer the user's question directly.",
    ]
    if rag_context:
        lines += [
            "",
            "Document excerpts retrieved from the Digital DAAK user manual:",
            "",
            rag_context,
            "",
            "When the question is about Digital DAAK, base your answer on these excerpts",
            "and cite pages like (page 7). If the excerpts do not contain the answer, say",
            "so instead of inventing details.",
        ]
    return "\n".join(lines)


async def _call_model(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    context = state.get("rag_context") or ""
    response = await llm.ainvoke(
        [SystemMessage(content=_system_prompt(context)), *state["messages"]]
    )
    return {"messages": [response]}


async def _run_tool(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    call = _extract_tool_call(_text(last.content))
    tool = TOOL_MAP.get(call["name"]) if call else None

    if tool is None:
        available = ", ".join(TOOL_MAP)
        observation = f"Tool error: unknown tool. Available tools: {available}."
    else:
        schema_props = (tool.args or {}).get("properties") or {}
        if not schema_props:
            arguments = {}
        else:
            arguments = {
                key: value
                for key, value in call["arguments"].items()
                if key in schema_props
            }
        try:
            result = await tool.ainvoke(arguments)
            observation = f"Tool result for {call['name']}: {result}"
        except Exception as exc:  # noqa: BLE001
            observation = f"Tool error for {call['name']}: {exc}"

    return {
        "messages": [HumanMessage(content=observation)],
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return END
    call = _extract_tool_call(_text(last.content))
    if call is None:
        return END
    if state.get("tool_calls", 0) >= MAX_TOOL_CALLS:
        return END
    return "tools"


def _build_graph(llm: ChatOpenAI) -> Any:
    async def call_model(state: AgentState) -> dict[str, Any]:
        return await _call_model(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", _run_tool)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = None


async def get_graph() -> Any:
    global _graph
    if _graph is None:
        model = await resolve_model()
        llm = ReasoningChatOpenAI(
            model=model,
            base_url=f"{settings.llama_base_url}/v1",
            api_key="llama-cpp",
            timeout=settings.llama_timeout,
            temperature=0.6,
        )
        _graph = _build_graph(llm)
    return _graph


_CALL_LEAD = re.compile(
    r"^\s*(?:(?:assistant|bot)\s*:\s*)?(?:```(?:json)?\s*)?", re.IGNORECASE
)


def _feed(entry: dict[str, Any], text: str) -> list[tuple[str, str]]:
    """Buffer content until we can tell a tool-call JSON from a real answer."""
    if entry["mode"] == "suppressed":
        return []
    if entry["mode"] == "streaming":
        return [("delta", text)]

    buffer = entry["buffer"] + text
    probe = _CALL_LEAD.sub("", buffer)

    if not probe:
        if len(buffer) < 40:
            entry["buffer"] = buffer
            return []
    elif probe.startswith("{"):
        if _extract_tool_call(buffer) is not None:
            entry["mode"] = "suppressed"
            entry["buffer"] = ""
            return []
        if "}" not in probe and len(probe) < 400:
            entry["buffer"] = buffer
            return []

    entry["mode"] = "streaming"
    entry["buffer"] = ""
    return [("delta", buffer)] if buffer else []


def _flush(entry: dict[str, Any]) -> list[tuple[str, str]]:
    if entry["mode"] != "pending" or not entry["buffer"]:
        return []
    buffer = entry["buffer"]
    if not buffer.strip() or _extract_tool_call(buffer) is not None:
        return []
    return [("delta", buffer)]


async def run_agent_stream(
    messages: list[dict[str, str]],
) -> AsyncIterator[tuple[str, str]]:
    graph = await get_graph()
    lc_messages = [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=message["content"])
        for message in messages
    ]

    rag_context = ""
    last_question = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    if last_question:
        try:
            rag_context = await asyncio.to_thread(build_context, last_question)
        except Exception:  # noqa: BLE001
            rag_context = ""

    runs: dict[str, dict[str, Any]] = {}

    try:
        async for event in graph.astream_events(
            {"messages": lc_messages, "tool_calls": 0, "rag_context": rag_context},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                entry = runs.setdefault(event["run_id"], {"buffer": "", "mode": "pending"})
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
                if reasoning:
                    yield ("reasoning", reasoning)
                text = _text(chunk.content)
                if text:
                    for item in _feed(entry, text):
                        yield item
            elif kind == "on_chat_model_end":
                entry = runs.pop(event["run_id"], None)
                if entry is not None:
                    for item in _flush(entry):
                        yield item
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"agent request failed: {exc}") from exc
