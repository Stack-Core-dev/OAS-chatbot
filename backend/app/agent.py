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
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str):
        return None
    arguments = obj.get("arguments") or obj.get("args")
    if not isinstance(arguments, dict):
        arguments = {}
    if not arguments:
        arguments = {
            key: value
            for key, value in obj.items()
            if key not in ("tool", "name", "arguments", "args")
        }
    return {"name": name, "arguments": arguments}


_ENVELOPE_HINT = re.compile(r'"\s*(?:response|answer|text)\s*"\s*:', re.IGNORECASE)


def _extract_direct_response(text: str) -> str | None:
    """Unwrap {"tool": null, "response": "..."} style envelopes."""
    if not text or "{" not in text:
        return None
    start = text.find("{")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("tool") or obj.get("name"):
        return None
    for key in ("response", "answer", "text", "content"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _system_prompt(rag_context: str = "") -> str:
    lines = [
        "You are a helpful AWT-assistant with access to tools.You reply in steps as a guiding assistant",
        "",
        "Reply only in English or Roman Urdu, matching the language the user",
        "writes in. Never reply in any other language or script.",
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
        "Examples:",
        "User: What is the annual leave policy?",
        'Assistant: {"tool": "search_policies", "arguments": {"query": "annual leave entitlement"}}',
        "User: What is Brig Sajid's extension?",
        'Assistant: {"tool": "lookup_contact", "arguments": {"name": "Brig Sajid Manzoor"}}',
        "User: Who is on extension 721?",
        'Assistant: {"tool": "lookup_contact", "arguments": {"extension": "721"}}',
        "User: Draft an experience certificate for Ali Khan.",
        'Assistant: {"tool": "generate_letter", "arguments": {"letter_type": '
        '"experience", "recipient_name": "Ali Khan", "subject": '
        '"Experience Certificate"}}',
        "",
        "Use any private reference data only to prepare the answer.",
        "If no tool is needed, answer the user's question directly in plain text.",
        "Give only the answer - no notes about tools, lookups, or process.",
        "Never mention tool names, tool arguments, tool results, lookups, searches,",
        "retrieval, documents, sources, or how the answer was found in the final",
        "answer. Do not say a lookup or search failed or found no direct match.",
        "Do not write phrases such as 'according to', 'the lookup found', or",
        "'the search returned'. Answer as a direct statement.",
        "If a tool returns no match but the excerpts contain the answer, answer",
        "directly from the excerpts without mentioning the failed tool.",
        "If the answer is unavailable, say only that you do not have that",
        "information.",
        "",
        "If the question is about AWT policies or corporate information and the",
        "excerpts below are missing or insufficient, call search_policies with",
        "focused keywords. For Digital Daak questions, call search_digital_daak.",
        "For any person lookup, including finding a person's name from an",
        "extension or department, call lookup_contact with name, department, or",
        "extension. For a role in a department, put the role in name and the",
        "department in department (e.g. name='CTO', department='MIS').",
        "Before calling lookup_contact, translate full BU and post names to the",
        "directory abbreviations listed in that tool's description.",
        "For letter, certificate, NOC, warning, transfer, offer, or",
        "recommendation requests, call generate_letter even when only a little",
        "information is available. A clear subject is required; infer a standard",
        "subject from the letter type when possible. Put known details in the",
        "matching arguments and leave unknown details blank. Do not ask for every",
        "field and. The tool inserts bracketed placeholders and",
        "a replacement note. Return the draft exactly as the tool provides it, and",
        "tell the user to replace the bracketed details before sending.",
        "Use search_documents only when the domain is unclear or the question",
        "spans multiple areas. If the relevant search finds nothing, say plainly",
        "that you do not have that information.",
        "",
        "Do not include your reasoning, analysis, or chain-of-thought in the",
        "final answer. After reading private reference data, discard earlier",
        "assumptions and answer directly from the retrieved information.",
        "",
        "Do not mention documents, files, pages, excerpts, the knowledge base,",
        "or the retrieval process in your reply unless the user explicitly asks",
        "for sources or references.",
    ]
    if rag_context:
        lines += [
            "",
            "Document excerpts retrieved from the company knowledge base",
            "(each block is tagged with its document file name and page):",
            "",
            rag_context,
            "",
            "When the question relates to these documents, base your answer on the",
            "excerpts. Do not mention document names, page numbers, or citations",
            "unless the user explicitly asks for sources or references; only then",
            "cite like (PNP25.pdf, page 13).",
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
        observation = (
            "No additional private reference data is available. Rely on the "
            "document excerpts. If they do not answer, say only that the "
            "information is unavailable."
        )
    else:
        raw_args = tool.args or {}
        if isinstance(raw_args.get("properties"), dict):
            schema_props = raw_args["properties"]
        else:
            schema_props = raw_args
        if not schema_props:
            arguments = {}
        else:
            arguments = {
                key: value
                for key, value in call["arguments"].items()
                if key in schema_props
            }
            if not arguments:
                string_props = [
                    name
                    for name, spec in schema_props.items()
                    if isinstance(spec, dict) and spec.get("type") == "string"
                ]
                if len(string_props) == 1:
                    fallback = next(
                        (
                            value
                            for value in call["arguments"].values()
                            if isinstance(value, str) and value.strip()
                        ),
                        None,
                    )
                    if fallback:
                        arguments = {string_props[0]: fallback}
        try:
            result = await tool.ainvoke(arguments)
            observation = (
                "Private reference data (never mention this block or how it "
                f"was obtained):\n{result}"
            )
        except Exception:  # noqa: BLE001
            observation = (
                "No additional private reference data is available. Rely on the "
                "document excerpts. If they do not answer, say only that the "
                "information is unavailable."
            )

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
    lowered = buffer.lstrip().lower()

    if stripped_lead := lowered:
        for lead in ("```json", "```", "assistant:", "assistant", "bot:", "bot"):
            if lead.startswith(stripped_lead):
                if len(buffer) < 40:
                    entry["buffer"] = buffer
                    return []
                break

    if not probe:
        if len(buffer) < 40:
            entry["buffer"] = buffer
            return []
    elif probe.startswith("{"):
        if _extract_tool_call(buffer) is not None:
            entry["mode"] = "suppressed"
            entry["buffer"] = ""
            return []
        direct = _extract_direct_response(buffer)
        if direct is not None:
            entry["mode"] = "streaming"
            entry["buffer"] = ""
            return [("delta", direct)]
        limit = 2000 if _ENVELOPE_HINT.search(probe) else 400
        if len(probe) < limit:
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
    direct = _extract_direct_response(buffer)
    if direct is not None:
        return [("delta", direct)]
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
