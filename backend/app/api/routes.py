import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.admin import storage as admin_storage
from app.agent import run_agent_stream
from app.config import get_settings
from app.llm import LLMError, chat as llm_chat

logger = logging.getLogger(__name__)

settings = get_settings()
router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class ChatResponse(BaseModel):
    reply: str
    created_at: datetime
    session_id: str


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
    reasoning: str,
    latency_ms: int,
    status: str,
    error: str,
) -> None:
    admin_storage.record_turn(
        session_id=session_id,
        question=question,
        answer=answer,
        reasoning=reasoning,
        latency_ms=latency_ms,
        # model=settings.llama_model,
        model=settings.geniex_model,
        status=status,
        error=error,
    )


async def _record(
    session_id: str,
    question: str,
    answer: str,
    reasoning: str,
    latency_ms: int,
    status: str,
    error: str,
) -> None:
    try:
        await asyncio.to_thread(
            _save_turn,
            session_id,
            question,
            answer,
            reasoning,
            latency_ms,
            status,
            error,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record conversation turn")


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or uuid.uuid4().hex
    messages = [message.model_dump() for message in payload.history]
    messages.append({"role": "user", "content": payload.message})
    started = time.perf_counter()

    try:
        reply = await llm_chat(messages)
    except LLMError as exc:
        await _record(
            session_id,
            payload.message,
            "",
            "",
            int((time.perf_counter() - started) * 1000),
            "error",
            str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        await _record(
            session_id,
            payload.message,
            "",
            "",
            int((time.perf_counter() - started) * 1000),
            "error",
            str(exc),
        )
        raise

    await _record(
        session_id,
        payload.message,
        reply,
        "",
        int((time.perf_counter() - started) * 1000),
        "ok",
        "",
    )
    return ChatResponse(
        reply=reply,
        created_at=datetime.now(timezone.utc),
        session_id=session_id,
    )


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    session_id = payload.session_id or uuid.uuid4().hex
    messages = [message.model_dump() for message in payload.history]
    messages.append({"role": "user", "content": payload.message})

    async def event_stream():
        started = time.perf_counter()
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        state = {"status": "ok", "error": ""}

        yield sse({"session_id": session_id})

        try:
            async for kind, text in run_agent_stream(messages):
                if kind == "reasoning":
                    reasoning_parts.append(text)
                    yield sse({"reasoning": text})
                else:
                    answer_parts.append(text)
                    yield sse({"delta": text})
            yield sse({"done": True})
        except LLMError as exc:
            state["status"] = "error"
            state["error"] = str(exc)
            yield sse({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            state["status"] = "error"
            state["error"] = str(exc)
            logger.exception("Streaming chat failed")
            yield sse({"error": str(exc)})
        finally:
            await _record(
                session_id,
                payload.message,
                "".join(answer_parts),
                "".join(reasoning_parts),
                int((time.perf_counter() - started) * 1000),
                state["status"],
                state["error"],
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
