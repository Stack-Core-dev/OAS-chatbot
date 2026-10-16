import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import run_agent_stream
from app.llm import LLMError, chat as llm_chat

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    created_at: datetime


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest) -> ChatResponse:
    messages = [message.model_dump() for message in payload.history]
    messages.append({"role": "user", "content": payload.message})

    try:
        reply = await llm_chat(messages)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(reply=reply, created_at=datetime.now(timezone.utc))


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    messages = [message.model_dump() for message in payload.history]
    messages.append({"role": "user", "content": payload.message})

    async def event_stream():
        try:
            async for kind, text in run_agent_stream(messages):
                yield sse({"reasoning": text} if kind == "reasoning" else {"delta": text})
            yield sse({"done": True})
        except LLMError as exc:
            yield sse({"error": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
