from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Get the current system date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


@tool
def search_documents(query: str) -> str:
    """Search the AWT document knowledge base (policies, manuals, procedures).

    Use this when the provided document excerpts do not answer the question.
    Pass focused keywords, not the whole question.
    """
    from app.rag import retrieve

    hits = retrieve(query, k=4)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{source}, page {page}] {text.strip()[:700]}"
        for text, source, page, _ in hits
    )


TOOLS = [get_current_datetime, search_documents]
