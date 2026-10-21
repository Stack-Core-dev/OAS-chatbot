"""Standalone retrieval test over the local ChromaDB collection.

Embeds the query with the local Qwen3-Embedding-0.6B model (using the
model's query instruction prompt) and prints the top matches. This script
is intentionally NOT connected to the FastAPI app.

Run from the backend directory:
    .venv\\Scripts\\python.exe scripts\\search.py "your question"
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent

DEFAULT_MODEL = BACKEND_DIR / "Qwen3-Embedding-0.6B"
DEFAULT_CHROMA_DIR = BACKEND_DIR / "chroma_db"
DEFAULT_COLLECTION = "oas_documents"

FALLBACK_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


def query_prompt(model_dir: Path) -> str:
    config_path = model_dir / "config_sentence_transformers.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prompt = (config.get("prompts") or {}).get("query")
        if prompt:
            return prompt.strip()
    return FALLBACK_QUERY_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the local ChromaDB collection.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    if not args.chroma_dir.exists():
        print("No ChromaDB data found - run scripts/ingest_pdf.py first.", file=sys.stderr)
        return 1

    embeddings = HuggingFaceEmbeddings(
        model_name=str(args.model),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(args.chroma_dir),
    )

    text = f"{query_prompt(args.model)} {args.query}"
    results = vector_store.similarity_search_with_score(text, k=args.k)

    if not results:
        print("No results found.")
        return 1

    for rank, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source", "?")
        page = document.metadata.get("page", "?")
        preview = document.page_content[:300].replace("\n", " ")
        print(f"[{rank}] {source}, page {page}  (distance {score:.4f})")
        print(f"    {preview}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
