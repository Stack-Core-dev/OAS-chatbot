"""Standalone PDF ingestion for the OAS knowledge base.

Pipeline: PDF -> text extraction (pypdf, with OCR fallback via PyMuPDF +
RapidOCR for scanned pages) -> LangChain chunks (RecursiveCharacterTextSplitter)
-> embeddings (local Qwen3-Embedding-0.6B) -> ChromaDB.

This script is intentionally NOT connected to the FastAPI app.

Run from the backend directory:
    .venv\\Scripts\\python.exe scripts\\ingest_pdf.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import psutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_PDF = BACKEND_DIR / "pdfs" / "DigitalDAAKManual.pdf"
DEFAULT_MODEL = BACKEND_DIR / "Qwen3-Embedding-0.6B"
DEFAULT_CHROMA_DIR = BACKEND_DIR / "chroma_db"
DEFAULT_COLLECTION = "oas_documents"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into ChromaDB using the local Qwen3 embedding model."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep existing chunks instead of rebuilding the collection",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="password for encrypted PDFs (empty user passwords work without it)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="do not OCR pages that have no extractable text",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="resolution used to render scanned pages for OCR (default: 200)",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=20,
        help="pages with fewer extractable characters are sent to OCR (default: 20)",
    )
    return parser.parse_args()


def _ocr_page(doc, page_number: int, engine, dpi: int) -> str:
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=dpi)
    result, _ = engine(pix.tobytes("png"))
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()


def extract_pages(
    pdf_path: Path,
    use_ocr: bool = True,
    ocr_dpi: int = 200,
    ocr_min_chars: int = 20,
    password: str | None = None,
) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path), password=password)
    fitz_doc = None
    ocr_engine = None
    ocr_pages = 0
    pages: list[tuple[int, str]] = []
    try:
        with tqdm(total=len(reader.pages), unit="page", desc="Reading", ncols=80) as bar:
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if use_ocr and len(text) < ocr_min_chars:
                    if fitz_doc is None:
                        import pymupdf

                        fitz_doc = pymupdf.open(str(pdf_path))
                        if fitz_doc.needs_pass and not fitz_doc.authenticate(password or ""):
                            raise SystemExit(
                                f"Could not decrypt {pdf_path.name} for OCR "
                                "- pass the correct --password."
                            )
                    if ocr_engine is None:
                        print("Loading OCR engine (RapidOCR, bundled models) ...")
                        from rapidocr_onnxruntime import RapidOCR

                        ocr_engine = RapidOCR()
                    ocr_text = _ocr_page(fitz_doc, number, ocr_engine, ocr_dpi)
                    if ocr_text:
                        text = ocr_text
                        ocr_pages += 1
                pages.append((number, text))
                bar.update(1)
    finally:
        if fitz_doc is not None:
            fitz_doc.close()
    if ocr_pages:
        print(f"  OCR applied to {ocr_pages} scanned page(s)")
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        for index, piece in enumerate(splitter.split_text(text), start=1):
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                        "chunk": index,
                    },
                )
            )
    return documents


def load_embeddings(model_dir: Path) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=str(model_dir),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_system_status(process: psutil.Process) -> str:
    memory = psutil.virtual_memory()
    process_mb = process.memory_info().rss / (1024 * 1024)
    return (
        f"CPU {psutil.cpu_percent(interval=None):4.1f}% | "
        f"RAM {memory.percent:4.1f}% "
        f"({memory.used / 1024**3:.1f}/{memory.total / 1024**3:.1f} GB) | "
        f"proc {process_mb:.0f} MB"
    )


def main() -> int:
    args = parse_args()

    for path, label in ((args.pdf, "PDF"), (args.model, "Model directory")):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1

    print(f"Extracting text from {args.pdf.name} ...")
    pages = extract_pages(
        args.pdf,
        use_ocr=not args.no_ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_min_chars=args.ocr_min_chars,
        password=args.password,
    )
    with_text = sum(1 for _, text in pages if text.strip())
    print(f"  {len(pages)} pages ({with_text} with text)")

    documents = chunk_pages(pages, args.pdf.name, args.chunk_size, args.chunk_overlap)
    print(f"  {len(documents)} chunks (size={args.chunk_size}, overlap={args.chunk_overlap})")
    if not documents:
        print("No text could be extracted - nothing to index.", file=sys.stderr)
        return 1

    print(f"Loading embedding model from {args.model} ...")
    embeddings = load_embeddings(args.model)

    if not args.append and args.chroma_dir.exists():
        print("Removing existing ChromaDB data (use --append to keep it) ...")
        shutil.rmtree(args.chroma_dir)

    print(f"Embedding chunks and storing into ChromaDB at {args.chroma_dir} ...")
    vector_store = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(args.chroma_dir),
    )

    total = len(documents)
    width = len(str(total))
    process = psutil.Process()
    psutil.cpu_percent(interval=None)
    started = time.perf_counter()
    print(f"Embedding {total} chunks (system status after each chunk) ...")
    for index, document in enumerate(documents, start=1):
        chunk_started = time.perf_counter()
        vector_store.add_documents([document], batch_size=1)
        now = time.perf_counter()
        elapsed = now - started
        average = elapsed / index
        remaining = average * (total - index)
        metadata = document.metadata
        print(
            f"[{index:>{width}}/{total}] {index / total * 100:5.1f}% | "
            f"{metadata.get('source', '?')} p.{metadata.get('page', '?')} "
            f"chunk {metadata.get('chunk', '?')} | "
            f"this {now - chunk_started:.2f}s | avg {average:.2f}s/chunk | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(remaining)} | "
            f"{_format_system_status(process)}",
            flush=True,
        )

    total = vector_store._collection.count()
    print(f"Done. Collection '{args.collection}' now holds {total} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
