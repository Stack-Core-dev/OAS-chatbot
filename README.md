# OAS-Assistant

A local RAG chatbot for Army Welfare Trust (AWT) documents. It answers questions from a
PDF knowledge base (Digital DAAK manual, P&P-25, communiqués, ...) with page citations,
and can call tools (e.g. current date/time) when needed. Everything runs on your machine:
no cloud APIs, no data leaves the box.

```
┌─────────────────────────── BROWSER ───────────────────────────┐
│  React widget (Vite :5173)                                    │
│  launcher button → chat box → streams SSE                     │
└────────────────────────────┬──────────────────────────────────┘
                             │ /api proxy → 127.0.0.1:8000
┌────────────────────────────▼──────────────────────────────────┐
│  FastAPI (uvicorn :8000)                                      │
│  /api/chat/stream → agent.py (LangGraph)                      │
│    ├─ rag.py  : embed question → ChromaDB → top-4 excerpts    │
│    ├─ tools.py: get_current_datetime                          │
│    └─ llm.py  : resolves model from llama.cpp                 │
└───────────┬──────────────────────────────┬────────────────────┘
            │ OpenAI-compatible API        │ vector search
┌───────────▼──────────────┐   ┌───────────▼───────────────────┐
│ llama.cpp server :8080   │   │ ChromaDB (backend/chroma_db)   │
│ DeepSeek-R1-8B Q4 (GGUF) │   │ collection "oas_documents"     │
│ emits reasoning_content  │   │ chunks tagged source + page    │
└──────────────────────────┘   └───────────────────────────────┘
```

## How it works

1. The browser sends the question (plus history) to `POST /api/chat/stream`.
2. The backend embeds the question with **Qwen3-Embedding-0.6B** and retrieves the
   **top-4** most similar chunks from ChromaDB.
3. Excerpts are injected into the system prompt, tagged like `[PNP25.pdf, page 13]`,
   with instructions to answer from them and cite `(file, page)`.
4. The **LangGraph agent** calls the local LLM (DeepSeek-R1 via llama.cpp). The model's
   `reasoning_content` is streamed separately from the answer.
5. **Tool protocol**: the 8B model has no native tool support, so it requests tools by
   printing `{"tool": "get_current_datetime", "arguments": {}}`. The agent detects that
   JSON, hides it from the stream, executes the tool, feeds the result back, and the
   model writes the final answer (max 3 tool loops).
6. The stream ends with a `done` event. The UI shows a live "Thinking" box for the
   reasoning phase, then the answer with markdown rendering.

SSE events: `{"reasoning": "..."}`, `{"delta": "..."}`, `{"done": true}`, `{"error": "..."}`.

## Repository layout

```
backend/
  app/                 FastAPI application
    main.py            app entry (uvicorn app.main:app)
    config.py          settings (.env aware)
    api/routes.py      /api/health, /api/chat, /api/chat/stream (SSE)
    agent.py           LangGraph agent, tool JSON protocol, stream filtering
    rag.py             ChromaDB retrieval + context builder
    tools.py           tool registry (get_current_datetime)
    llm.py             llama.cpp client + model alias resolution
  scripts/             standalone ingestion tooling (not imported by the app)
    ingest_pdf.py      PDF → chunks → embeddings → ChromaDB (OCR fallback)
    search.py          retrieval test CLI
  pdfs/                source PDFs (gitignored)
  chroma_db/           vector store (gitignored, generated)
  Qwen3-Embedding-0.6B/ local embedding model (gitignored)
frontend/
  src/App.jsx          chat widget (launcher, streaming, thinking box)
  src/App.css          theme (frosted glass, dark)
  src/api.js           SSE client
  vite.config.js       dev server :5173, proxies /api → :8000
```

## Prerequisites

- Windows (PowerShell) with Python 3.11 and Node.js 18+.
- **llama.cpp server** running an OpenAI-compatible endpoint on port 8080 with the
  DeepSeek-R1 model, e.g.:
  ```powershell
  .\llama-cpp\llama-server.exe -m D:\llama-cpp-cpu\DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --port 8080
  ```
- **Embedding model**: download `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers
  format) into `backend/Qwen3-Embedding-0.6B`:
  ```powershell
  huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir backend\Qwen3-Embedding-0.6B
  ```

## Setup

### 1. Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(`requirements.txt` includes everything; `scripts/requirements.txt` is the standalone
subset used by the ingestion scripts.)

Optional `.env` in `backend/` to override defaults:

| Variable              | Default                                  | Purpose                          |
| --------------------- | ---------------------------------------- | -------------------------------- |
| `HOST` / `PORT`       | `127.0.0.1` / `8000`                     | API bind address                 |
| `CORS_ORIGINS`        | `http://localhost:5173,...`              | allowed browser origins          |
| `LLAMA_BASE_URL`      | `http://127.0.0.1:8080`                  | llama.cpp server                 |
| `LLAMA_MODEL`         | `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M`       | preferred model id/alias         |
| `LLAMA_TIMEOUT`       | `300`                                    | LLM request timeout (seconds)    |
| `CHROMA_DIR`          | `backend/chroma_db`                      | vector store location            |
| `EMBEDDING_MODEL_DIR` | `backend/Qwen3-Embedding-0.6B`           | embedding model location         |
| `RAG_COLLECTION`      | `oas_documents`                          | ChromaDB collection name         |
| `RAG_TOP_K`           | `4`                                      | excerpts injected per question   |

### 2. Knowledge base

Put PDFs in `backend/pdfs/`, then ingest each one:

```powershell
cd backend
.venv\Scripts\python.exe scripts\ingest_pdf.py --pdf pdfs\PNP25.pdf --append
```

- `--append` keeps existing chunks (use when **adding** a new PDF).
- Omit `--append` to **rebuild** the whole collection from scratch (use when a PDF is
  updated/replaced so old chunks are dropped).
- Text is extracted with pypdf; pages without a text layer (scans) are rendered with
  PyMuPDF and OCR'd with RapidOCR automatically.
  - `--no-ocr` disable OCR fallback
  - `--ocr-dpi 300` higher quality for small print (slower)
  - `--ocr-min-chars 20` pages below this character count go to OCR
  - `--password <pw>` for password-protected PDFs (empty-password files need nothing)
- Chunking: 1000 chars, 150 overlap. Metadata per chunk: `source` (file name),
  `page`, `chunk`.

Verify retrieval:

```powershell
.venv\Scripts\python.exe scripts\search.py "what is PNP25?"
.venv\Scripts\python.exe scripts\search.py --k 6 "leave policy"
```

First run loads the embedding model (~30 s); ingestion embedding rate on CPU is slow
(progress bar advances every 10%).

### 3. Frontend

```powershell
cd frontend
npm install
```

## Running

```powershell
# terminal 1 - LLM
D:\llama-cpp-cpu\llama-server.exe -m D:\llama-cpp-cpu\DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --port 8080

# terminal 2 - backend  (restart after every code change; no --reload by default)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app

# terminal 3 - frontend
cd frontend
npm run dev          # http://localhost:5173
```

Open http://localhost:5173, click the launcher button, ask about the documents.

Other commands:

```powershell
npm run build        # production build
npm run lint         # oxlint
```

## API

| Endpoint            | Method | Description                       |
| ------------------- | ------ | --------------------------------- |
| `/api/health`       | GET    | `{"status": "ok"}`                |
| `/api/chat`         | POST   | plain non-streaming LLM reply     |
| `/api/chat/stream`  | POST   | SSE stream used by the frontend   |

Request body: `{"message": "...", "history": [{"role": "user", "content": "..."}]}`.

## Troubleshooting

- **`Configured model '...' not found; falling back ...`** — informational; the server
  picked the only available model. Set `LLAMA_MODEL` to the reported id to silence it.
- **Chat shows a raw `{"tool": ...}` JSON** — the backend is running old code; restart
  uvicorn. Tool JSON is suppressed by `agent.py` on the current code.
- **`llama.cpp server unreachable` / 502** — start the llama-server, check port 8080.
- **Answer cites the wrong file** — rebuild the collection if PDFs were replaced
  (ingest without `--append`).
- **Port 8000 in use / stale behavior on Windows** — old uvicorn processes can linger:
  `Get-NetTCPConnection -LocalPort 8000 -State Listen` then `Stop-Process -Id <pid>`.
- **Encrypted PDF** — install includes `cryptography`; pass `--password` if a user
  password exists (owner-password-only files open automatically).

## Notes

- Gitignored: `*.pdf`, `backend/chroma_db/`, `backend/Qwen3-Embedding-0.6B/`, `.venv/`,
  `node_modules/`, `dist/`. Source documents and generated data stay local.
- The embedding model runs on CPU by default (`device: "cpu"` in `rag.py` /
  `ingest_pdf.py`). GPU/NPU acceleration is not supported by this stack on Windows ARM.
- Current knowledge base (example): DigitalDAAKManual.pdf, PNP25.pdf, MD.pdf →
  ~544 chunks in collection `oas_documents`.
