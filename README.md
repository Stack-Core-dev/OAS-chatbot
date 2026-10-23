# AWT-Assistant

A local RAG chatbot for Army Welfare Trust (AWT) documents. It answers questions from a
PDF knowledge base (Digital DAAK manual, P&P-25, communiques, ...) with page-grounded
answers, and can call tools (current date/time, document searches, contact lookup, and
letter drafting) when needed. Everything runs on your machine: no cloud APIs, no data
leaves the box.

## Recent changes

- The LLM backend is now **GenieX** at `http://127.0.0.1:18181`, using the model
  reported by `/v1/models` (currently `local/gemma-4-e2b-q4_0:Q4_0`). Legacy
  llama.cpp settings and client code are retained as comments in `config.py`,
  `llm.py`, `agent.py`, and `.env.example`.
- Start GenieX with **`--nctx 8192`**. The default 4096-token window can be
  exhausted by conversation history plus reasoning, leaving no room for the
  final answer.
- Retrieval now uses the **top 3** chunks (`RAG_TOP_K=3`). If a retrieved chunk
  belongs to an admin-created entry, every chunk sharing that `entry_id` is
  included, so the context can exceed three chunks.
- Admin text entries are chunked at **1,000 characters** with **150-character
  overlap**.
- Contact lookup uses the updated five-column directory: `Name`, `Designation`,
  `Department`, `Floor`, and `Contact`. Mobile numbers were removed.
- Chat answers are restricted to **English or Roman Urdu** and do not mention
  tools, lookups, retrieval, sources, or how the answer was found.
- The initial chat view shows only the bot avatar, without greeting text.

```
┌─────────────────────────── BROWSER ───────────────────────────┐
│  React widget (Vite :5173)                                    │
│  launcher button → chat box → streams SSE                     │
└────────────────────────────┬──────────────────────────────────┘
                             │ /api proxy → 127.0.0.1:8000
┌────────────────────────────▼──────────────────────────────────┐
│  FastAPI (uvicorn :8000)                                      │
│  /api/chat/stream → agent.py (LangGraph)                      │
│    ├─ rag.py  : embed question → ChromaDB → top-3 excerpts    │
│    ├─ tools.py: time, docs, contacts, letters                 │
│    └─ llm.py  : resolves model from GenieX                   │
└───────────┬──────────────────────────────┬────────────────────┘
            │ OpenAI-compatible API        │ vector search
┌───────────▼──────────────┐   ┌───────────▼───────────────────┐
│ GenieX server :18181     │   │ ChromaDB (backend/chroma_db)   │
│ gemma-4-e2b Q4           │   │ collection "oas_documents"     │
│ emits reasoning_content  │   │ chunks tagged source + page    │
└──────────────────────────┘   └───────────────────────────────┘
```

## How it works

1. The browser sends the question, history, and a session ID to
   `POST /api/chat/stream`.
2. The backend embeds the question with **Qwen3-Embedding-0.6B** and retrieves the
   **top-3** most similar chunks from ChromaDB. If a matched chunk belongs to an
   admin-created entry, all chunks for that entry are included.
3. Excerpts are injected into the system prompt, tagged like `[PNP25.pdf, page 13]`,
   with instructions to answer from them. Citations are shown only when the user
   explicitly asks for sources.
4. The **LangGraph agent** calls the local LLM (via GenieX). The model's
   `reasoning_content` is streamed separately from the answer.
5. **Tool protocol**: the local model has no native tool support, so it requests tools
   by printing `{"tool": "get_current_datetime", "arguments": {}}`. The agent detects
   that JSON, hides it from the stream, executes the tool, feeds the result back, and
   the model writes the final answer (max 4 tool loops). If the injected excerpts do
   not answer a document question, the model can use the document, contact, or letter
   tools with refined arguments — the hybrid RAG fallback.
6. The stream ends with a `done` event. The UI shows a live "Thinking" box for the
   reasoning phase, then the answer with markdown rendering.

SSE events: `{"session_id": "..."}`, `{"reasoning": "..."}`, `{"delta": "..."}`,
`{"done": true}`, `{"error": "..."}`.

## Repository layout

```
backend/
  app/                 FastAPI application
    main.py            app entry (uvicorn app.main:app)
    config.py          settings (.env aware)
    api/routes.py      /api/health, /api/chat, /api/chat/stream (SSE)
    admin/             management API (auth, SQLite, sessions, overview)
    agent.py           LangGraph agent, tool JSON protocol, stream filtering
    rag.py             ChromaDB retrieval, top-3 + admin-entry expansion
    knowledge.py       admin text chunking + ChromaDB writes
    contacts.py        contact directory parsing, SQLite/FTS lookup
    letter_agent.py    letter/certificate drafting
    tools.py           tool registry (time, document/contact search, letters)
    llm.py             GenieX client + model alias resolution
  scripts/             standalone ingestion tooling (not imported by the app)
    ingest_pdf.py      PDF → chunks → embeddings → ChromaDB (OCR fallback)
    search.py          retrieval test CLI
  sql/contacts.sql     contact directory data (gitignored)
  pdfs/                source PDFs (gitignored)
  chroma_db/           vector store (gitignored, generated)
  Qwen3-Embedding-0.6B/ local embedding model (gitignored)
frontend/
  src/App.jsx          chat widget (launcher, streaming, thinking box)
  src/App.css          theme (frosted glass, dark)
  src/admin/           management console (overview, sessions, runtime)
  src/api.js           SSE client
  vite.config.js       dev server :5173, proxies /api → :8000
```

## Prerequisites

- Windows (PowerShell) with Python 3.11 and Node.js 18+.
- **GenieX server** running an OpenAI-compatible endpoint on port 18181 with the
  configured model, e.g.:
  ```powershell
  geniex serve --host 127.0.0.1:18181 --nctx 8192
  ```
  Use the model id reported by `http://127.0.0.1:18181/v1/models` in
  `GENIEX_MODEL`. Legacy llama.cpp settings and client lines remain commented in
  `config.py`, `llm.py`, and `agent.py` for reference.
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
| `GENIEX_BASE_URL`     | `http://127.0.0.1:18181`                 | GenieX server                    |
| `GENIEX_MODEL`        | `local/gemma-4-e2b-q4_0:Q4_0`            | preferred model id/alias         |
| `GENIEX_TIMEOUT`      | `300`                                    | LLM request timeout (seconds)    |
| `CHROMA_DIR`          | `backend/chroma_db`                      | vector store location            |
| `EMBEDDING_MODEL_DIR` | `backend/Qwen3-Embedding-0.6B`           | embedding model location         |
| `RAG_COLLECTION`      | `oas_documents`                          | ChromaDB collection name         |
| `RAG_TOP_K`           | `3`                                      | excerpts injected per question   |
| `ADMIN_PASSWORD`      | *(unset)*                                | management console password      |
| `ADMIN_SESSION_HOURS` | `12`                                     | admin session lifetime           |
| `ADMIN_SESSION_SECRET` | derived from password                   | optional cookie-signing secret   |
| `ADMIN_DB_PATH`       | `backend/admin.db`                       | session/audit SQLite file        |

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
- Chunking: 1000 chars, 150 overlap; PDF chunks do not cross page boundaries.
  Metadata per chunk: `source` (file name), `page`, `chunk`.

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

### 4. Management dashboard

Create or update `backend/.env` with an admin password:

```dotenv
ADMIN_PASSWORD=replace-with-a-long-random-password
# ADMIN_SESSION_SECRET=replace-with-an-independent-random-value
# ADMIN_SESSION_HOURS=12
```

`ADMIN_PASSWORD` is required. There is no default password, and `/api/admin/*`
returns `503` until it is configured. Restart uvicorn after changing `.env`.

The dashboard lives at `http://localhost:5173/admin`. It stores complete chat
sessions and their user/assistant turns in `backend/admin.db` (`chat_sessions`
and `chat_messages`), shows the active runtime prompt/tools, and lets you add,
edit, or delete manual text entries in the knowledge base. Legacy
single-answer rows are imported into session IDs prefixed with `legacy-` on
startup. Text is chunked and embedded with the same local Qwen3 model used for
PDFs, so changes are immediately available to RAG retrieval. PDF upload,
reindexing, and prompt editing are not implemented yet.

For LAN access, use `http://<host-ip>:5173/admin`. The Vite proxy keeps `/api`
same-origin; only add that origin to `CORS_ORIGINS` if the frontend targets the
API directly.

### 5. Contact directory

`backend/sql/contacts.sql` supplies the `lookup_contact` tool. It uses five
columns: `Name`, `Designation`, `Department`, `Floor`, and `Contact`. The file is
loaded into an in-memory SQLite FTS5 index on first lookup; lookups support name,
department, and contact/extension matching with fuzzy fallback. The SQL file is
gitignored and stays local.

## Running

```powershell
# terminal 1 - LLM
geniex serve --host 127.0.0.1:18181 --nctx 8192

# terminal 2 - backend  (restart after every code change; no --reload by default)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app

# terminal 3 - frontend
cd frontend
npm run dev          # http://localhost:5173
```

Open http://localhost:5173, click the launcher button, ask about the documents.
The management console is at http://localhost:5173/admin.

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

Request body:
`{"message": "...", "history": [{"role": "user", "content": "..."}], "session_id": "optional-uuid"}`.
The response (`/api/chat`) includes `session_id`; the SSE stream sends it as its
first event. If omitted, the backend generates one.

### Management API

| Endpoint                      | Method | Description                              |
| ----------------------------- | ------ | ---------------------------------------- |
| `/api/admin/session`          | GET    | auth state and configuration status      |
| `/api/admin/login`            | POST   | exchange `{"password": "..."}` for cookie |
| `/api/admin/logout`           | POST   | clear the admin cookie                   |
| `/api/admin/overview`         | GET    | service health, stats, recent sessions   |
| `/api/admin/sessions`         | GET    | paginated session log                    |
| `/api/admin/sessions/{id}`    | GET    | full session transcript                  |
| `/api/admin/documents`        | GET    | ChromaDB source/chunk inventory          |
| `/api/admin/documents/text`   | POST   | chunk, embed, and add a text entry       |
| `/api/admin/documents/text/{id}` | PUT | replace a text entry                     |
| `/api/admin/documents/text/{id}` | DELETE | remove a text entry                   |
| `/api/admin/runtime`          | GET    | active model, prompt, tools, RAG config  |

All endpoints except `session` and `login` require the signed HTTP-only
`awt_admin_session` cookie.

Each chat session has its own `session_id`. The backend stores a session record
plus one user message and one assistant message per turn, including reasoning,
latency, status, and errors. The admin UI lists sessions and opens the complete
transcript for review.

Text entry body: `{"heading": "Updated leave policy", "content": "..."}`. The
heading is stored as `Admin: <heading>`, and long content is split into
1000-character chunks with 150-character overlap before embedding. When any chunk
from an entry matches a query, retrieval includes every chunk for that entry. Updating
or deleting an entry replaces/removes all of its chunks by entry ID.

## Troubleshooting

- **`Configured model '...' not found; falling back ...`** — informational; the server
  picked the only available model. Set `GENIEX_MODEL` to the reported id to silence it.
- **Chat shows a raw `{"tool": ...}` JSON** — the backend is running old code; restart
  uvicorn. Tool JSON is suppressed by `agent.py` on the current code.
- **`GenieX server unreachable` / 502** — start `geniex serve`, check port 18181.
- **GenieX context-length error or reasoning with no final answer** — restart it
  with `--nctx 8192` (or `16384`). The default is 4096; long history plus
  reasoning can consume the whole window and leave the answer empty.
- **Answer cites the wrong file** — rebuild the collection if PDFs were replaced
  (ingest without `--append`).
- **Port 8000 in use / stale behavior on Windows** — old uvicorn processes can linger:
  `Get-NetTCPConnection -LocalPort 8000 -State Listen` then `Stop-Process -Id <pid>`.
- **Encrypted PDF** — install includes `cryptography`; pass `--password` if a user
  password exists (owner-password-only files open automatically).

## Notes

- Gitignored: `*.pdf`, `backend/chroma_db/`, `backend/Qwen3-Embedding-0.6B/`,
  `backend/admin.db`, `backend/.env`, `backend/sql/contacts.sql`, `.venv/`,
  `node_modules/`, `dist/`. Source documents, credentials, contact data, and
  generated data stay local.
- The embedding model runs on CPU by default (`device: "cpu"` in `rag.py` /
  `ingest_pdf.py`). GenieX uses its configured CPU/GPU/NPU backend for generation.
- The admin knowledge-base page reports the current source and chunk counts for
  collection `oas_documents`.
