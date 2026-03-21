import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from pydantic import BaseModel

from rag_pipeline import build_graph
from vector_store import add_documents_to_index, get_embeddings, load_or_build_index

load_dotenv()

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

# Global runtime state — loaded once at startup
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("OJAS.AI — Starting up")
    print("=" * 50)

    print("\n[1/4] Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = get_embeddings()
    print("  Done.")

    print("\n[2/4] Loading FAISS vector store...")
    store = load_or_build_index(embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 5})
    print("  Done.")

    print("\n[3/4] Initializing Groq LLM (llama-3.1-8b-instant)...")
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
    )
    print("  Done.")

    print("\n[4/4] Compiling LangGraph Self-RAG pipeline...")
    graph = build_graph(retriever, llm)
    print("  Done.")

    state["store"] = store
    state["embeddings"] = embeddings
    state["graph"] = graph

    print("\n" + "=" * 50)
    print("OJAS.AI — Ready to serve requests")
    print("=" * 50 + "\n")

    yield  # App runs here

    print("OJAS.AI — Shutting down")


app = FastAPI(
    title="Ojas.ai Self-RAG API",
    description="Ayurvedic knowledge assistant powered by Self-RAG and Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    issup: str | None = None
    isuse: str | None = None
    evidence: list[str] = []


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": "graph" in state}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet. Try again in a moment.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial = {
        "question": req.question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
    }

    result = graph.invoke(initial, config={"recursion_limit": 80})

    return ChatResponse(
        answer=result.get("answer", ""),
        issup=result.get("issup"),
        isuse=result.get("isuse"),
        evidence=result.get("evidence", []),
    )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if "store" not in state:
        raise HTTPException(status_code=503, detail="Vector store not ready yet.")

    save_path = DOCS_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_added = add_documents_to_index(
            str(save_path),
            state["store"],
            state["embeddings"],
        )
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")

    return {
        "message": f"Successfully uploaded and indexed '{file.filename}'",
        "chunks_added": chunks_added,
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    pdf_path = DOCS_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"'{filename}' not found.")

    pdf_path.unlink()

    # Rebuild the index without the deleted file
    print(f"Rebuilding index after deleting {filename}...")
    from vector_store import build_index
    state["store"] = build_index(state["embeddings"])
    state["graph"] = build_graph(
        state["store"].as_retriever(search_kwargs={"k": 5}),
        state.get("llm"),
    )

    return {"message": f"'{filename}' deleted and index rebuilt."}


@app.get("/documents")
async def list_documents():
    pdfs = sorted([f.name for f in DOCS_DIR.glob("*.pdf")])
    return {"documents": pdfs, "count": len(pdfs)}
