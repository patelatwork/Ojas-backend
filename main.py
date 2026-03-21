import os
import shutil
import traceback
import asyncio
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

state = {}


async def initialize_pipeline():
    """Run in background so the app starts up immediately and doesn't time out."""
    try:
        print("\n[1/4] Loading HuggingFace embeddings...")
        embeddings = get_embeddings()
        print("  [1/4] Done.")

        print("\n[2/4] Loading FAISS vector store...")
        store = load_or_build_index(embeddings)
        retriever = store.as_retriever(search_kwargs={"k": 5})
        print("  [2/4] Done.")

        print("\n[3/4] Initializing Groq LLM...")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set!")
        llm = ChatGroq(
            api_key=groq_key,
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024,
        )
        print("  [3/4] Done.")

        print("\n[4/4] Compiling LangGraph pipeline...")
        graph = build_graph(retriever, llm)
        print("  [4/4] Done.")

        state["store"] = store
        state["embeddings"] = embeddings
        state["graph"] = graph
        state["llm"] = llm
        state["ready"] = True

        print("\n✓ OJAS.AI pipeline is ready!\n")

    except Exception as e:
        print(f"\n✗ Pipeline initialization FAILED: {e}")
        traceback.print_exc()
        state["ready"] = False
        state["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("OJAS.AI — Starting (pipeline loading in background)")
    print("=" * 50)
    # Start pipeline in background — app becomes available immediately
    asyncio.create_task(initialize_pipeline())
    yield
    print("OJAS.AI — Shutting down")


app = FastAPI(
    title="Ojas.ai Self-RAG API",
    description="Ayurvedic knowledge assistant powered by Self-RAG and Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    issup: str | None = None
    isuse: str | None = None
    evidence: list[str] = []


@app.get("/health")
async def health():
    if state.get("error"):
        return {
            "status": "error",
            "pipeline_ready": False,
            "error": state["error"]
        }
    return {
        "status": "ok" if state.get("ready") else "loading",
        "pipeline_ready": state.get("ready", False)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if state.get("error"):
        raise HTTPException(status_code=500, detail=f"Pipeline failed to start: {state['error']}")

    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Pipeline is still loading, please wait 1-2 minutes and try again.")

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

    result = state["graph"].invoke(initial, config={"recursion_limit": 80})

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

    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

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


@app.get("/documents")
async def list_documents():
    pdfs = sorted([f.name for f in DOCS_DIR.glob("*.pdf")])
    return {"documents": pdfs, "count": len(pdfs)}