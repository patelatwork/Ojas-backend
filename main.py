import os
import shutil
import traceback
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("OJAS.AI — Starting up")
    print("=" * 50)

    try:
        print("\n[1/4] Loading HuggingFace embeddings...")
        embeddings = get_embeddings()
        print("  Done.")
    except Exception as e:
        print(f"  FAILED at step 1: {e}")
        traceback.print_exc()
        raise

    try:
        print("\n[2/4] Loading FAISS vector store...")
        store = load_or_build_index(embeddings)
        retriever = store.as_retriever(search_kwargs={"k": 5})
        print("  Done.")
    except Exception as e:
        print(f"  FAILED at step 2: {e}")
        traceback.print_exc()
        raise

    try:
        print("\n[3/4] Initializing Groq LLM...")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set!")
        llm = ChatGroq(
            api_key=groq_key,
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024,
        )
        print("  Done.")
    except Exception as e:
        print(f"  FAILED at step 3: {e}")
        traceback.print_exc()
        raise

    try:
        print("\n[4/4] Compiling LangGraph pipeline...")
        graph = build_graph(retriever, llm)
        print("  Done.")
    except Exception as e:
        print(f"  FAILED at step 4: {e}")
        traceback.print_exc()
        raise

    state["store"] = store
    state["embeddings"] = embeddings
    state["graph"] = graph
    state["llm"] = llm

    print("\n" + "=" * 50)
    print("OJAS.AI — Ready!")
    print("=" * 50 + "\n")

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
    return {"status": "ok", "pipeline_ready": "graph" in state}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

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


@app.get("/documents")
async def list_documents():
    pdfs = sorted([f.name for f in DOCS_DIR.glob("*.pdf")])
    return {"documents": pdfs, "count": len(pdfs)}