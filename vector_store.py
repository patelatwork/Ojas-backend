from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = Path("docs")
INDEX_DIR = Path("faiss_index")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(embeddings):
    """Load all PDFs from docs/ and create a fresh FAISS index."""
    docs = []
    for pdf in DOCS_DIR.glob("**/*.pdf"):
        print(f"  Loading: {pdf.name}")
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"  Error loading {pdf.name}: {e}")

    if not docs:
        raise ValueError("No PDFs found in docs/ folder. Add at least one PDF before starting.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    print(f"  Created {len(chunks)} chunks from {len(docs)} pages")

    store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"  Index saved to {INDEX_DIR}/")
    return store


def load_or_build_index(embeddings):
    """Load saved index if it exists, otherwise build from docs/."""
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        print(f"  Loading existing index from {INDEX_DIR}/")
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    print("  No saved index found — building from docs/")
    return build_index(embeddings)


def add_documents_to_index(pdf_path: str, store: FAISS, embeddings):
    """Add a new PDF to the existing FAISS index and persist to disk."""
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)
    store.add_documents(chunks)
    store.save_local(str(INDEX_DIR))
    return len(chunks)
