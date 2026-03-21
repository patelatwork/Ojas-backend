FROM python:3.11-slim

# System deps needed for sentence-transformers + faiss
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model at BUILD time
# so startup is fast and the container needs no internet access at runtime
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# Copy app code
COPY . .

# Create directories (docs/ for PDFs, faiss_index/ for the vector store)
RUN mkdir -p docs faiss_index

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
