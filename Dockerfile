# One image for both the RAG API and the Phoenix server (Linux container, so the
# Windows Application Control policy that blocks some native DLLs does not apply).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch first (the default wheel pulls several GB of CUDA libraries).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .

EXPOSE 8000 6006
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
