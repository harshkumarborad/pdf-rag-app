FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf and chromadb
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create persistent data directories
RUN mkdir -p /app/data/chroma_db /app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
