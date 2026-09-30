FROM python:3.11-slim

WORKDIR /app

# Install essential build and curl packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ app/
COPY streamlit/ streamlit/
COPY data/ data/
COPY .env.example .env.example

# Create storage directories
RUN mkdir -p data/documents data/vectorstore

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
