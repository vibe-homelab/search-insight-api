FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY config.yaml .
RUN mkdir -p data/lancedb

RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8600/healthz || exit 1

EXPOSE 8600
CMD ["python", "-m", "uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8600"]
