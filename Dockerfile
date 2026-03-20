FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY config.yaml .
RUN mkdir -p data/lancedb
EXPOSE 8600
CMD ["python", "-m", "uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8600"]
