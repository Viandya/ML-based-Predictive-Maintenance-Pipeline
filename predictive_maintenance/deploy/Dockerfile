FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY models/ ./models/

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "scripts/inference_demo.py"]
