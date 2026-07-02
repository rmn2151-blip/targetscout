FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

# RDKit needs a couple of system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "targetscout.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
