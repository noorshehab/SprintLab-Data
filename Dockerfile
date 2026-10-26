FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/deploy.txt ./requirements/deploy.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements/deploy.txt \
    && python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
