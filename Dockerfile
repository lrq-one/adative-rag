FROM python:3.12-slim AS base

RUN pip install --no-cache-dir uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# OCR engine for product labels and documents.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project --extra postgres --extra vision
COPY src ./src
COPY examples ./examples

RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra postgres --extra vision
ENV UV_NO_SYNC=1

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "src/ai_agent_template/main.py", "--host", "0.0.0.0", "--port", "8000"]
