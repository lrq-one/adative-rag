FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# OCR engine; English and Chinese language packs for research documents.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --no-dev --no-install-project && uv sync --no-dev
RUN uv pip install --python /app/.venv/bin/python "pymupdf>=1.25,<2" "pytesseract>=0.3.13,<1" "Pillow>=10,<13"
ENV UV_NO_SYNC=1

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "src/ai_agent_template/main.py", "--host", "0.0.0.0", "--port", "8000"]
