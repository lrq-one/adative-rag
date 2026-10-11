"""Scientific document parsing with section/page metadata and optional OCR/VLM."""

import asyncio
import base64
import os
import re
import uuid
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from langchain_core.documents import Document
from pypdf import PdfReader

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import PersistentCache
from ai_agent_template.rag.repository import digest, fingerprint
from ai_agent_template.rag.schemas import IndexStats

PARSER_VERSION = "scientific-parser-v3"
VISUAL_PROMPT_VERSION = "visible-scientific-content-v1"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".pdf", ".png", ".jpg", ".jpeg"}


@lru_cache(maxsize=1)
def ocr_identity() -> str:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "tesseract-unavailable"


def parser_identity(settings: Settings) -> str:
    from importlib.metadata import version

    return fingerprint(
        [
            PARSER_VERSION,
            version("pypdf"),
            settings.rag_ocr_enabled,
            ocr_identity() if settings.rag_ocr_enabled else None,
            settings.rag_vlm_enabled,
            settings.rag_vlm_model,
            VISUAL_PROMPT_VERSION,
            settings.model_base_url,
        ]
    )


def _page_text(page: Any, cache: Any, stats: Any) -> str:
    # Canonical resolved content/resources ignore indirect object numbers and PDF metadata.
    # Necessary lightweight PDF reads remain; unchanged pages skip extract_text.
    def canonical(value: Any, depth: Any = 0) -> Any:
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

        if depth > 12:
            raise ValueError("Deep/cyclic PDF resources cannot be safely fingerprinted")
        if isinstance(value, IndirectObject):
            return canonical(value.get_object(), depth + 1)
        if isinstance(value, StreamObject):
            return [
                digest(value.get_data()),
                {
                    str(key): canonical(item, depth + 1)
                    for key, item in value.items()
                    if str(key) not in {"/Length", "/Filter", "/DecodeParms"}
                },
            ]
        if isinstance(value, DictionaryObject):
            return {str(key): canonical(item, depth + 1) for key, item in value.items()}
        if isinstance(value, ArrayObject):
            return [canonical(item, depth + 1) for item in value]
        return str(value)

    if not cache:
        return (page.extract_text() or "").strip()
    try:
        key = fingerprint(
            [
                PARSER_VERSION,
                canonical(page.get("/Contents")),
                canonical(page.get("/Resources")),
                str(page.mediabox),
                page.rotation,
            ]
        )
    except Exception:
        return (page.extract_text() or "").strip()
    with cache.compute_lock("page-text-" + key):
        found = cache.get("page-text", key)
        if found is not None:
            if stats:
                stats.page_text_cache_hits += 1
            return str(found)
        text = (page.extract_text() or "").strip()
        cache.put_many("page-text", {key: text})
        return text


def _markdown_sections(text: str) -> list[Document]:
    heading = "Document"
    blocks: list[Document] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+\S", line):
            if buffer and "\n".join(buffer).strip():
                blocks.append(
                    Document(page_content="\n".join(buffer), metadata={"section": heading})
                )
            heading = line.lstrip("# ").strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer and "\n".join(buffer).strip():
        blocks.append(Document(page_content="\n".join(buffer), metadata={"section": heading}))
    return blocks


def _ocr_image(image: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise ValueError(
            "OCR requires the optional Pillow and pytesseract packages and the Tesseract binary."
        ) from error
    return str(pytesseract.image_to_string(Image.open(BytesIO(image)))).strip()


async def _describe_image(image: bytes, mime: str, settings: Settings) -> str:
    if not settings.openai_api_key:
        raise ValueError("VLM analysis requires OPENAI_API_KEY")
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.rag_vlm_model,
        api_key=settings.openai_api_key,
        temperature=0,
        base_url=settings.model_base_url,
    )
    encoded = base64.b64encode(image).decode("ascii")
    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe only visible scientific figure content, labels, units and trends. "
                    "Do not invent numeric measurements or conclusions. "
                    "Mention any unreadable elements."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    )
    result = await model.ainvoke([prompt])
    return str(result.content)


async def _load_upload_documents(
    file: UploadFile,
    settings: Settings,
    cache: PersistentCache | None = None,
    stats: IndexStats | None = None,
) -> list[Document]:
    """Return independent source sections; empty documents are not indexed."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    content = await file.read()
    if suffix in {"", ".txt", ".md", ".markdown", ".json"}:
        text = content.decode("utf-8")
        return (
            _markdown_sections(text)
            if suffix in {".md", ".markdown"}
            else [Document(page_content=text, metadata={"section": "Document"})]
            if text.strip()
            else []
        )

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        docs: list[Document] = []
        pdf = None
        for idx, page in enumerate(reader.pages):
            text = await asyncio.to_thread(_page_text, page, cache, stats)
            if (settings.rag_ocr_enabled and not text) or settings.rag_vlm_enabled:
                try:
                    import fitz
                except ImportError as error:
                    raise ValueError(
                        "PDF OCR/VLM requires the optional PyMuPDF package."
                    ) from error
                if pdf is None:
                    pdf = fitz.open(stream=content, filetype="pdf")
                rendered = pdf[idx].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                if settings.rag_ocr_enabled and not text:
                    text = await _visual_cached(
                        rendered, "image/png", settings, cache, stats, "ocr"
                    )
                if settings.rag_vlm_enabled:
                    description = await _visual_cached(
                        rendered, "image/png", settings, cache, stats, "vlm"
                    )
                    text = (text + "\n\n[Page visual interpretation]\n" + description).strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": idx + 1,
                            "content_type": "pdf_page",
                            "bbox": None,
                            "parser_status": "text_extracted_bbox_unavailable",
                        },
                    )
                )
        if pdf is not None:
            pdf.close()
        return docs

    if suffix in {".png", ".jpg", ".jpeg"}:
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        parts = []
        if settings.rag_ocr_enabled:
            parts.append(await _visual_cached(content, mime, settings, cache, stats, "ocr"))
        if settings.rag_vlm_enabled:
            parts.append(await _visual_cached(content, mime, settings, cache, stats, "vlm"))
        if not any(part.strip() for part in parts):
            raise ValueError("Image input requires OCR or VLM to be enabled")
        return [
            Document(
                page_content="\n\n".join(parts),
                metadata={"content_type": "image", "page": 1},
            )
        ]

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


async def _visual_cached(
    content: bytes,
    mime: str,
    settings: Settings,
    cache: PersistentCache | None,
    stats: IndexStats | None,
    kind: str,
) -> str:
    key = fingerprint(
        [
            digest(content),
            mime,
            kind,
            settings.rag_vlm_model if kind == "vlm" else ocr_identity(),
            VISUAL_PROMPT_VERSION,
            parser_identity(settings),
            settings.model_base_url,
        ]
    )
    lock = cache.compute_lock(kind + key) if cache else None
    if lock:
        await asyncio.to_thread(lock.acquire)
    try:
        cached = cache.get(kind, key) if cache else None
        if cached is not None:
            if stats:
                field = "ocr_cache_hits" if kind == "ocr" else "vlm_cache_hits"
                setattr(stats, field, getattr(stats, field) + 1)
            return str(cached)
        value = (
            await asyncio.to_thread(_ocr_image, content)
            if kind == "ocr"
            else await _describe_image(content, mime, settings)
        )
        if cache:
            cache.put_many(kind, {key: value})
        return value
    finally:
        if lock:
            lock.release()


async def load_upload_documents(
    file: UploadFile,
    settings: Settings,
    cache: PersistentCache | None = None,
    stats: IndexStats | None = None,
) -> list[Document]:
    content = await file.read()
    suffix = Path(file.filename or "upload").suffix.lower()
    key = fingerprint(
        [
            digest(content),
            suffix,
            parser_identity(settings),
            settings.rag_ocr_enabled,
            settings.rag_vlm_enabled,
            settings.rag_vlm_model,
            VISUAL_PROMPT_VERSION,
        ]
    )
    if cache:
        cached = cache.get("parsed", key)
        if cached is not None:
            if stats:
                stats.parse_cache_hits += 1
            return [
                Document(page_content=item["text"], metadata=item["metadata"]) for item in cached
            ]
    docs = await _load_upload_documents(
        UploadFile(file=BytesIO(content), filename=file.filename), settings, cache, stats
    )
    if cache:
        cache.put_many(
            "parsed", {key: [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]}
        )
    return docs


def save_original(content: bytes, suffix: str, data_path: str) -> str:
    directory = Path(data_path) / "files"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (digest(content) + suffix)
    if not target.exists():
        temporary = directory / (uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
    return str(target.resolve())


async def load_upload_text(file: UploadFile) -> str:
    """Compatibility wrapper for existing text-only callers."""
    docs = await load_upload_documents(file, Settings())
    return "\n\n".join(doc.page_content for doc in docs)
