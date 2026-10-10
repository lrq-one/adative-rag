"""Scientific document parsing with section/page metadata and optional OCR/VLM."""

import base64
import re
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from langchain_core.documents import Document
from pypdf import PdfReader

from ai_agent_template.core.settings import Settings

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".pdf", ".png", ".jpg", ".jpeg"
}


def _markdown_sections(text: str) -> list[Document]:
    heading = "Document"
    blocks: list[Document] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+\S", line):
            if buffer and "\n".join(buffer).strip():
                blocks.append(Document(page_content="\n".join(buffer), metadata={"section": heading}))
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
        model=settings.rag_vlm_model, api_key=settings.openai_api_key, temperature=0
    )
    encoded = base64.b64encode(image).decode("ascii")
    prompt = HumanMessage(content=[
        {"type": "text", "text": (
            "Describe only visible scientific figure content, labels, units and trends. "
            "Do not invent numeric measurements or conclusions. "
            "Mention any unreadable elements."
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ])
    result = await model.ainvoke([prompt])
    return str(result.content)


async def load_upload_documents(file: UploadFile, settings: Settings) -> list[Document]:
    """Return independent source sections; empty documents are not indexed."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    content = await file.read()
    if suffix in {"", ".txt", ".md", ".markdown", ".json"}:
        text = content.decode("utf-8")
        return _markdown_sections(text) if suffix in {".md", ".markdown"} else [
            Document(page_content=text, metadata={"section": "Document"})
        ] if text.strip() else []

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        docs: list[Document] = []
        pdf = None
        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
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
                    text = _ocr_image(rendered)
                if settings.rag_vlm_enabled:
                    description = await _describe_image(rendered, "image/png", settings)
                    text = (text + "\n\n[Page visual interpretation]\n" + description).strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"page": idx + 1, "content_type": "pdf_page"},
                ))
        if pdf is not None:
            pdf.close()
        return docs

    if suffix in {".png", ".jpg", ".jpeg"}:
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        parts = []
        if settings.rag_ocr_enabled:
            parts.append(_ocr_image(content))
        if settings.rag_vlm_enabled:
            parts.append(await _describe_image(content, mime, settings))
        if not any(part.strip() for part in parts):
            raise ValueError("Image input requires OCR or VLM to be enabled")
        return [Document(
            page_content="\n\n".join(parts),
            metadata={"content_type": "image", "page": 1},
        )]

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


async def load_upload_text(file: UploadFile) -> str:
    """Compatibility wrapper for existing text-only callers."""
    docs = await load_upload_documents(file, Settings())
    return "\n\n".join(doc.page_content for doc in docs)
