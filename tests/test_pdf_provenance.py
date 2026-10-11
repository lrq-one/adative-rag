from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import PersistentCache
from ai_agent_template.rag.loaders import load_upload_documents
from ai_agent_template.rag.schemas import IndexStats
from ai_agent_template.rag.service import RagService
from tests.test_rag_service import FakeVectorStore


def pdf_bytes(texts, title="original"):
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in texts:
        page = writer.add_blank_page(width=500, height=700)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 650 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Rotate")] = NumberObject(0)
    writer.add_metadata({"/Title": title})
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def test_pdf_pages_and_metadata_only_page_cache(tmp_path):
    settings = Settings(rag_data_path=str(tmp_path))
    cache = PersistentCache(tmp_path / "cache.sqlite3")

    async def parse(content):
        stats = IndexStats()
        docs = await load_upload_documents(
            UploadFile(file=BytesIO(content), filename="paper.pdf"), settings, cache, stats
        )
        return docs, stats

    first, _ = await parse(pdf_bytes(["ModelA MAE was 25.05 s.", "Independent second page."]))
    assert [doc.metadata["page"] for doc in first] == [1, 2]
    assert all(doc.metadata["bbox"] is None for doc in first)
    again, stats = await parse(
        pdf_bytes(["ModelA MAE was 25.05 s.", "Independent second page."], "changed metadata")
    )
    assert stats.parse_cache_hits == 0 and stats.page_text_cache_hits == 2
    modified, stats = await parse(
        pdf_bytes(["ModelA MAE was 26.05 s.", "Independent second page."], "changed page")
    )
    assert stats.page_text_cache_hits == 1 and "26.05" in modified[0].page_content
    service = RagService(settings, FakeVectorStore())
    result = service.ingest_sections(sections=again, source="paper.pdf")
    manifest = service.repository.active(result.document_id)
    assert service.repository.resolve(manifest.evidence_ids[1])[0].page_number == 2


async def test_vlm_cache_persists_and_invalidates_model_and_endpoint(tmp_path, monkeypatch):
    from ai_agent_template.rag import loaders

    calls = []

    async def describe(image, mime, settings):
        calls.append(settings.rag_vlm_model)
        return "Visible label only."

    monkeypatch.setattr(loaders, "_describe_image", describe)
    cache = PersistentCache(tmp_path / "cache.sqlite3")
    settings = Settings(rag_vlm_enabled=True)

    async def parse(config):
        stats = IndexStats()
        await load_upload_documents(
            UploadFile(file=BytesIO(b"fake-image"), filename="figure.png"), config, cache, stats
        )
        return stats

    await parse(settings)
    assert (await parse(settings)).parse_cache_hits == 1 and len(calls) == 1
    await parse(settings.model_copy(update={"rag_vlm_model": "other-model"}))
    await parse(settings.model_copy(update={"model_base_url": "https://different.example/v1"}))
    assert len(calls) == 3
