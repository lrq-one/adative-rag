from fastapi.testclient import TestClient

from ai_agent_template.api.app import create_app
from ai_agent_template.api.dependencies import get_rag_service
from ai_agent_template.api.routes import rag as routes
from ai_agent_template.core.settings import Settings
from tests.test_incremental_index import make_service
from tests.test_pdf_provenance import pdf_bytes


def test_uploaded_original_evidence_history_delete_and_jobs(tmp_path, monkeypatch):
    service, _, qdrant = make_service(tmp_path)
    settings = Settings(rag_data_path=str(tmp_path), rate_limit_enabled=False)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)
    content = pdf_bytes(["ModelA MAE was 25.05 s.", "A separate second page."])
    uploaded = client.post(
        "/v1/rag/files", files={"file": ("paper.pdf", content, "application/pdf")}
    )
    assert uploaded.status_code == 200
    data = uploaded.json()
    doc_id = data["document_id"]
    manifest = service.repository.active(doc_id)
    evidence_id = manifest.evidence_ids[1]
    evidence = client.get("/v1/rag/evidence/" + evidence_id).json()
    assert evidence["status"] == "active" and evidence["evidence"]["page_number"] == 2
    original = client.get(f"/v1/rag/document-versions/{doc_id}/1/file")
    assert original.status_code == 200 and original.content == content
    updated = client.post(
        "/v1/rag/files",
        data={"document_id": doc_id},
        files={
            "file": (
                "paper.pdf",
                pdf_bytes(["Corrected first page.", "A separate second page."]),
                "application/pdf",
            )
        },
    )
    assert updated.status_code == 200 and updated.json()["stats"]["vector_reuses"] == 1
    assert client.get("/v1/rag/evidence/" + evidence_id).json()["status"] == "historical"
    assert len(client.get(f"/v1/rag/document-versions/{doc_id}").json()) == 2
    assert len(client.get("/v1/rag/index-jobs").json()) == 2
    assert client.delete("/v1/rag/documents/" + doc_id).status_code == 200
    assert client.get("/v1/rag/evidence/" + evidence_id).status_code == 404
    assert client.get(f"/v1/rag/document-versions/{doc_id}/1/file").status_code == 404
    assert service.search("page") == []
    qdrant.close()
