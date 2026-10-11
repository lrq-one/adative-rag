"""Citation and list formatting regressions without weakening numeric evidence checks."""

import pytest

from ai_agent_template.rag.schemas import RagSource
from ai_agent_template.rag.verification import validate_citations


@pytest.mark.parametrize("marker", ["1. ", "2) ", "    3. ", "> 4. ", "> > 5) "])
def test_ordered_list_markers_are_not_measurements(marker: str) -> None:
    sources = [RagSource(source="notes.md", content="核对单位、评价指标和数据划分。")]
    assert validate_citations(f"{marker}核对单位 [notes.md]", sources).supported


def test_list_items_still_require_evidence_for_measurements() -> None:
    sources = [RagSource(source="paper.pdf", content="MAE was 25.05 seconds.")]
    assert validate_citations("1. MAE was 25.05 seconds [paper.pdf]", sources).supported
    assert not validate_citations("1. MAE was 30.00 seconds [paper.pdf]", sources).supported
    assert not validate_citations("1. MAE was 2 seconds [paper.pdf]", sources).supported


def test_leading_measurements_are_not_removed_as_list_markers() -> None:
    sources = [RagSource(source="paper.pdf", content="Check the units.")]
    assert not validate_citations("25.05 seconds was the MAE [paper.pdf]", sources).supported
    assert not validate_citations("25 seconds was the MAE [paper.pdf]", sources).supported


def test_source_list_markers_cannot_support_a_numeric_claim() -> None:
    sources = [RagSource(source="paper.pdf", content="1. Check the units.\n2. Check the split.")]
    assert not validate_citations("The MAE was 2 seconds [paper.pdf]", sources).supported


@pytest.mark.parametrize("label", ["source", "invented.pdf", "notes.md section=Methods", "1"])
def test_placeholder_and_unknown_citations_remain_rejected(label: str) -> None:
    sources = [RagSource(source="notes.md", content="Check the split.")]
    assert not validate_citations(f"Check the split [{label}]", sources).supported
