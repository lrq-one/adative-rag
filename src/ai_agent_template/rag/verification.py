"""Lightweight, deterministic citation and numeric-evidence validation.

This is a safety gate, not a semantic entailment oracle. It only checks
that cited source labels exist and numeric claims are traceable.
"""

import re
from dataclasses import dataclass

from ai_agent_template.rag.schemas import RagSource

_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")
_CITATION = re.compile(r"\[([^\[\]\n]+)\]")
_ORDERED_LIST_MARKER = re.compile(r"^[ \t]*(?:>[ \t]*)*\d{1,9}[.)][ \t]+(?=\S)", re.MULTILINE)


@dataclass(frozen=True)
class Verification:
    supported: bool
    reason: str = ""


def validate_citations(answer: str, sources: list[RagSource]) -> Verification:
    if not sources:
        return Verification(False, "No retrieved evidence")
    citations = _CITATION.findall(answer)
    available = {s.source for s in sources}
    if not citations:
        return Verification(False, "Answer contains no verifiable citation")
    unknown = set(citations) - available
    if unknown:
        return Verification(False, "Answer cites unknown source labels")
    # Ignore only Markdown list markers, not the measurements within list items.
    # This does not establish mathematical or semantic entailment.
    without_citations = _CITATION.sub("", answer)
    claimed = set(_NUMBER.findall(_ORDERED_LIST_MARKER.sub("", without_citations)))
    evidence = "\n".join(
        str(s.metadata.get("parent_excerpt") or s.content) for s in sources if s.source in citations
    )
    reference_numbers = set(_NUMBER.findall(_ORDERED_LIST_MARKER.sub("", evidence)))
    if claimed - reference_numbers:
        return Verification(False, "One or more numeric claims lack source evidence")
    return Verification(True)
