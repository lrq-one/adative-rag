"""Commerce authority gates augment the shared deterministic and semantic evidence checks."""

import re
from typing import Any

from ai_agent_template.commerce.retrieval import scope
from ai_agent_template.commerce.schemas import ProductVariant
from ai_agent_template.rag.evidence import ClaimAssessment, GroundedClaim, VerificationReport
from ai_agent_template.rag.schemas import EvidenceRecord


class CommerceVerifier:
    def __init__(self, repository: Any, delegate: Any) -> None:
        self.repository, self.delegate = repository, delegate

    async def assess(
        self, claims: list[GroundedClaim], records: dict[str, EvidenceRecord]
    ) -> VerificationReport:
        report: VerificationReport = await self.delegate.assess(claims, records)
        if report.error:
            return report
        current = scope.get()
        known = {
            item.sku for item in self.repository.list(ProductVariant, current.principal.store_id)
        }
        checks = []
        for index, claim in enumerate(claims):
            selected = [records[key] for key in claim.evidence_ids if key in records]
            declared = {record.metadata.get("sku") for record in selected}
            mentioned = {sku for sku in known if sku.casefold() in claim.text.casefold()}
            reason = ""
            if mentioned - declared:
                reason = "Claim SKU differs from its bound product records"
            if re.search(
                r"price|inventory|in stock|stock level|价格|库存|precio|existencias",
                claim.text,
                re.I,
            ) and not any(record.metadata.get("business_data") for record in selected):
                reason = "Price/inventory claims require current authoritative catalog data"
            if any(
                record.metadata.get("market") not in {None, "GLOBAL", current.market}
                for record in selected
            ):
                reason = "Evidence belongs to another market"
            if reason:
                checks.append(
                    ClaimAssessment(
                        claim_index=index,
                        claim=claim.text,
                        evidence_ids=claim.evidence_ids,
                        verdict="unsupported",
                        reason=reason,
                        failure_type="contradiction",
                    )
                )
            else:
                checks.append(report.checks[index])
        return report.model_copy(
            update={
                "checks": checks,
                "supported": all(check.verdict == "supported" for check in checks),
            }
        )
