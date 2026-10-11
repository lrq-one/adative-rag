"""Evidence-unit coverage and independent generation scoring, with explicit exclusions."""

from statistics import mean
from typing import Any

from ai_agent_template.evaluation.datasets import EvaluationSample
from ai_agent_template.rag.evidence import ClaimAssessment, GroundedClaim, LLMEvidenceVerifier


def retrieval_scores(sample: EvaluationSample, sources: list[dict[str, Any]]) -> dict[str, Any]:
    if sample.annotation_status != "available" or not sample.gold_sets:
        return {
            "status": "not_evaluable",
            "reason": sample.annotation_status if sample.gold_sets else "no_text_gold_evidence",
        }
    unique = []
    seen = set()
    for source in sources:
        evidence = source.get("evidence") or {}
        key = evidence.get("evidence_id")
        if key and key not in seen:
            seen.add(key)
            unique.append(evidence)

    def covered(prefix: Any) -> set[str]:
        intervals: dict[str, list[tuple[int, int, int]]] = {}
        for evidence in prefix:
            meta = evidence["metadata"]
            for unit in meta.get("dataset_spans", []):
                lo, hi = (
                    max(meta["chunk_start"], unit["start"]),
                    min(meta["chunk_end"], unit["end"]),
                )
                if lo < hi:
                    intervals.setdefault(unit["unit_id"], []).append(
                        (lo - unit["start"], hi - unit["start"], unit["end"] - unit["start"])
                    )
        found = set()
        for unit, values in intervals.items():
            merged: list[tuple[int, int]] = []
            for lo, hi, length in sorted(values):
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
                else:
                    merged.append((lo, hi))
            if sum(hi - lo for lo, hi in merged) >= 0.95 * values[0][2]:
                found.add(unit)
        return found

    def best_recall(k: Any) -> float:
        found = covered(unique[:k])
        return max(len(found.intersection(gold)) / len(set(gold)) for gold in sample.gold_sets)

    all_gold = set(unit for gold in sample.gold_sets for unit in gold)
    rank = next(
        (
            k
            for k in range(1, min(10, len(unique)) + 1)
            if covered(unique[:k]).intersection(all_gold)
        ),
        None,
    )
    return {
        "status": "evaluated",
        "recall_at_5": best_recall(5),
        "recall_at_10": best_recall(10),
        "hit_rate_at_5": float(best_recall(5) > 0),
        "mrr_at_10": 1 / rank if rank else 0,
    }


class GenerationScorer:
    def __init__(self, settings: Any, judge_model: str, faithfulness: bool = True) -> None:
        from ai_agent_template.agent.models import build_chat_model

        self.judge = LLMEvidenceVerifier(
            build_chat_model(settings, model_name=judge_model, max_tokens=4096)
        )
        self.limiter = None
        self.faithfulness = None
        if faithfulness:
            from httpx import AsyncClient
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import Faithfulness

            if not settings.openai_api_key:
                raise ValueError("Independent evaluation requires OPENAI_API_KEY")
            client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.model_base_url,
                timeout=120,
                max_retries=2,
                http_client=AsyncClient(event_hooks={"request": [self.limit_request]}),
            )
            kwargs: dict[str, Any] = (
                {"extra_body": {"enable_thinking": False}}
                if judge_model.lower().startswith("qwen")
                else {}
            )
            self.faithfulness = Faithfulness(
                llm=llm_factory(judge_model, client=client, temperature=0, **kwargs)
            )

    def set_rate_limiter(self, limiter: Any) -> None:
        self.limiter = limiter
        self.judge.policy._model.rate_limiter = limiter

    async def limit_request(self, request: Any) -> None:
        if self.limiter:
            await self.limiter.aacquire()

    async def score(
        self,
        question: Any,
        answer: Any,
        claims: Any,
        records: Any,
        contexts: Any,
        refused: Any = False,
    ) -> dict[str, Any]:
        result = {
            "citation_precision": None,
            "citation_status": "not_evaluable",
            "faithfulness": None,
            "faithfulness_status": "not_evaluable",
            "refused": refused,
        }
        if refused or not claims:
            result["reason"] = "refusal_or_no_verifiable_claims"
            return result
        pairs = [
            GroundedClaim(text=claim.text, evidence_ids=[label])
            for claim in claims
            for label in claim.evidence_ids
        ]
        checks: list[ClaimAssessment] = []
        errors: list[str] = []
        tokens = 0
        for offset in range(0, len(pairs), 20):
            report = await self.judge.assess(pairs[offset : offset + 20], records)
            tokens += report.tokens
            if report.error:
                errors.append(report.error)
            else:
                checks.extend(
                    check.model_copy(update={"claim_index": check.claim_index + offset})
                    for check in report.checks
                )
        evaluated = [check for check in checks if check.verdict != "uncertain"]
        result.update(
            {
                "citation_pairs_total": len(pairs),
                "citation_pairs_evaluated": len(evaluated),
                "citation_pairs_supported": sum(
                    check.verdict == "supported" for check in evaluated
                ),
                "citation_pairs_not_evaluable": len(pairs) - len(evaluated),
                "citation_judge_tokens": tokens,
                "citation_checks": [check.model_dump() for check in checks],
            }
        )
        if evaluated:
            result.update(
                {
                    "citation_precision": mean(check.verdict == "supported" for check in evaluated),
                    "citation_status": "automatic_judge",
                }
            )
        if errors or len(evaluated) != len(pairs):
            result["citation_reason"] = ";".join(errors) or "uncertain_judgments"
        if self.faithfulness:
            try:
                scored = await self.faithfulness.ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts
                )
                import math

                value = float(scored.value)
                if not math.isfinite(value):
                    raise ValueError("Undefined faithfulness")
                result.update({"faithfulness": value, "faithfulness_status": "ragas"})
            except Exception as error:
                result["faithfulness_reason"] = type(error).__name__
        return result
