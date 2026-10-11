import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ai_agent_template.agent.graph import build_agent_graph, initial_state
from ai_agent_template.core.settings import Settings
from ai_agent_template.evaluation.config import EvaluationConfig, Experiment
from ai_agent_template.evaluation.datasets import (
    EvaluationSample,
    Passage,
    adapt_qasper,
    adapt_ragbench,
    load_samples,
)
from ai_agent_template.evaluation.experiments import aggregate, run_experiments
from ai_agent_template.evaluation.scoring import retrieval_scores
from ai_agent_template.rag.service import RagService
from tests.test_evidence_pipeline import Policy
from tests.test_rag_service import FakeVectorStore


def sample():
    return EvaluationSample(
        sample_id="q1",
        dataset="ragbench",
        subset="testset",
        split="test",
        data_version="frozen",
        question="What is measured?",
        passages=[Passage(unit_id="p1", document_key="doc", text="The MAE was 25.05 s.")],
        reference_answers=["REFERENCE MUST NOT BECOME THE GENERATED ANSWER"],
        gold_sets=[["p1"]],
        answerable=True,
    )


def source(evidence_id, unit, lo=0, hi=10, length=10):
    return {
        "evidence": {
            "evidence_id": evidence_id,
            "metadata": {
                "chunk_start": lo,
                "chunk_end": hi,
                "dataset_spans": [{"unit_id": unit, "start": 0, "end": length}],
            },
        }
    }


def test_multi_evidence_recall_and_duplicates():
    case = sample().model_copy(update={"gold_sets": [["p1", "p2"]]})
    result = retrieval_scores(case, [source("a", "p1"), source("a", "p1"), source("b", "p2")])
    assert result["recall_at_5"] == 1 and result["mrr_at_10"] == 1
    assert retrieval_scores(case, [source("a", "p1")])["recall_at_5"] == 0.5


def test_split_paragraph_coverage_requires_both_halves():
    case = sample()
    assert retrieval_scores(case, [source("a", "p1", 0, 5)])["recall_at_5"] == 0
    assert (
        retrieval_scores(case, [source("a", "p1", 0, 5), source("b", "p1", 5, 10)])["recall_at_5"]
        == 1
    )
    assert retrieval_scores(case, [])["mrr_at_10"] == 0
    assert (
        retrieval_scores(case.model_copy(update={"gold_sets": []}), [])["status"] == "not_evaluable"
    )


def test_reference_alternatives_are_not_union_gold():
    case = sample().model_copy(update={"gold_sets": [["p1"], ["p2", "p3"]]})
    assert retrieval_scores(case, [source("a", "p1")])["recall_at_5"] == 1


def test_ragbench_native_annotations_are_distinct_from_reference():
    row = {
        "id": "a",
        "question": "Question",
        "documents": ["Document sentence."],
        "documents_sentences": [[["0a", "Document sentence."]]],
        "response": "Pre-generated reference",
        "all_relevant_sentence_keys": ["0a"],
    }
    adapted = adapt_ragbench(row, "covidqa", "test", "sha")
    assert adapted.reference_answers == ["Pre-generated reference"] and adapted.answerable is None
    assert adapted.gold_sets[0][0] == adapted.passages[0].unit_id
    with pytest.raises(ValueError):
        adapt_ragbench({"question": "Incomplete"}, "covidqa", "test", "sha")


def test_qasper_multiple_annotations_and_unanswerable():
    row = {
        "id": "paper",
        "full_text": [{"section_name": "Results", "paragraphs": ["Evidence A.", "Evidence B."]}],
        "qas": [
            {
                "question_id": "q",
                "question": "Question",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "free_form_answer": "Answer",
                            "evidence": ["Evidence A.", "Evidence B."],
                        }
                    }
                ],
            },
            {
                "question_id": "u",
                "question": "Unknown",
                "answers": [{"answer": {"unanswerable": True}}],
            },
        ],
    }
    cases = adapt_qasper(row, "test", "sha")
    assert len(cases[0].gold_sets[0]) == 2 and cases[1].answerable is False
    row["qas"][0]["answers"][0]["answer"]["evidence"] = ["Figure 2"]
    assert adapt_qasper(row, "test", "sha")[0].annotation_status == "ambiguous_or_nontext_evidence"


def test_split_and_matrix_validation(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(sample().model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_samples("ragbench", "testset", "train", 1, str(path))
    with pytest.raises(ValueError, match="judge"):
        EvaluationConfig(dataset="ragbench", experiments=[Experiment(name="x")])
    with pytest.raises(ValueError, match="one variable"):
        EvaluationConfig(
            dataset="ragbench",
            phase="retrieval",
            experiments=[
                Experiment(name="base"),
                Experiment(name="changed", retrieval="dense", chunking="section"),
            ],
        )


class Writer:
    async def ainvoke(self, messages):
        return AIMessage(content="The MAE was 25.05 s. [E1]")


class FakeExecutor:
    executed = 0

    def __init__(self, settings, config, experiment, samples, directory):
        self.experiment = experiment
        self.rag = RagService(Settings(rag_data_path=str(directory / "index")), FakeVectorStore())
        self.rag.ingest_sections(
            sections=[
                Document(page_content=p.text, metadata={"dataset_unit_id": p.unit_id})
                for p in samples[0].passages
            ],
            source="paper.pdf",
        )

    async def run_sample(self, case):
        FakeExecutor.executed += 1
        graph = build_agent_graph(Writer(), self.rag, query_policy=Policy())
        state = await graph.ainvoke(initial_state(case.question))
        return {
            "answer": state["final_answer"],
            "claims": [claim.model_dump() for claim in state["claims"]],
            "citations": [citation.model_dump() for citation in state["citations"]],
            "retrieved": [source.model_dump() for source in self.rag.search(case.question)],
            "retrieval_calls": [{"mode": self.experiment.retrieval}],
            "stop_reason": state["stop_reason"],
            "decision_tokens": 0,
            "latency_ms": 10.0,
        }

    def close(self):
        pass


class FakeScorer:
    def __init__(self, *args):
        pass

    async def score(self, *args):
        return {
            "citation_precision": 1.0,
            "faithfulness": 1.0,
            "refused": False,
            "citation_status": "offline_fake_judge",
            "faithfulness_status": "offline_fake",
        }


async def test_offline_actual_graph_matrix_and_resume(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(sample().model_dump_json(), encoding="utf-8")
    config = EvaluationConfig(
        dataset="ragbench",
        subset="testset",
        input_path=str(data),
        max_samples=1,
        requests_per_second=1000,
        judge_model="fake-judge",
        experiments=[Experiment(name="dense", retrieval="dense"), Experiment(name="hybrid")],
    )
    output = tmp_path / "run"
    FakeExecutor.executed = 0
    settings = Settings(openai_api_key="fake")
    await run_experiments(
        config, settings, output, executor_factory=FakeExecutor, scorer_factory=FakeScorer
    )
    assert FakeExecutor.executed == 2
    rows = [
        json.loads(line)
        for line in (output / "dense/predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["answer"] != rows[0]["reference_answers"][0] and rows[0]["citations"]
    assert (output / "comparison.csv").is_file() and (output / "manifest.json").is_file()
    original_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    data.unlink()  # Resume must use the saved native data snapshot, not mutable inputs.
    await run_experiments(
        config, settings, resume=output, executor_factory=FakeExecutor, scorer_factory=FakeScorer
    )
    assert FakeExecutor.executed == 2
    assert len((output / "dense/predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))["created_at"]
        == original_manifest["created_at"]
    )


def test_unavailable_scores_are_not_counted_as_correct():
    result = aggregate(
        [
            {
                "status": "success",
                "sample_id": "a",
                "answer": "refusal",
                "answerable": False,
                "scores": {"faithfulness": None, "citation_precision": None, "refused": True},
                "latency_ms": 1.0,
                "retrieval_calls": [],
            }
        ]
    )
    assert result["faithfulness"] is None and result["citation_precision_evaluated"] == 0
    assert result["unanswerable_refusal_rate"] == 1
