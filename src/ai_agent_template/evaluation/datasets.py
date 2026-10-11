"""Public native-split adapters; annotations are evidence units, never document-level gold."""

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_agent_template.rag.repository import digest, fingerprint


class Passage(BaseModel):
    unit_id: str
    document_key: str
    text: str
    section: str | None = None


class EvaluationSample(BaseModel):
    sample_id: str
    dataset: str
    subset: str
    split: Literal["train", "validation", "test"]
    data_version: str
    question: str
    passages: list[Passage]
    reference_answers: list[str] = Field(default_factory=list)
    gold_sets: list[list[str]] = Field(default_factory=list)
    answerable: bool | None = None
    annotation_status: str = "available"
    raw_annotations: dict[str, Any] = Field(default_factory=dict)


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def adapt_ragbench(
    row: dict[str, Any], subset: str, split: str, data_version: str
) -> EvaluationSample:
    if not all(key in row for key in ("id", "question", "documents", "response")):
        raise ValueError("RAGBench requires id, question, documents and response")
    passages = []
    native_to_unit = {}
    sentences = row.get("documents_sentences", [])
    for doc_index, text in enumerate(row["documents"]):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("RAGBench documents must be nonempty strings")
        doc_key = "ragbench:" + digest(text)
        if doc_index < len(sentences) and sentences[doc_index]:
            for key, sentence in sentences[doc_index]:
                unit_id = doc_key + ":" + str(key)[len(str(doc_index)) :]
                passages.append(Passage(unit_id=unit_id, document_key=doc_key, text=sentence))
                native_to_unit[str(key)] = unit_id
        else:
            passages.append(Passage(unit_id=doc_key + ":document", document_key=doc_key, text=text))
    relevant = row.get("all_relevant_sentence_keys")
    gold = [native_to_unit[key] for key in relevant or [] if key in native_to_unit]
    status = (
        "available"
        if relevant is not None and all(key in native_to_unit for key in relevant)
        else "sentence_annotations_missing_or_unmapped"
    )
    return EvaluationSample(
        sample_id=str(row["id"]),
        dataset="ragbench",
        subset=subset,
        split=split,
        data_version=data_version,
        question=row["question"],
        passages=passages,
        reference_answers=[row["response"]],
        gold_sets=[gold] if gold else [],
        # RAGBench adherence is not an answerability annotation.
        answerable=None,
        annotation_status=status,
        raw_annotations={
            key: row[key]
            for key in (
                "sentence_support_information",
                "all_relevant_sentence_keys",
                "response_sentences",
            )
            if key in row
        },
    )


def adapt_qasper(row: dict[str, Any], split: str, data_version: str) -> list[EvaluationSample]:
    if not all(key in row for key in ("id", "full_text", "qas")):
        raise ValueError("QASPER requires id, full_text and qas")
    full = row["full_text"]
    if isinstance(full, dict):
        full = [
            {"section_name": name, "paragraphs": paragraphs}
            for name, paragraphs in zip(full["section_name"], full["paragraphs"], strict=True)
        ]
    doc_key = "qasper:" + str(row["id"])
    passages = [
        Passage(
            unit_id=f"{doc_key}:{section_index}:{paragraph_index}",
            document_key=doc_key,
            text=text,
            section=section.get("section_name"),
        )
        for section_index, section in enumerate(full)
        for paragraph_index, text in enumerate(section["paragraphs"])
        if text.strip()
    ]
    qas = row["qas"]
    if isinstance(qas, dict):
        qas = [dict(zip(qas, values, strict=True)) for values in zip(*qas.values(), strict=True)]
    results = []
    for qa in qas:
        if not all(key in qa for key in ("question_id", "question", "answers")):
            raise ValueError("QASPER question fields are missing")
        annotations = qa["answers"]
        if isinstance(annotations, dict):
            annotations = [
                dict(zip(annotations, values, strict=True))
                for values in zip(*annotations.values(), strict=True)
            ]
        answers, gold_sets, unavailable, unanswerable_flags = [], [], False, []
        for annotation in annotations:
            answer = annotation.get("answer", annotation)
            unanswerable_flags.append(bool(answer.get("unanswerable", False)))
            if answer.get("unanswerable"):
                continue
            ref = answer.get("free_form_answer") or " ".join(answer.get("extractive_spans", []))
            if not ref and answer.get("yes_no") is not None:
                ref = "yes" if answer["yes_no"] else "no"
            if ref:
                answers.append(ref)
            gold = []
            for text in answer.get("evidence", []):
                matches = [
                    passage.unit_id
                    for passage in passages
                    if normalized(passage.text) == normalized(text)
                ]
                if len(matches) != 1:
                    unavailable = (
                        True  # Figure evidence and ambiguous repeated text need explicit mapping.
                    )
                else:
                    gold.extend(matches)
            if gold:
                gold_sets.append(list(dict.fromkeys(gold)))
        results.append(
            EvaluationSample(
                sample_id=str(qa["question_id"]),
                dataset="qasper",
                subset="default",
                split=split,
                data_version=data_version,
                question=qa["question"],
                passages=passages,
                reference_answers=answers,
                gold_sets=gold_sets,
                answerable=not all(unanswerable_flags) if unanswerable_flags else None,
                annotation_status="ambiguous_or_nontext_evidence" if unavailable else "available",
                raw_annotations={"answers": annotations},
            )
        )
    return results


def load_samples(
    dataset: str,
    subset: str,
    split: str,
    maximum: int,
    input_path: str | None = None,
    revision: str = "main",
    snapshot: dict[str, Any] | None = None,
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    if split not in {"train", "validation", "test"} or maximum < 1:
        raise ValueError("Use a native train/validation/test split and a positive sample limit")
    if snapshot is not None:
        rows = snapshot["raw_rows"]
        provenance = {key: value for key, value in snapshot.items() if key != "raw_rows"}
    elif input_path:
        path = Path(input_path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        provenance = {
            "source": str(path.resolve()),
            "content_sha256": digest(path.read_bytes()),
            "revision": revision,
        }
        sidecar = path.with_suffix(".provenance.json")
        if sidecar.exists():
            origin = json.loads(sidecar.read_text(encoding="utf-8"))
            if any(
                origin.get(key) != expected
                for key, expected in (("dataset", dataset), ("subset", subset), ("split", split))
            ):
                raise ValueError(
                    "Input provenance dataset/subset/split differs from requested data"
                )
            provenance["origin"] = origin
    else:
        from datasets import load_dataset

        repo = {"ragbench": "galileo-ai/ragbench", "qasper": "allenai/qasper"}.get(dataset)
        if not repo:
            raise ValueError("Supported public datasets: ragbench, qasper")
        from huggingface_hub import HfApi

        pinned = HfApi().dataset_info(repo, revision=revision).sha
        stream = load_dataset(
            repo,
            subset if dataset == "ragbench" else None,
            split=split,
            revision=pinned,
            streaming=True,
        )
        rows = []
        for row in stream:
            rows.append(row)
            if len(rows) >= maximum:
                break
        provenance = {"source": repo, "revision": pinned, "content_sha256": fingerprint(rows)}
    version = provenance["content_sha256"]
    samples = []
    for row in rows:
        if "dataset" in row and "passages" in row:
            sample = EvaluationSample.model_validate(row)
            if sample.dataset != dataset or sample.split != split or sample.subset != subset:
                raise ValueError("Normalized sample dataset/subset/split does not match the run")
            samples.append(sample)
        elif dataset == "ragbench":
            samples.append(adapt_ragbench(row, subset, split, version))
        elif dataset == "qasper":
            samples.extend(adapt_qasper(row, split, version))
        else:
            raise ValueError("Unsupported dataset")
    return samples[:maximum], {**provenance, "raw_rows": rows}
