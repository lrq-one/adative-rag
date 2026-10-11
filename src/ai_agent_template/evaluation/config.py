"""Explicit matrix configs; unsupported modes fail instead of silently sharing a retriever."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    route: Literal["static", "rule", "llm"] = "static"
    retrieval: Literal["dense", "sparse", "hybrid"] = "hybrid"
    feedback: bool = False
    verification: Literal["off", "deterministic", "semantic"] = "semantic"
    chunking: Literal["recursive", "section", "parent_child"] = "parent_child"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = Field(default=384, ge=1)
    reranker: bool = False


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["ragbench", "qasper"]
    subset: str = "default"
    split: Literal["train", "validation", "test"] = "test"
    revision: str = "main"
    input_path: str | None = None
    storage_root: str | None = None
    phase: Literal["retrieval", "generation", "both"] = "both"
    context_mode: Literal["retrieved", "provided"] = "retrieved"
    max_samples: int = Field(default=3, ge=1)
    top_k: int = Field(default=10, ge=10, le=30)
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    seed: int = 42
    concurrency: int = Field(default=1, ge=1, le=8)
    requests_per_second: float = Field(default=0.5, gt=0)
    retries: int = Field(default=1, ge=0, le=3)
    judge_model: str | None = None
    faithfulness: bool = True
    experiments: list[Experiment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if len({experiment.name for experiment in self.experiments}) != len(self.experiments):
            raise ValueError("Experiment names must be unique")
        if self.phase != "retrieval" and not self.judge_model:
            raise ValueError("RAGAS Faithfulness requires an explicitly configured judge_model")
        if len(self.experiments) > 1:
            baseline = self.experiments[0].model_dump(exclude={"name"})
            for experiment in self.experiments[1:]:
                values = experiment.model_dump(exclude={"name"})
                changed = sum(
                    values[key] != baseline[key]
                    for key in baseline
                    if key not in {"embedding_model", "embedding_dimensions"}
                )
                changed += (values["embedding_model"], values["embedding_dimensions"]) != (
                    baseline["embedding_model"],
                    baseline["embedding_dimensions"],
                )
                if changed != 1:
                    raise ValueError(
                        "Each ablation must change exactly one variable relative to the baseline"
                    )
        return self


def load_config(path: str) -> EvaluationConfig:
    import yaml

    return EvaluationConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
