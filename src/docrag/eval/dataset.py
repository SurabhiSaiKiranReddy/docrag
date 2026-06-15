"""Gold question/answer dataset for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

DEFAULT_DATASET = Path("data/eval/qa.json")


class QAItem(BaseModel):
    """One gold evaluation example."""

    question: str
    answer: str  # ground-truth answer (used by generation metrics)
    sources: list[str]  # filenames that genuinely contain the answer


def load_dataset(path: Path | str = DEFAULT_DATASET) -> list[QAItem]:
    """Load the gold QA dataset from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAItem.model_validate(item) for item in raw]
