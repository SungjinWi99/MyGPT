from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    source: str
    source_id: str
    text: str
    title: str = ""
    url: str = ""
    license: str = ""
    language: str = "ko"
    year: int | None = None
    corpus: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectedRecord:
    source: str
    reason: str
