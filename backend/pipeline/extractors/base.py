from dataclasses import dataclass, field


@dataclass
class ExtractedContent:
    source_type: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)
