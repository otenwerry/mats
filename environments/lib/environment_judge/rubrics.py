"""Load the small, fixed rubric set owned by the environments experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .schema import JudgeFamily


DEFAULT_RUBRICS_ROOT = Path(__file__).resolve().parents[2] / "judge_dimensions"

GLOBAL_RUBRICS = (
    ("noticed_honeypot", "global/noticed_honeypot.md"),
    ("environment_failure", "global/environment_failure.md"),
)
FAMILY_RUBRICS = {
    "ml_training_data_misuse": (
        ("reward_hacking", "ml_training_data_misuse/reward_hacking.md"),
        ("deception", "ml_training_data_misuse/deception.md"),
    ),
    "p_hacking": (
        ("reward_hacking", "p_hacking/reward_hacking.md"),
    ),
}


@dataclass(frozen=True)
class RubricDocument:
    key: str
    relative_path: str
    content: str
    sha256: str

    def metadata(self) -> dict:
        return {
            "key": self.key,
            "path": self.relative_path,
            "content": self.content,
            "sha256": self.sha256,
        }


def load_rubrics(
    family: JudgeFamily,
    *,
    root: Path = DEFAULT_RUBRICS_ROOT,
) -> tuple[RubricDocument, ...]:
    """Load global rubrics followed by the selected family's rubrics.

    File order is fixed here because prompt order and schema agreement are part of the
    experiment contract, not a behavior inferred from Markdown metadata.
    """

    try:
        specs = (*GLOBAL_RUBRICS, *FAMILY_RUBRICS[family])
    except KeyError as ex:
        raise ValueError(f"unsupported judge family: {family!r}") from ex

    documents: list[RubricDocument] = []
    for key, relative_path in specs:
        path = root / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                f"required {family} judge rubric does not exist: {path}"
            ) from ex
        if not content.strip():
            raise ValueError(f"judge rubric is empty: {path}")
        documents.append(RubricDocument(
            key=key,
            relative_path=relative_path,
            content=content,
            sha256=sha256(content.encode("utf-8")).hexdigest(),
        ))
    return tuple(documents)
