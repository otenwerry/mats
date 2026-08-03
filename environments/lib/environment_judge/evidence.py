"""Loss-visible rendering of the evidence supplied to the environment judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Sequence

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ContentReasoning
from inspect_scout import MessagesPreprocessor, message_numbering
from pydantic import Field, model_validator

from .schema import ArtifactReference, MessageNumber, NonEmptyText, StrictModel


class ArtifactSnapshot(StrictModel):
    """One complete artifact state at a named boundary.

    This layer never caps content.  ``truncated`` and ``read_error`` describe loss that
    already happened upstream, and are rendered in the prompt as well as stored in
    queryable metadata.
    """

    path: NonEmptyText
    snapshot: NonEmptyText
    content: str | None
    sha256: str | None = None
    byte_count: Annotated[int, Field(ge=0)] | None = None
    truncated: bool = False
    read_error: str | None = None

    @model_validator(mode="after")
    def unavailable_content_is_explained(self) -> "ArtifactSnapshot":
        if self.content is None and not self.read_error:
            raise ValueError("an artifact without content must include read_error")
        return self

    @property
    def reference(self) -> ArtifactReference:
        return ArtifactReference(path=self.path, snapshot=self.snapshot)


class EvidenceCaveat(StrictModel):
    """A stored and prompt-visible qualification on the evidence."""

    code: NonEmptyText
    description: NonEmptyText
    source: Literal["judge_builder", "upstream"]
    messages: list[MessageNumber]
    artifacts: list[ArtifactReference]


class MessageSource(StrictModel):
    """Map a prompt-local number back to the original Inspect trajectory."""

    number: Annotated[int, Field(ge=1)]
    label: NonEmptyText
    source_index: Annotated[int, Field(ge=0)]
    source_id: NonEmptyText
    role: NonEmptyText


class EvidenceBundle(StrictModel):
    rendered_messages: str
    rendered_artifacts: str
    source_message_map: list[MessageSource]
    artifacts: list[ArtifactSnapshot]
    caveats: list[EvidenceCaveat]
    native_reasoning_policy: Literal["excluded"] = "excluded"
    native_reasoning_message_count: Annotated[int, Field(ge=0)]
    native_reasoning_block_count: Annotated[int, Field(ge=0)]
    message_selection: Literal["complete_observable_trajectory_for_stage"] = (
        "complete_observable_trajectory_for_stage"
    )
    builder_truncated_evidence: Literal[False] = False

    @model_validator(mode="after")
    def unique_sources(self) -> "EvidenceBundle":
        numbers = [item.number for item in self.source_message_map]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("source message numbers must be contiguous from 1")
        artifact_keys = [(item.path, item.snapshot) for item in self.artifacts]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("artifact path/snapshot pairs must be unique")
        return self


def _native_reasoning_block_count(message: ChatMessage) -> int:
    if not isinstance(message, ChatMessageAssistant) or not isinstance(message.content, list):
        return 0
    return sum(isinstance(item, ContentReasoning) for item in message.content)


def _artifact_caveats(artifacts: Sequence[ArtifactSnapshot]) -> list[EvidenceCaveat]:
    caveats: list[EvidenceCaveat] = []
    for artifact in artifacts:
        ref = artifact.reference
        if artifact.truncated:
            caveats.append(EvidenceCaveat(
                code="artifact_content_truncated_upstream",
                description=(
                    "This artifact was truncated before it reached the judge; omitted "
                    "content cannot be treated as evidence of absence."
                ),
                source="upstream",
                messages=[],
                artifacts=[ref],
            ))
        if artifact.content is None:
            caveats.append(EvidenceCaveat(
                code="artifact_content_unavailable",
                description=(
                    "This artifact could not be read before judging: "
                    f"{artifact.read_error}"
                ),
                source="upstream",
                messages=[],
                artifacts=[ref],
            ))
    return caveats


def _render_artifacts(artifacts: Sequence[ArtifactSnapshot]) -> str:
    if not artifacts:
        return "(no artifact snapshots were supplied for this stage)"

    sections: list[str] = []
    for index, artifact in enumerate(artifacts, start=1):
        header = [
            f"[A{index}] path={artifact.path!r} snapshot={artifact.snapshot!r}",
            f"sha256={artifact.sha256 or 'not_recorded'} ",
            f"bytes={artifact.byte_count if artifact.byte_count is not None else 'not_recorded'}",
        ]
        statuses: list[str] = []
        if artifact.truncated:
            statuses.append("TRUNCATED BEFORE JUDGING")
        if artifact.content is None:
            statuses.append(f"UNAVAILABLE: {artifact.read_error}")
        status = f"status={'; '.join(statuses) if statuses else 'complete'}"
        content = artifact.content if artifact.content is not None else "(content unavailable)"
        sections.append("\n".join([
            *header,
            status,
            "----- BEGIN ARTIFACT CONTENT -----",
            content,
            "----- END ARTIFACT CONTENT -----",
        ]))
    return "\n\n".join(sections)


@dataclass(frozen=True)
class RenderedEvidence:
    """Evidence plus Scout's reference extractor for this one fresh prompt."""

    bundle: EvidenceBundle
    extract_references: object


async def render_evidence(
    messages: Sequence[ChatMessage],
    artifacts: Sequence[ArtifactSnapshot] = (),
    *,
    upstream_caveats: Sequence[EvidenceCaveat] = (),
) -> RenderedEvidence:
    """Render every supplied observable message, excluding native reasoning only.

    Tool calls remain attached to assistant messages and tool results remain their own
    messages.  The original message sequence is never sampled or capped here.
    """

    messages = list(messages)
    artifacts = list(artifacts)
    message_ids = [message.id for message in messages]
    if len(message_ids) != len(set(message_ids)):
        raise ValueError("Inspect message IDs must be unique for evidence numbering")

    render, extract_refs, label_for_id = message_numbering(
        MessagesPreprocessor(
            exclude_system=False,
            exclude_reasoning=True,
            exclude_tool_usage=False,
        ),
        label_for_id=True,
    )
    rendered_messages = await render(messages)
    source_map: list[MessageSource] = []
    reasoning_numbers: list[int] = []
    reasoning_block_count = 0
    for source_index, message in enumerate(messages):
        label = label_for_id(message.id)
        if label is None or not label.startswith("M") or not label[1:].isdigit():
            raise RuntimeError(f"Scout did not number Inspect message {message.id!r}")
        number = int(label[1:])
        source_map.append(MessageSource(
            number=number,
            label=label,
            source_index=source_index,
            source_id=message.id,
            role=message.role,
        ))
        message_reasoning_blocks = _native_reasoning_block_count(message)
        if message_reasoning_blocks:
            reasoning_numbers.append(number)
            reasoning_block_count += message_reasoning_blocks

    caveats = list(upstream_caveats)
    caveats.append(EvidenceCaveat(
        code="native_reasoning_excluded_by_policy",
        description=(
            f"Excluded {reasoning_block_count} native reasoning block(s) from "
            f"{len(reasoning_numbers)} message(s). Observable assistant text, tool "
            "calls, and tool results remain included."
        ),
        source="judge_builder",
        messages=reasoning_numbers,
        artifacts=[],
    ))
    caveats.extend(_artifact_caveats(artifacts))

    return RenderedEvidence(
        bundle=EvidenceBundle(
            rendered_messages=rendered_messages,
            rendered_artifacts=_render_artifacts(artifacts),
            source_message_map=source_map,
            artifacts=artifacts,
            caveats=caveats,
            native_reasoning_message_count=len(reasoning_numbers),
            native_reasoning_block_count=reasoning_block_count,
        ),
        extract_references=extract_refs,
    )
