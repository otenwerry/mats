from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path


PETRI = Path(__file__).resolve().parents[1]
LIB = PETRI / "lib"
for path in (str(PETRI), str(LIB)):
    if path not in sys.path:
        sys.path.insert(0, path)

from inspect_ai.tool import ToolDef, ToolInfo

import exp_continuation as continuation
from fixed_target_tools import (  # noqa: E402
    FIXED_TARGET_TOOL_DEFINITIONS,
    fixed_target_auditor_tools,
    fixed_target_tools_auditor_instruction,
    fixed_target_tools_metadata,
    register_fixed_target_tools,
)


def _tool(name: str, description: str = "A test tool") -> ToolInfo:
    return ToolInfo(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def _event(*tools: ToolInfo):
    return SimpleNamespace(tools=list(tools))


class FixedContinuationToolTests(unittest.TestCase):
    def test_new_audits_share_one_visible_fixed_tool_interface(self):
        self.assertEqual(
            [tool["name"] for tool in FIXED_TARGET_TOOL_DEFINITIONS],
            ["bash", "read_file", "write_file"],
        )
        json.dumps(fixed_target_tools_metadata())
        sample = SimpleNamespace(metadata={"condition": "allow"})
        register_fixed_target_tools(sample)
        self.assertEqual(sample.metadata["tools"], fixed_target_tools_metadata())
        self.assertIn("Do not invent, create, remove", fixed_target_tools_auditor_instruction())

    def test_shared_fixed_auditor_controls_cannot_change_tools(self):
        names = {ToolDef(tool).name for tool in fixed_target_auditor_tools()}
        self.assertIn("send_tool_call_result", names)
        self.assertNotIn("create_tool", names)
        self.assertNotIn("remove_tool", names)
        self.assertNotIn("set_system_message", names)

    def test_exact_stable_tool_definitions_are_preserved(self):
        bash = _tool("bash", "Run a shell command")
        read = _tool("file_read", "Read a file")
        fixed = continuation.fixed_tool_set_from_target_events(
            [_event(bash, read), _event(bash.model_copy(), read.model_copy())],
            traj_id=42,
        )

        self.assertEqual(fixed.source_traj_id, 42)
        self.assertEqual(fixed.names, ["bash", "file_read"])
        self.assertEqual(
            fixed.metadata_tools(),
            [
                bash.model_dump(mode="json", exclude_none=False),
                read.model_dump(mode="json", exclude_none=False),
            ],
        )
        # This also proves the payload Petri will read is ordinary JSON metadata.
        json.dumps(fixed.metadata_tools())

    def test_changed_tool_schema_is_rejected_before_a_run(self):
        with self.assertRaisesRegex(
            SystemExit,
            r"original #42 changed target tool definitions between target calls 1 and 2",
        ):
            continuation.fixed_tool_set_from_target_events(
                [_event(_tool("file_read", "Read a file")),
                 _event(_tool("file_read", "Read a file, including hidden files"))],
                traj_id=42,
            )

    def test_changed_tool_membership_is_rejected_before_a_run(self):
        with self.assertRaisesRegex(SystemExit, r"first=\['bash', 'file_read'\].*call 2=\['bash'\]"):
            continuation.fixed_tool_set_from_target_events(
                [_event(_tool("bash"), _tool("file_read")), _event(_tool("bash"))],
                traj_id=42,
            )

    def test_fixed_tools_are_registered_in_reserved_sample_metadata(self):
        fixed = continuation.fixed_tool_set_from_target_events(
            [_event(_tool("bash"))],
            traj_id=42,
        )
        sample = SimpleNamespace(metadata={"condition": "allow"})
        continuation._register_fixed_target_tools(sample, fixed)

        self.assertEqual(sample.metadata["condition"], "allow")
        self.assertEqual(sample.metadata["tools"], fixed.metadata_tools())

    def test_auditor_can_return_results_but_cannot_change_tools(self):
        names = {
            ToolDef(tool).name for tool in continuation._fixed_continuation_auditor_tools()
        }
        self.assertIn("send_tool_call_result", names)
        self.assertNotIn("create_tool", names)
        self.assertNotIn("remove_tool", names)
        self.assertNotIn("set_system_message", names)


if __name__ == "__main__":
    unittest.main()
