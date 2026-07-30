"""The real tools must present EXACTLY the simulated audits' wire schema.

Byte-identical names/descriptions/schemas are what make real-environment and simulated
trajectories directly comparable (and what a future continuation across the two modes
depends on — exp_continuation aborts if tool fingerprints differ).
"""

import sys
import unittest
from pathlib import Path

PETRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from fixed_target_tools import (  # noqa: E402
    fixed_target_tools_fingerprint,
    fixed_target_tools_metadata,
)
from real_target_tools import (  # noqa: E402
    real_target_tools,
    real_target_tools_infos,
)


class RealToolSchemaTests(unittest.TestCase):
    def test_tool_infos_byte_equal_fixed_definitions(self):
        real = [
            info.model_dump(mode="json", exclude_none=False)
            for info in real_target_tools_infos()
        ]
        self.assertEqual(real, fixed_target_tools_metadata())

    def test_fingerprint_matches_simulated_interface(self):
        import hashlib
        import json

        payload = json.dumps(
            [i.model_dump(mode="json", exclude_none=False) for i in real_target_tools_infos()],
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            hashlib.sha256(payload.encode()).hexdigest(),
            fixed_target_tools_fingerprint(),
        )

    def test_tools_are_registered_inspect_tools(self):
        tools = real_target_tools()
        self.assertEqual(len(tools), 3)
        from inspect_ai.tool import ToolDef

        names = [ToolDef(t).name for t in tools]
        self.assertEqual(names, ["bash", "read_file", "write_file"])


if __name__ == "__main__":
    unittest.main()
