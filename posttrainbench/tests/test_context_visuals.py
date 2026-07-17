import unittest

from lib import context_visuals as cv


class ContextVisualsTest(unittest.TestCase):
    def test_hack_events_include_canonical_first_and_direct_actions(self):
        first, events = cv._hack_events({
            "final": {"first_hack_event": 12},
            "turn_kinds": {
                "4": {"kind": "context"},
                "12": {"kind": "hack"},
                "19": {"kind": "hack"},
            },
        })
        self.assertEqual(first, 12)
        self.assertEqual(events, [12, 19])

    def test_tool_result_annotation_maps_to_preceding_call(self):
        calls = [
            cv.ContextCall(1, 2, 3, 10_000),
            cv.ContextCall(2, 6, 7, 20_000),
        ]
        unmapped = cv._map_annotations(calls, [4, 7], first_hack_event=4)
        self.assertEqual(unmapped, [])
        self.assertEqual(calls[0].hack_events, [4])
        self.assertTrue(calls[0].is_first_hack)
        self.assertEqual(calls[1].hack_events, [7])

    def test_compaction_distance_counts_model_calls_after_boundary(self):
        calls = [
            cv.ContextCall(1, 2, 3, 10_000),
            cv.ContextCall(2, 8, 9, 12_000),
            cv.ContextCall(3, 12, 13, 15_000),
        ]
        cv._map_compactions(calls, [5])
        self.assertIsNone(calls[0].calls_since_compaction)
        self.assertEqual(calls[1].calls_since_compaction, 1)
        self.assertEqual(calls[2].calls_since_compaction, 2)

    def test_rank_is_computed_within_one_trajectory(self):
        calls = [
            cv.ContextCall(1, 1, 1, 10),
            cv.ContextCall(2, 2, 2, 30),
            cv.ContextCall(3, 3, 3, 20),
            cv.ContextCall(4, 4, 4, 40),
        ]
        self.assertEqual(cv._rank_percent(calls[2], calls), 50.0)


if __name__ == "__main__":
    unittest.main()
