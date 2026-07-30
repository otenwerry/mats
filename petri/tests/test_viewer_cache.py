import pickle
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PETRI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI_ROOT / "lib"))

import viewer_load  # noqa: E402


class ViewerCacheTests(unittest.TestCase):
    def test_concurrent_writers_tolerate_the_same_stale_entry_disappearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            run_dir = root / "run"
            run_dir.mkdir()
            stale = cache_dir / "mode__run__stale.pkl"
            stale.write_bytes(b"stale")
            current = cache_dir / "mode__run__current.pkl"

            real_glob = Path.glob
            both_writers_listed_stale = threading.Barrier(2)

            def synchronized_glob(path, pattern):
                matches = list(real_glob(path, pattern))
                if path == cache_dir and pattern == "mode__run__*.pkl":
                    both_writers_listed_stale.wait(timeout=5)
                return iter(matches)

            with (
                mock.patch.object(viewer_load, "_CACHE_DIR", cache_dir),
                mock.patch.object(viewer_load, "_CACHE_ENABLED", True),
                mock.patch.object(viewer_load, "_cache_key", return_value="current"),
                mock.patch.object(Path, "glob", synchronized_glob),
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(viewer_load._cache_put, run_dir, "mode", {"ok": True})
                        for _ in range(2)
                    ]
                    for future in futures:
                        future.result(timeout=10)

            self.assertFalse(stale.exists())
            with current.open("rb") as fh:
                self.assertEqual(pickle.load(fh), {"ok": True})
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])

    def test_failed_pickle_preserves_existing_complete_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            run_dir = root / "run"
            run_dir.mkdir()
            current = cache_dir / "mode__run__current.pkl"
            with current.open("wb") as fh:
                pickle.dump({"old": "complete"}, fh)

            with (
                mock.patch.object(viewer_load, "_CACHE_DIR", cache_dir),
                mock.patch.object(viewer_load, "_CACHE_ENABLED", True),
                mock.patch.object(viewer_load, "_cache_key", return_value="current"),
                mock.patch.object(viewer_load.pickle, "dump", side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    viewer_load._cache_put(run_dir, "mode", {"new": "partial"})

            with current.open("rb") as fh:
                self.assertEqual(pickle.load(fh), {"old": "complete"})
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])

    def test_build_lock_waits_for_a_running_viewer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "viewer.lock"
            calls = []

            def fake_flock(_fd, operation):
                calls.append(operation)
                if len(calls) == 1:
                    raise BlockingIOError

            with (
                mock.patch.object(viewer_load, "_VIEWER_BUILD_LOCK", lock_path),
                mock.patch.object(viewer_load.fcntl, "flock", side_effect=fake_flock),
            ):
                with viewer_load.viewer_build_lock():
                    self.assertTrue(lock_path.exists())

            self.assertEqual(
                calls,
                [
                    viewer_load.fcntl.LOCK_EX | viewer_load.fcntl.LOCK_NB,
                    viewer_load.fcntl.LOCK_EX,
                    viewer_load.fcntl.LOCK_UN,
                ],
            )


class ViewerLoadIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_judge_sample_stays_loaded_and_queryable(self):
        class EmptyReader:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def index(self):
                if False:
                    yield None

        class EmptyTranscripts:
            def reader(self):
                return EmptyReader()

        usage = SimpleNamespace(
            input_tokens=1,
            output_tokens=10,
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            total_cost=None,
        )
        sample = SimpleNamespace(
            id="seed",
            epoch=1,
            scores={},
            attachments={},
            metadata={},
            role_usage={"target": usage},
            model_usage={"provider/target": usage},
            events=[],
            messages=[],
        )
        model = SimpleNamespace(model="provider/target")
        log = SimpleNamespace(
            eval=SimpleNamespace(
                task="audit_target_seed",
                model_roles={
                    "target": model,
                    "auditor": SimpleNamespace(model="provider/auditor"),
                    "judge": SimpleNamespace(model="provider/judge"),
                },
                metadata={"judge_dimensions": ["reward_hacking"]},
            ),
            samples=[sample],
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(viewer_load, "transcripts_from", return_value=EmptyTranscripts()),
            mock.patch.object(viewer_load, "list_eval_logs", return_value=["fake.eval"]),
            mock.patch.object(viewer_load, "read_eval_log", return_value=log),
            mock.patch.object(viewer_load, "_load_rejudge", return_value={}),
            mock.patch.object(viewer_load, "_load_dim_sidecars", return_value={}),
        ):
            audits = await viewer_load._load_mode_impl(Path(tmp))

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["judge_score_status"], "missing")
        self.assertEqual(audits[0]["integrity_status"], "invalid")
        self.assertIn("judge_score_missing", audits[0]["integrity_issues"])
        self.assertIn("judge_dimensions_missing", audits[0]["integrity_issues"])
        self.assertIn("transcript_unavailable", audits[0]["integrity_issues"])


if __name__ == "__main__":
    unittest.main()
