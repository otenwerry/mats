"""Archive rollback data to S3 and pull it back on demand. PAID (exp_ — AWS spend).

Moves the bulky, no-longer-hot rollback data off the laptop while keeping it a
one-command pull away:

  - every pulled rollout under  mats-local/rollback/results/<dir>   (~17 GB raw)
  - the staged job homes        mats/posttrainbench/rollback/builds  (one tarball)
  - failed prep attempts        mats-local/rollback/failed_prep      (one tarball)

Nothing the viewer reads at runtime is archived (viewer_data stays local; the
per-run workspace snapshots must be built FIRST — see
rollback.build_rollback_workspaces).

Each item is streamed  tar -> zstd -> S3  (no local tarball, so this works with
a nearly-full disk), with its sha256 + sizes recorded in
mats-local/rollback/s3_archive_manifest.json. The lifecycle is three explicit,
resumable steps — nothing local is deleted until its upload has been
re-downloaded and hash-verified:

    uv run python -m rollback.exp_s3_archive push            # upload anything not yet up
    uv run python -m rollback.exp_s3_archive verify          # re-download + sha256 check
    uv run python -m rollback.exp_s3_archive delete-local --yes   # rm verified sources

Getting things back (restores to the exact original location):

    uv run python -m rollback.exp_s3_archive list            # what's archived / local?
    uv run python -m rollback.exp_s3_archive pull <name> [<name> ...]

Bucket: --bucket or PTB_S3_BUCKET (created on first push if missing).
Credentials: the standard boto3 chain (~/.aws/credentials or AWS_* env vars).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from . import config

BUCKET = os.environ.get("PTB_S3_BUCKET", "")
REGION = os.environ.get("PTB_S3_REGION", "us-east-1")
PREFIX = "posttrainbench/rollback/"
MANIFEST = config.ROLLBACK_LOCAL / "s3_archive_manifest.json"
BUILDS_DIR = Path(__file__).resolve().parent / "builds"
ZSTD_LEVEL = "-8"
PROGRESS_EVERY = 100 * 1024 * 1024  # print upload/download progress every 100 MB


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")


def _items() -> list[dict]:
    """Everything that belongs in the archive: one entry per results dir,
    plus builds/ and failed_prep/ as single tarballs."""
    items = []
    results = config.ROLLBACK_RESULTS
    if results.exists():
        # skip dot-entries: results/ contains stray box home-dir droppings
        # (.bashrc, .ssh, .ptb_secrets, …) that are not rollouts — and .ssh /
        # .ptb_secrets hold credentials that must never be uploaded
        for d in sorted(p for p in results.iterdir()
                        if p.is_dir() and not p.name.startswith(".")):
            items.append({"name": d.name, "kind": "results", "source": d,
                          "key": f"{PREFIX}results/{d.name}.tar.zst"})
    for name, src in [("builds", BUILDS_DIR),
                      ("failed_prep", config.ROLLBACK_LOCAL / "failed_prep")]:
        if src.exists() and any(src.iterdir()):
            items.append({"name": name, "kind": name, "source": src,
                          "key": f"{PREFIX}{name}.tar.zst"})
    return items


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class _HashingReader:
    """File-like wrapper that sha256-hashes and counts everything read through it."""

    def __init__(self, raw):
        self.raw, self.sha, self.n = raw, hashlib.sha256(), 0

    def read(self, size=-1):
        chunk = self.raw.read(size)
        if chunk:
            self.sha.update(chunk)
            self.n += len(chunk)
        return chunk


def _progress_printer(name: str):
    state = {"sent": 0, "next": PROGRESS_EVERY}

    def cb(nbytes: int):
        state["sent"] += nbytes
        if state["sent"] >= state["next"]:
            print(f"    ... {name}: {state['sent'] / 1e6:.0f} MB", flush=True)
            state["next"] += PROGRESS_EVERY

    return cb


def _client():
    return boto3.client("s3", region_name=REGION)


def _ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        print(f"creating bucket s3://{BUCKET} in {REGION}")
        kwargs = {"Bucket": BUCKET}
        if REGION != "us-east-1":  # us-east-1 rejects a LocationConstraint
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
        s3.create_bucket(**kwargs)


def push(names: list[str] | None = None) -> None:
    s3 = _client()
    _ensure_bucket(s3)
    manifest = _load_manifest()
    todo = [it for it in _items() if it["name"] not in manifest
            and (not names or it["name"] in names)]
    done_bytes = sum(e["bytes_compressed"] for e in manifest.values())
    print(f"push: {len(todo)} item(s) to upload, {len(manifest)} already up "
          f"({done_bytes / 1e9:.2f} GB compressed).")
    for i, it in enumerate(todo, 1):
        src = it["source"]
        raw_bytes = _dir_bytes(src)
        print(f"[{i}/{len(todo)}] {it['name']}  ({raw_bytes / 1e6:.0f} MB raw)", flush=True)
        tar = subprocess.Popen(["tar", "-C", str(src.parent), "-cf", "-", src.name],
                               stdout=subprocess.PIPE)
        zstd = subprocess.Popen(["zstd", ZSTD_LEVEL, "-T0", "-q", "-c"],
                                stdin=tar.stdout, stdout=subprocess.PIPE)
        tar.stdout.close()
        reader = _HashingReader(zstd.stdout)
        s3.upload_fileobj(reader, BUCKET, it["key"], Callback=_progress_printer(it["name"]))
        if tar.wait() != 0 or zstd.wait() != 0:
            # the upload consumed a broken stream — remove the bad object and stop
            s3.delete_object(Bucket=BUCKET, Key=it["key"])
            sys.exit(f"ERROR: tar/zstd failed for {it['name']}; object removed, not recorded.")
        manifest[it["name"]] = {
            "kind": it["kind"], "source": str(src), "s3_key": it["key"],
            "sha256": reader.sha.hexdigest(), "bytes_compressed": reader.n,
            "bytes_raw": raw_bytes, "uploaded_at": _now(),
            "verified_at": None, "local_deleted_at": None,
        }
        _save_manifest(manifest)
        print(f"    up: {reader.n / 1e6:.0f} MB compressed "
              f"({raw_bytes / max(reader.n, 1):.1f}x)  sha256={reader.sha.hexdigest()[:12]}…")
    print("push done.")


def _download_hash(s3, key: str, sink=None) -> tuple[str, int]:
    """Stream an object, returning (sha256, nbytes); optionally tee into sink."""
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
    sha, n, next_mark = hashlib.sha256(), 0, PROGRESS_EVERY
    for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
        sha.update(chunk)
        n += len(chunk)
        if sink is not None:
            sink.write(chunk)
        if n >= next_mark:
            print(f"    ... {n / 1e6:.0f} MB", flush=True)
            next_mark += PROGRESS_EVERY
    return sha.hexdigest(), n


def verify(names: list[str] | None = None) -> None:
    s3 = _client()
    manifest = _load_manifest()
    todo = {k: v for k, v in manifest.items() if not v["verified_at"]
            and (not names or k in names)}
    print(f"verify: {len(todo)} unverified item(s).")
    bad = []
    for i, (name, e) in enumerate(sorted(todo.items()), 1):
        print(f"[{i}/{len(todo)}] {name} ({e['bytes_compressed'] / 1e6:.0f} MB)", flush=True)
        got, n = _download_hash(s3, e["s3_key"])
        if got == e["sha256"] and n == e["bytes_compressed"]:
            e["verified_at"] = _now()
            _save_manifest(manifest)
        else:
            bad.append(name)
            print(f"    MISMATCH: expected {e['sha256'][:12]}…/{e['bytes_compressed']}B, "
                  f"got {got[:12]}…/{n}B")
    if bad:
        sys.exit(f"verify FAILED for {len(bad)} item(s): {', '.join(bad)} — "
                 "re-push them (delete their manifest entries first).")
    print("verify done: everything matches.")


def delete_local(yes: bool) -> None:
    manifest = _load_manifest()
    todo = {k: v for k, v in manifest.items()
            if v["verified_at"] and not v["local_deleted_at"] and Path(v["source"]).exists()}
    total = sum(v["bytes_raw"] for v in todo.values())
    print(f"delete-local: {len(todo)} verified item(s), {total / 1e9:.2f} GB to free.")
    if not todo:
        return
    if not yes:
        sys.exit("pass --yes to actually delete (only verified uploads are eligible).")
    for name, e in sorted(todo.items()):
        shutil.rmtree(e["source"])
        e["local_deleted_at"] = _now()
        _save_manifest(manifest)
        print(f"  rm {e['source']}")
    print("delete-local done.")


def pull(names: list[str]) -> None:
    s3 = _client()
    manifest = _load_manifest()
    for name in names:
        e = manifest.get(name)
        if not e:
            sys.exit(f"'{name}' not in the manifest — see `list` for archived names.")
        dest = Path(e["source"])
        if dest.exists():
            print(f"= {name}: already present at {dest}, skipping.")
            continue
        print(f"pull {name} ({e['bytes_compressed'] / 1e6:.0f} MB) -> {dest}")
        untar = subprocess.Popen(["tar", "-C", str(dest.parent), "-xf", "-"],
                                 stdin=subprocess.PIPE)
        zstd = subprocess.Popen(["zstd", "-d", "-q", "-c"],
                                stdin=subprocess.PIPE, stdout=untar.stdin)
        got, _ = _download_hash(s3, e["s3_key"], sink=zstd.stdin)
        zstd.stdin.close()
        untar.stdin.close()
        if zstd.wait() != 0 or untar.wait() != 0:
            sys.exit(f"ERROR: extraction failed for {name}.")
        if got != e["sha256"]:
            sys.exit(f"ERROR: {name} downloaded but sha256 MISMATCHES the manifest — "
                     f"treat {dest} as suspect.")
        print(f"  restored {dest}")


def list_items() -> None:
    manifest = _load_manifest()
    if not manifest:
        print("manifest empty — nothing archived yet.")
        return
    w = max(len(n) for n in manifest)
    for name, e in sorted(manifest.items()):
        local = Path(e["source"]).exists()
        state = ("LOCAL+S3" if local else "s3-only") if e["verified_at"] else \
                ("uploaded-UNVERIFIED" if not local else "local, unverified upload")
        print(f"{name:<{w}}  {e['bytes_compressed'] / 1e6:>7.0f} MB  {state}")
    up = sum(e["bytes_compressed"] for e in manifest.values())
    print(f"\n{len(manifest)} archived, {up / 1e9:.2f} GB compressed in "
          f"s3://{BUCKET}/{PREFIX}")


def main():
    global BUCKET
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["push", "verify", "delete-local", "pull", "list"])
    ap.add_argument("names", nargs="*",
                    help="archive names (required for pull; optional filter for push/verify)")
    ap.add_argument("--bucket", default=BUCKET, help="S3 bucket (or PTB_S3_BUCKET)")
    ap.add_argument("--yes", action="store_true", help="confirm delete-local")
    args = ap.parse_args()
    BUCKET = args.bucket
    if not BUCKET:
        sys.exit("no bucket: pass --bucket or set PTB_S3_BUCKET.")
    if args.command == "push":
        push(args.names)
    elif args.command == "verify":
        verify(args.names)
    elif args.command == "delete-local":
        delete_local(args.yes)
    elif args.command == "pull":
        if not args.names:
            sys.exit("pull needs at least one archive name (see `list`).")
        pull(args.names)
    else:
        list_items()


if __name__ == "__main__":
    main()
