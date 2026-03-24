#!/usr/bin/env python3
"""Strip request_args and output fields from guidellm-results.json.zst files.

These fields contain the raw prompt text and model response content which are
not needed for analysis and account for the majority of file size. All metrics,
throughput, latency, and statistical data are preserved.

Usage:
    python3 scripts/strip-guidellm-request-content.py [--dry-run]
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FIELDS_TO_STRIP = {"request_args", "output"}
DRY_RUN = "--dry-run" in sys.argv


def strip_fields(obj):
    """Recursively strip FIELDS_TO_STRIP from any dict in the structure."""
    if isinstance(obj, dict):
        return {
            k: strip_fields(v)
            for k, v in obj.items()
            if k not in FIELDS_TO_STRIP
        }
    elif isinstance(obj, list):
        return [strip_fields(item) for item in obj]
    return obj


def process_file(path: Path) -> tuple[int, int]:
    """Decompress, strip, recompress. Returns (original_bytes, new_bytes)."""
    original_size = path.stat().st_size

    # Decompress
    result = subprocess.run(
        ["zstd", "-d", "-c", str(path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"zstd decompress failed: {result.stderr.decode()}")

    data = json.loads(result.stdout)
    stripped = strip_fields(data)
    stripped_json = json.dumps(stripped, separators=(",", ":")).encode()

    if DRY_RUN:
        # Estimate compressed size by compressing to /dev/null
        proc = subprocess.run(
            ["zstd", "-q", "-19", "-c"],
            input=stripped_json,
            capture_output=True,
        )
        new_size = len(proc.stdout)
        return original_size, new_size

    # Write to temp file then atomically replace
    tmp = path.with_suffix(".tmp.zst")
    try:
        proc = subprocess.run(
            ["zstd", "-q", "-19", "-c", "-o", str(tmp)],
            input=stripped_json,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zstd compress failed: {proc.stderr.decode()}")
        tmp.rename(path)
        new_size = path.stat().st_size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return original_size, new_size


def main():
    repo_root = Path(__file__).parent.parent
    files = sorted(repo_root.glob("results/*/guidellm-results.json.zst"))

    if not files:
        print("No guidellm-results.json.zst files found.")
        return

    print(f"{'DRY RUN - ' if DRY_RUN else ''}Processing {len(files)} files...")
    print()

    total_before = 0
    total_after = 0
    errors = []

    for i, path in enumerate(files, 1):
        rel = path.relative_to(repo_root)
        try:
            before, after = process_file(path)
            total_before += before
            total_after += after
            pct = (1 - after / before) * 100 if before else 0
            print(f"[{i:3d}/{len(files)}] {pct:5.1f}% reduction  "
                  f"{before/1024/1024:6.1f}MB -> {after/1024/1024:5.1f}MB  {rel.parent.name}")
        except Exception as e:
            before = path.stat().st_size
            total_before += before
            total_after += before  # unchanged — count at original size
            print(f"[{i:3d}/{len(files)}] SKIP (corrupt): {rel.parent.name}: {e}")
            errors.append((rel, e))

    print()
    print(f"Total: {total_before/1024/1024/1024:.2f} GB -> "
          f"{total_after/1024/1024/1024:.2f} GB  "
          f"({(1 - total_after/total_before)*100:.1f}% reduction)")
    if errors:
        print(f"\n{len(errors)} files skipped (corrupt/invalid JSON) — left unchanged:")
        for rel, e in errors:
            print(f"  {rel}")


if __name__ == "__main__":
    main()
