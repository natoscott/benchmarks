#!/usr/bin/env python3
"""Strip request_args and output fields from guidellm-results.json.zst files.

These fields contain raw prompt text and model response content which are not
needed for analysis and account for the majority of file size (~86% reduction).
All metrics, throughput, latency, and statistical data are preserved.

Usage:
    # Strip a single file (used by run-benchmark.sh after each run):
    python3 scripts/strip-guidellm-request-content.py <file.json.zst>

    # Strip all results under a benchmark directory (batch mode):
    python3 scripts/strip-guidellm-request-content.py [--dry-run]
"""

import json
import subprocess
import sys
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

    result = subprocess.run(["zstd", "-d", "-c", str(path)], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"zstd decompress failed: {result.stderr.decode()}")

    data = json.loads(result.stdout)
    stripped = strip_fields(data)
    stripped_json = json.dumps(stripped, separators=(",", ":")).encode()

    if DRY_RUN:
        proc = subprocess.run(["zstd", "-q", "-19", "-c"], input=stripped_json, capture_output=True)
        return original_size, len(proc.stdout)

    # Atomically replace: write to .tmp then rename
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
        return original_size, path.stat().st_size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main():
    # Single-file mode: argument is a specific .json.zst path
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        path = Path(args[0])
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        try:
            before, after = process_file(path)
            pct = (1 - after / before) * 100 if before else 0
            print(f"Stripped {path.name}: {before/1024:.0f}KB -> {after/1024:.0f}KB ({pct:.0f}% reduction)")
        except Exception as e:
            print(f"ERROR stripping {path}: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode: find all guidellm-results.json.zst under results/
    repo_root = Path(__file__).parent.parent
    files = sorted(repo_root.glob("*/results/*/guidellm-results.json.zst"))
    if not files:
        # Also try relative to cwd (when run from inside a benchmark dir)
        files = sorted(Path.cwd().glob("results/*/guidellm-results.json.zst"))

    if not files:
        print("No guidellm-results.json.zst files found.")
        return

    print(f"{'DRY RUN - ' if DRY_RUN else ''}Processing {len(files)} files...")

    total_before = total_after = 0
    errors = []

    for i, path in enumerate(files, 1):
        try:
            before, after = process_file(path)
            total_before += before
            total_after += after
            pct = (1 - after / before) * 100 if before else 0
            print(f"[{i:3d}/{len(files)}] {pct:5.1f}%  "
                  f"{before/1024/1024:6.1f}MB -> {after/1024/1024:5.1f}MB  {path.parent.name}")
        except Exception as e:
            before = path.stat().st_size
            total_before += before
            total_after += before
            print(f"[{i:3d}/{len(files)}] SKIP (corrupt): {path.parent.name}: {e}")
            errors.append((path, e))

    print(f"\nTotal: {total_before/1024/1024/1024:.2f} GB -> "
          f"{total_after/1024/1024/1024:.2f} GB  "
          f"({(1 - total_after/total_before)*100:.1f}% reduction)")
    if errors:
        print(f"\n{len(errors)} files skipped (corrupt/invalid JSON):")
        for p, _ in errors:
            print(f"  {p}")


if __name__ == "__main__":
    main()
