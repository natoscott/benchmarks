#!/usr/bin/env python3
"""Monitor batch job progress during a benchmark run.

Polls the batch-gateway API for job status and writes a timeline JSON.
Runs inside the guidellm pod via kubectl exec, or directly if the
batch-gateway API is reachable.

Usage:
    # From the guidellm pod:
    python3 monitor-batch-progress.py --url http://batch-gateway-apiserver:8000 \
        --output /tmp/batch-timeline.json

    # Via kubectl exec:
    kubectl exec -n llm-d-batch deployment/guidellm -- \
        python3 /dev/stdin --url http://... --output /tmp/timeline.json \
        < scripts/monitor-batch-progress.py
"""

import argparse
import json
import sys
import time
from urllib.request import urlopen, Request


def poll_batches(base_url, token, limit=50):
    """Fetch all batch jobs and return their status."""
    try:
        req = Request(
            f"{base_url}/v1/batches?limit={limit}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = json.loads(urlopen(req, timeout=5).read())
        jobs = []
        for b in data.get("data", []):
            rc = b.get("request_counts", {})
            jobs.append({
                "id": b.get("id", ""),
                "status": b.get("status", "unknown"),
                "completed": rc.get("completed", 0),
                "failed": rc.get("failed", 0),
                "total": rc.get("total", 0),
            })
        return jobs
    except Exception as e:
        print(f"  poll error: {e}", file=sys.stderr, flush=True)
        return []


def all_terminal(jobs):
    if not jobs:
        return False
    return all(
        j["status"] in ("completed", "failed", "cancelled", "expired")
        for j in jobs
    )


def main():
    parser = argparse.ArgumentParser(description="Monitor batch job progress")
    parser.add_argument("--url", required=True, help="Batch gateway base URL")
    parser.add_argument("--output", required=True, help="Output timeline JSON path")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    parser.add_argument("--timeout", type=int, default=7200, help="Max runtime (seconds)")
    parser.add_argument("--token", default="benchmark", help="Auth token")
    parser.add_argument("--limit", type=int, default=50, help="Max jobs to query")
    args = parser.parse_args()

    timeline = []
    start = time.time()

    print(f"monitor-batch-progress: polling {args.url} every {args.interval}s", flush=True)

    while True:
        elapsed = int(time.time() - start)
        if elapsed >= args.timeout:
            print(f"monitor-batch-progress: timeout after {elapsed}s", flush=True)
            break

        jobs = poll_batches(args.url, args.token, args.limit)
        timeline.append({"elapsed": elapsed, "jobs": jobs})

        parts = [f"{j['id'][-8:]}: {j['status']} {j['completed']}/{j['total']}" for j in jobs]
        print(f"  batch [{elapsed}s]: {' | '.join(parts) if parts else 'no jobs'}", flush=True)

        if all_terminal(jobs):
            print("monitor-batch-progress: all jobs terminal", flush=True)
            break

        time.sleep(args.interval)

    with open(args.output, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"monitor-batch-progress: {len(timeline)} samples -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
