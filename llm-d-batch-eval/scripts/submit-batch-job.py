#!/usr/bin/env python3
"""Submit a single batch job to the batch-gateway API server.

Uploads a JSONL file, creates a batch with the specified completion window,
and prints the batch ID and status.

Usage:
    python3 scripts/submit-batch-job.py \
        --url http://batch-gateway-apiserver.llm-d-batch.svc.cluster.local:8000 \
        --file job-a.jsonl \
        --window 30m
"""

import argparse
import json
import sys
from urllib.request import urlopen, Request


def upload_file(base_url, filepath, token):
    with open(filepath, "rb") as f:
        file_data = f.read()

    boundary = "----BatchBenchmarkBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        f"batch\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filepath}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = Request(
        f"{base_url}/v1/files",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    resp = json.loads(urlopen(req).read())
    return resp["id"]


def create_batch(base_url, file_id, completion_window, token):
    body = json.dumps({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": completion_window,
    }).encode()

    req = Request(
        f"{base_url}/v1/batches",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    resp = json.loads(urlopen(req).read())
    return resp


def main():
    parser = argparse.ArgumentParser(description="Submit a batch job")
    parser.add_argument("--url", required=True, help="Batch gateway base URL")
    parser.add_argument("--file", required=True, help="JSONL input file path")
    parser.add_argument("--window", required=True, help="Completion window (e.g. 30m, 2h, 24h)")
    parser.add_argument("--token", default="benchmark", help="Auth token (default: benchmark)")
    args = parser.parse_args()

    file_id = upload_file(args.url, args.file, args.token)
    print(f"Uploaded: {file_id}", flush=True)

    batch = create_batch(args.url, file_id, args.window, args.token)
    print(f"Batch: {batch['id']} status={batch['status']}", flush=True)

    result = {"file_id": file_id, "batch_id": batch["id"], "status": batch["status"]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
