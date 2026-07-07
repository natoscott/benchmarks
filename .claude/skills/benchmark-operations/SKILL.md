# Benchmark Operations

Operational patterns for running benchmarks on remote OpenShift clusters.
These patterns are established across `llm-d-batch-eval` and
`llm-d-inference-scheduling-eval` and should be followed in all eval projects.

## Critical Rules

1. **Check for running benchmarks before starting new ones.** Before launching
   `run-all-scenarios.sh` or `run-benchmark.sh`, check for existing processes:
   ```bash
   ps aux | grep 'run-all-scenarios\|run-benchmark' | grep -v grep
   ```
   Kill stale processes before restarting. Multiple concurrent benchmark
   processes on the same cluster corrupt results and fight over GPU resources.

2. **Clean partial results before restarting.** Failed or interrupted runs may
   leave directories without `benchmark-config.txt` (the completion marker).
   The skip logic (`if [ -f benchmark-config.txt ]`) only works when the marker
   exists. Remove partial result directories before re-running.

3. **Always verify replica count exercises the feature under test.** EPP
   scheduler comparisons require multiple replicas (endpoints) — with a single
   replica, all configs produce identical results because there is no routing
   choice. Size replicas to fill one node before spanning multiple nodes.

## Result Directory and Idempotent Runs

Results follow a naming convention that encodes all run parameters:
```
results/${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PROFILE}_${CONFIG}_replica${REPLICAS}
```

Each run writes `benchmark-config.txt` as its **last step** — this file acts
as the completion marker. The skip logic at the top of `run-benchmark.sh`:
```bash
if [ -f "${OUTPUT_DIR}/benchmark-config.txt" ]; then
    echo "SKIPPING (already complete): ${OUTPUT_DIR}"
    exit 0
fi
```

This allows `run-all-scenarios.sh` to be restarted safely — completed runs are
skipped, only incomplete or new runs execute. When designing new eval projects,
always follow this pattern: encode all variable parameters in the directory
name, write the completion marker last.

## Log Monitoring

Always pipe benchmark output to a short, meaningful file in `/tmp` for
monitoring with `tail -F`. Do NOT use the Claude task output path — it is
long, session-specific, and not meaningful to the human operator.

```bash
# In run-all-scenarios.sh usage comments:
bash scripts/run-all-scenarios.sh 2>&1 | stdbuf -oL tee /tmp/epp-eval-benchmark.log
# Monitor: tail -F /tmp/epp-eval-benchmark.log

# For batch-eval:
bash scripts/run-all-scenarios.sh 2>&1 | stdbuf -oL tee /tmp/batch-gateway-benchmark.log
```

Use `stdbuf -oL` for line-buffered output so `tail -F` shows progress in
real time. Use `tail -F` (capital F) to handle log file rotation/recreation.

## New Evaluation Area Checklist

Before starting work on a new benchmark scenario or evaluation area, verify
that the following are available locally. Prompt the user if any are missing.

1. **Source code for the component under test.** Clone or verify local copies
   of the relevant llm-d, vLLM, kserve, or EPP repositories at the version
   being evaluated. The source is essential for understanding configuration
   options, metric names, scorer behavior, and default values — do not rely
   on documentation alone.

2. **Project documentation.** Ensure upstream guides, design docs, and READMEs
   for the feature area are accessible. For llm-d, check `guides/` directories
   (e.g. `guides/optimized-baseline/` for EPP configs). For RHOAI, check the
   product documentation and any linked Jira strategy attachments.

3. **Issue tracker access.** Verify that Jira (via `acli` or similar CLI),
   GitHub issues, and PR history for the relevant repos can be queried. The
   benchmark scope and acceptance criteria often reference specific tickets.
   Ask the user to confirm access if CLI tools are not yet authenticated.

4. **Colleague workloads and profiles.** Check whether team members have
   existing benchmark profiles, workload generators, or results that can be
   reused or compared against. These are often in personal repos or shared
   directories — ask the user rather than searching blindly.

## Artifact Collection Patterns

### guidellm Results

Create the tarball in-pod, then `kubectl cp` the single file down. Do NOT
stream tar through `kubectl exec` stdout — the connection can drop mid-stream
and silently truncate files.

```bash
# Correct: tar in-pod, kubectl cp, extract locally
kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    bash -c 'cd /models/benchmark-output && rm -f warmup.json && \
             tar czf /tmp/guidellm-results.tar.gz *.json'
kubectl cp "${NAMESPACE}/${GUIDELLM_POD}:/tmp/guidellm-results.tar.gz" \
    "${OUTPUT_DIR}/guidellm-results.tar.gz"
tar xzf "${OUTPUT_DIR}/guidellm-results.tar.gz" -C "${OUTPUT_DIR}" && \
    rm -f "${OUTPUT_DIR}/guidellm-results.tar.gz"

# Then compress and strip locally
for f in "${OUTPUT_DIR}"/*.json; do
    [ -f "$f" ] || continue
    zstd -q -f --rm "$f" 2>/dev/null || true
done
for f in "${OUTPUT_DIR}"/*.json.zst; do
    [ -f "$f" ] || continue
    python3 "${SHARED_SCRIPTS}/strip-guidellm-request-content.py" "$f" 2>/dev/null || true
done
```

**Why not streaming tar:** `kubectl exec -- tar czf - | tar xzf -` truncates
when the API server connection drops or buffers flush incompletely. This
produces corrupt JSON that `strip-guidellm-request-content.py` rejects with
`Unterminated string` errors.

### Chunked Transfer for Very Large Files

When `kubectl cp` itself truncates (files >100MB, intermittent API
connectivity), use the shared chunked transfer script:

```bash
SHARED_SCRIPTS="${REPO_ROOT}/../scripts"
"${SHARED_SCRIPTS}/transfer-large-file-chunked.sh" \
    "${KUBECONFIG}" "${NAMESPACE}" "${POD}" \
    "/remote/path/file" "${LOCAL_DIR}/file" "$((256 * 1024))"
```

This reads the file in 256KB chunks via `dd` over `kubectl exec`, reassembles
locally, and verifies size. Use this as a fallback when `kubectl cp` fails on
large files — for typical benchmark artifacts (tarballed JSON, compressed PCP
archives) `kubectl cp` is sufficient.

### PCP Archives

Stop pmlogger first, compress all archive files in-pod, tar, `kubectl cp`.

```bash
# 1. Stop pmlogger so archive files are complete
for PCP_POD in ${PCP_PODS}; do
    kubectl exec -n "${NAMESPACE}" "${PCP_POD}" -- \
        systemctl stop pmlogger 2>/dev/null || true
done
sleep 2

# 2. Compress and download per pod
for PCP_POD in ${PCP_PODS}; do
    kubectl exec -n "${NAMESPACE}" "${PCP_POD}" -- \
        bash -c 'cd /var/log/pcp/pmlogger/$(hostname) && \
                 for f in 2*; do [ -f "$f" ] && zstd -q --rm "$f"; done && \
                 tar cf /tmp/pcp-archives.tar *.zst'
    kubectl cp "${NAMESPACE}/${PCP_POD}:/tmp/pcp-archives.tar" \
        "${ARCHIVE_DIR}/pcp-archives.tar"
    tar xf "${ARCHIVE_DIR}/pcp-archives.tar" -C "${ARCHIVE_DIR}" && \
        rm -f "${ARCHIVE_DIR}/pcp-archives.tar"
done
```

**File glob for PCP archives:** Use `2*` (matches year prefix) to catch all
three archive file types: `20260706.11.20.0` (data), `20260706.11.20.index`,
`20260706.11.20.meta`. The older `[0-9]*` pattern misses `.index` and `.meta`
files. Do NOT use `*.index *.meta` as separate globs — they fail under
`set -e` when no matching files exist.

## Workload Readiness

When scaling LLMInferenceServices, wait for all workload pods to pass
readiness probes before running guidellm. The LLMInferenceService `Ready`
condition can be True while the kserve controller is still rolling out new
pods (especially after a `baseRefs` change that triggers a Deployment update).

```bash
# Wait for all replicas to be Ready
for i in $(seq 1 360); do
    READY_COUNT=$(kubectl get pods -n "${NAMESPACE}" \
        -l "app.kubernetes.io/name=${LLM_SERVICE_NAME},kserve.io/component=workload" \
        -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
        2>/dev/null | grep -c True 2>/dev/null || true)
    READY_COUNT="${READY_COUNT:-0}"
    if [[ "${READY_COUNT}" -ge "${REPLICAS}" ]]; then break; fi
    sleep 5
done
```

Then verify the gateway returns 200 (not 503) before starting guidellm.

## set -e / pipefail Hazards

All benchmark scripts use `set -euo pipefail`. Common traps:

- **`(( expr ))` returns 1 when false.** Use `[[ "$x" -ge "$y" ]]` instead.
- **`grep` returns 1 on no match.** Wrap with `{ grep ... || true; }` or
  `|| echo 0` when the result feeds a variable under pipefail.
- **`ls *.ext | wc -l`** fails under pipefail when no files match. Use
  `find ... | wc -l` instead.
- **`kubectl get ... -o jsonpath`** returns empty (not error) when no resources
  exist, but the kubectl command itself can fail. Always add `|| echo ""`.
