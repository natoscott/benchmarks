#!/usr/bin/env python3
"""
Analyze new vLLM KV cache CPU offload benchmark results
from PCP archive guidellm data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the guidellm CSV data
print("Loading guidellm data...")
df = pd.read_csv('guidellm_data.csv', parse_dates=['Time'])

# Get list of all UUIDs from column names
print("Extracting UUIDs...")
uuids = set()
for col in df.columns:
    if '-' in col and len(col.split('-')) > 1:
        # Extract UUID from column name (last 5 parts after the metric name)
        parts = col.split('-')
        if len(parts) >= 6:  # Has UUID
            uuid = '-'.join(parts[-5:])
            uuids.add(uuid)

uuids = sorted(list(uuids))
print(f"Found {len(uuids)} unique benchmark runs")

# For each UUID, extract the final metric values
results = []

for uuid in uuids:
    # Get columns for this UUID
    tps_col = f'guidellm.tokens_per_second.total.mean-{uuid}'
    ttft_col = f'guidellm.time_to_first_token_ms.total.mean-{uuid}'
    tpot_col = f'guidellm.time_per_output_token_ms.total.mean-{uuid}'
    concurrency_col = f'guidellm.concurrency-{uuid}'

    # Find rows where this benchmark has data
    data_mask = df[tps_col].notna()

    if data_mask.sum() == 0:
        continue

    # Get the last non-null value (final result)
    tps = df.loc[data_mask, tps_col].iloc[-1]
    ttft = df.loc[data_mask, ttft_col].iloc[-1] if ttft_col in df.columns else None
    tpot = df.loc[data_mask, tpot_col].iloc[-1] if tpot_col in df.columns else None
    concurrency = df.loc[data_mask, concurrency_col].iloc[-1] if concurrency_col in df.columns else None

    # Get time range for this benchmark
    first_time = df.loc[data_mask, 'Time'].iloc[0]
    last_time = df.loc[data_mask, 'Time'].iloc[-1]

    results.append({
        'uuid': uuid,
        'tokens_per_second': tps,
        'ttft_ms': ttft,
        'tpot_ms': tpot,
        'concurrency': concurrency,
        'start_time': first_time,
        'end_time': last_time,
        'duration_sec': (last_time - first_time).total_seconds()
    })

results_df = pd.DataFrame(results)

# Sort by start time to see the chronological order
results_df = results_df.sort_values('start_time').reset_index(drop=True)

print(f"\nExtracted {len(results_df)} benchmark results")
print("\nFirst few results:")
print(results_df.head(10)[['uuid', 'concurrency', 'tokens_per_second', 'ttft_ms', 'tpot_ms', 'start_time']])

# Save results
results_df.to_csv('extracted_results.csv', index=False)
print("\nSaved to extracted_results.csv")

# Try to identify patterns in concurrency
if 'concurrency' in results_df.columns and results_df['concurrency'].notna().any():
    print("\nConcurrency distribution:")
    print(results_df['concurrency'].value_counts().sort_index())
else:
    print("\nConcurrency metric not available, will need to infer from chronological order")
    print("Expected pattern: 60 runs = 2 models × 3 configs × 10 concurrency levels")
    print(f"Total runs: {len(results_df)}")
