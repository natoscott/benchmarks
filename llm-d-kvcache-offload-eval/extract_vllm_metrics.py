#!/usr/bin/env python3
"""
Extract vLLM performance metrics from PCP Parquet data.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

def load_parquet_dataset(parquet_dir):
    """Load a Parquet dataset directory."""
    try:
        table = pq.read_table(parquet_dir)
        df = table.to_pandas()
        return df
    except Exception as e:
        print(f"Error loading {parquet_dir}: {e}")
        return None

def extract_vllm_metrics(df):
    """Extract key vLLM metrics from PCP dataframe."""

    # Find vLLM-related columns
    vllm_cols = [col for col in df.columns if 'vllm' in col.lower()]

    if not vllm_cols:
        print("No vLLM metrics found")
        return None

    print(f"Found {len(vllm_cols)} vLLM metrics")

    # Extract key metrics if they exist
    metrics = {}

    # Throughput metrics
    if 'openmetrics.vllm.vllm.generation_tokens_total' in df.columns:
        tokens = df['openmetrics.vllm.vllm.generation_tokens_total']
        metrics['generation_tokens_total'] = tokens.max() - tokens.min()

    if 'openmetrics.vllm.vllm.prompt_tokens_total' in df.columns:
        tokens = df['openmetrics.vllm.vllm.prompt_tokens_total']
        metrics['prompt_tokens_total'] = tokens.max() - tokens.min()

    # KV cache metrics
    if 'openmetrics.vllm.vllm.kv_cache_usage_perc' in df.columns:
        metrics['kv_cache_usage_mean'] = df['openmetrics.vllm.vllm.kv_cache_usage_perc'].mean()
        metrics['kv_cache_usage_max'] = df['openmetrics.vllm.vllm.kv_cache_usage_perc'].max()

    # Prefix cache metrics
    if 'openmetrics.vllm.vllm.prefix_cache_queries_total' in df.columns:
        queries = df['openmetrics.vllm.vllm.prefix_cache_queries_total']
        metrics['prefix_cache_queries'] = queries.max() - queries.min()

    if 'openmetrics.vllm.vllm.prefix_cache_hits_total' in df.columns:
        hits = df['openmetrics.vllm.vllm.prefix_cache_hits_total']
        metrics['prefix_cache_hits'] = hits.max() - hits.min()

    # Calculate hit rate
    if 'prefix_cache_queries' in metrics and 'prefix_cache_hits' in metrics:
        if metrics['prefix_cache_queries'] > 0:
            metrics['prefix_cache_hit_rate'] = metrics['prefix_cache_hits'] / metrics['prefix_cache_queries']

    # Request metrics
    if 'openmetrics.vllm.vllm.num_requests_running' in df.columns:
        metrics['requests_running_mean'] = df['openmetrics.vllm.vllm.num_requests_running'].mean()
        metrics['requests_running_max'] = df['openmetrics.vllm.vllm.num_requests_running'].max()

    if 'openmetrics.vllm.vllm.num_requests_waiting' in df.columns:
        metrics['requests_waiting_mean'] = df['openmetrics.vllm.vllm.num_requests_waiting'].mean()
        metrics['requests_waiting_max'] = df['openmetrics.vllm.vllm.num_requests_waiting'].max()

    # Time range
    if 'timestamp' in df.columns:
        metrics['duration_seconds'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()

    return metrics

def main():
    analysis_dir = Path('analysis')

    results = []

    # Process each parquet dataset
    for parquet_dir in sorted(analysis_dir.glob('*.parquet')):
        config_name = parquet_dir.stem
        print(f"\nProcessing: {config_name}")
        print("=" * 60)

        df = load_parquet_dataset(parquet_dir)
        if df is None:
            continue

        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        metrics = extract_vllm_metrics(df)
        if metrics:
            metrics['config'] = config_name
            results.append(metrics)

            print(f"\nExtracted metrics:")
            for key, value in sorted(metrics.items()):
                if key != 'config':
                    print(f"  {key}: {value}")

    # Create summary dataframe
    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.set_index('config')

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(summary_df.to_string())

        # Save to CSV
        output_file = 'analysis/vllm_metrics_summary.csv'
        summary_df.to_csv(output_file)
        print(f"\nSaved summary to: {output_file}")
    else:
        print("\nNo metrics extracted")

if __name__ == '__main__':
    main()
