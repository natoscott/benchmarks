#!/usr/bin/env python3
"""
Analyze vLLM KV Cache CPU Offload Benchmark Results from PCP Parquet data
"""

import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the Parquet data
print("Loading Parquet data...")
df = pd.read_parquet('benchmark-data.parquet')

print(f"Loaded {len(df)} rows")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn names (first 50):")
print(list(df.columns)[:50])

# Check for guidellm metrics
guidellm_cols = [col for col in df.columns if col.startswith('guidellm')]
print(f"\nFound {len(guidellm_cols)} guidellm metrics")
print("GuideLLM columns:")
for col in sorted(guidellm_cols)[:20]:
    print(f"  {col}")

print("\nData shape:", df.shape)
print("\nFirst few timestamps:")
print(df.index[:5] if hasattr(df, 'index') else "No index")
