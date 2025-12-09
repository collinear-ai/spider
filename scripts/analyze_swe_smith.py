#!/usr/bin/env python3
"""Analyze SWE-smith dataset to plan optimal processing strategy."""

from datasets import load_dataset
import pandas as pd
from collections import Counter

print("Loading SWE-smith dataset...")
dataset = load_dataset("SWE-bench/SWE-smith", split="train")
df = pd.DataFrame(dataset)

print(f"\n{'='*60}")
print("📊 SWE-SMITH DATASET ANALYSIS")
print(f"{'='*60}\n")

print(f"Total instances: {len(df):,}")
print(f"Unique images: {df['image_name'].nunique():,}")
print(f"Unique repos: {df['repo'].nunique():,}\n")

# Group by repo
print("="*60)
print("📦 TOP 20 REPOSITORIES BY INSTANCE COUNT")
print("="*60)
repo_counts = df.groupby('repo').agg({
    'instance_id': 'count',
    'image_name': 'nunique'
}).rename(columns={'instance_id': 'instances', 'image_name': 'images'})
repo_counts = repo_counts.sort_values('instances', ascending=False)

print(f"\n{'Repo':<40} {'Instances':>10} {'Images':>8} {'Reuse':>8}")
print("-"*70)
for idx, (repo, row) in enumerate(repo_counts.head(20).iterrows(), 1):
    repo_short = repo.replace('swesmith/', '')[:38]
    reuse_ratio = row['instances'] / row['images']
    print(f"{idx:2}. {repo_short:<40} {row['instances']:>10} {row['images']:>8} {reuse_ratio:>7.1f}x")

# Calculate optimal processing order
print(f"\n{'='*60}")
print("⚡ OPTIMAL PROCESSING STRATEGY")
print(f"{'='*60}\n")

total_instances = len(df)
total_time_naive = total_instances * 7  # 7 min per instance if no caching
total_time_optimized = 0

for repo, row in repo_counts.iterrows():
    # First instance: build time + exec time
    # Subsequent instances: exec time only
    repo_time = 5 + (row['instances'] * 2)  # 5 min build + 2 min per instance
    total_time_optimized += repo_time

print(f"Without optimization (random order):")
print(f"  {total_instances:,} instances × 7 min = {total_time_naive:,} min")
print(f"  = {total_time_naive/60:.1f} hours = {total_time_naive/1440:.1f} days\n")

print(f"With repo grouping:")
print(f"  {repo_counts['images'].sum():,} builds × 5 min + {total_instances:,} instances × 2 min")
print(f"  = {total_time_optimized:,} min")
print(f"  = {total_time_optimized/60:.1f} hours = {total_time_optimized/1440:.1f} days\n")

speedup = total_time_naive / total_time_optimized
print(f"⚡ Speedup: {speedup:.1f}x faster!\n")

# Multi-machine estimation
num_machines = 8
parallel_time = total_time_optimized / num_machines
print(f"With {num_machines} machines in parallel:")
print(f"  {parallel_time/60:.1f} hours = {parallel_time/1440:.1f} days\n")

# Save repo list for processing
print("="*60)
print("💾 SAVING REPO LIST")
print("="*60)
output_file = "scripts/swe_smith_repos.txt"
with open(output_file, 'w') as f:
    for idx, (repo, row) in enumerate(repo_counts.iterrows(), 1):
        repo_clean = repo.replace('swesmith/', '')
        f.write(f"{idx}. {repo_clean}: {row['instances']} instances, {row['images']} images\n")
print(f"\n✓ Saved to: {output_file}")
print(f"  Use this to create filters for batch processing\n")

print("Example filters:")
print("-" * 60)
for idx, (repo, row) in enumerate(repo_counts.head(5).iterrows(), 1):
    repo_pattern = repo.replace('swesmith/', '').replace('__', r'__')
    print(f"{idx}. instance_filter: \"{repo_pattern}.*\"")
    print(f"   → {row['instances']} instances, est. {5 + row['instances']*2} minutes\n")
