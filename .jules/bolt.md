## 2024-04-14 - [Optimize DataFrame iteration in Pandas]
**Learning:** `itertuples(index=False)` is dramatically faster than `iterrows()` when iterating over pandas DataFrames row-by-row (benchmark showed a drop from ~0.55s to ~0.02s for 10k rows).
**Action:** When row-by-row iteration is necessary over vectorized operations, prioritize using `itertuples(index=False)` and dot notation for column access to maximize performance.
