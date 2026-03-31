
## 2024-11-13 - Pandas row-by-row iteration optimization
**Learning:** `iterrows()` is a known performance bottleneck in Pandas DataFrames due to type inference on every row creation.
**Action:** Use `itertuples(index=False)` instead for significantly faster row traversal, and replace dictionary access `row["Column"]` with attribute access `row.Column`.
