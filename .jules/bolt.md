## 2024-04-01 - [Data Ingestion Performance]
**Learning:** `pandas.DataFrame.iterrows()` is extremely slow for iterating over rows. `itertuples(index=False)` is about 18x faster in local benchmarks for creating Document objects from a DataFrame.
**Action:** Use `itertuples(index=False)` and attribute-style dot notation (e.g., `row.Chapter`) instead of `iterrows()` when iterating row-by-row in pandas for significant performance gains.
