## 2024-05-24 - Pandas DataFrame Iteration Performance
**Learning:** For optimized performance when iterating over Pandas DataFrames row-by-row, `iterrows()` is an anti-pattern as it yields a Series object for each row, adding significant overhead. `itertuples(index=False)` returns namedtuples and is drastically faster.
**Action:** Prioritize using `itertuples(index=False)` with attribute-style dot notation (e.g., `row.Chapter`) instead of `iterrows()` for data ingestion and processing loops to improve ingestion speed.
