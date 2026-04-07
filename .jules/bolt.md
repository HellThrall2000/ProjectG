## 2024-05-24 - Pandas iterrows() anti-pattern
**Learning:** Using `iterrows()` to iterate over Pandas DataFrames row-by-row is a performance bottleneck in `ingestion/ingest.py`. It converts each row to a Series, which introduces overhead and loses type information.
**Action:** Always prioritize using `itertuples(index=False)` with attribute-style dot notation (e.g., `row.Chapter`) instead of `iterrows()` for optimized performance when iterating over DataFrames row-by-row.
