## 2024-05-24 - Pandas Iteration Anti-Pattern
**Learning:** Found usage of `.iterrows()` in `ingestion/ingest.py` which is famously slow for iterating over Pandas DataFrames as it creates a Series object for each row.
**Action:** Always prefer `.itertuples(index=False)` with attribute access (e.g. `row.Chapter`) when iterating row-by-row over DataFrames to avoid row instantiation overhead.
