## 2024-05-24 - [Itertuples Optimization]
**Learning:** `iterrows()` in pandas is notoriously slow and an anti-pattern for performance-critical data ingestion because it boxes each row into a Series object. `itertuples(index=False)` is a much faster alternative that returns namedtuples.
**Action:** Replace `iterrows()` with `itertuples(index=False)` in `ingestion/ingest.py` to optimize data parsing during vector store population.
