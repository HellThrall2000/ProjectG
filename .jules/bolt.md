## 2024-05-24 - [Pandas Regex Vectorization Slower Than Apply]
**Learning:** An attempt to optimize the `get_speaker` function in `ingestion/ingest.py` using vectorized `str.extract` with regex was surprisingly ~2x slower than using `apply()` with a simple Python string `split`. The overhead of regex engine compilation and execution on a small 10k row dataset outweighs the benefits of vectorization here.
**Action:** Retain `apply()` with string operations for simple parsing tasks on small datasets in Pandas; do not blindly assume vectorization via regex is always faster.

## 2024-05-24 - [Pandas Iteration Bottleneck]
**Learning:** `iterrows()` is a severe performance bottleneck when iterating over DataFrames to construct LangChain Document objects, as it creates a Series for every row.
**Action:** Prioritize `itertuples(index=False)` with attribute-style dot notation (e.g., `row.Chapter`) over `iterrows()` for a ~22x speedup in row-wise operations.
