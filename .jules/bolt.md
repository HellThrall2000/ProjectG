## 2024-05-24 - Pandas DataFrame Iteration
**Learning:** Found that `itertuples(index=False)` is significantly faster than `iterrows()` for iterating over Pandas DataFrames row-by-row, as it avoids creating a Series object for each row.
**Action:** Always prefer `itertuples(index=False)` with attribute-style dot notation (e.g., `row.Chapter`) over `iterrows()` for row-by-row iteration in Pandas.
