## 2024-05-15 - [Pandas DataFrame Iteration]
**Learning:** Using `iterrows()` to loop over a Pandas DataFrame is inefficient because it creates a new Series object for each row. `itertuples(index=False)` is significantly faster (in a quick test on 10,000 rows, it took ~0.025s compared to ~0.61s for `iterrows()`) as it returns lightweight namedtuples.
**Action:** When row-by-row iteration over a DataFrame is unavoidable, prefer using `itertuples(index=False)` and access columns via attribute notation (e.g., `row.ColumnName`) instead of `iterrows()`.
