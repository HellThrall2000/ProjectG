## 2024-05-24 - Pandas iteration performance
**Learning:** Iterating over pandas DataFrames using `iterrows()` is slow because it creates a Series for each row. `itertuples(index=False)` is much faster because it returns namedtuples and preserves data types.
**Action:** For optimized performance when iterating over Pandas DataFrames row-by-row, prioritize using `itertuples(index=False)` with attribute-style dot notation (e.g., `row.Chapter`) instead of `iterrows()`.
