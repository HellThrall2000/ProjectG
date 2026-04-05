## 2024-05-14 - Optimize DataFrame Row Iteration
**Learning:** In Pandas, iterating over DataFrames row-by-row using `iterrows()` with dictionary-style lookup is significantly slower due to the overhead of creating Series objects for each row.
**Action:** Use `itertuples(index=False)` combined with attribute-style dot notation (e.g., `row.Chapter`) instead of `iterrows()` to improve iteration performance when looping over rows.
