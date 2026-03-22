## 2024-05-18 - Pandas DataFrame Iteration Optimization
**Learning:** Iterating over a Pandas DataFrame with `iterrows()` creates a new Series object for each row, causing significant overhead. `itertuples(index=False)` is measurably faster as it yields namedtuples and avoids Series instantiation.
**Action:** Prioritize `itertuples(index=False)` with attribute-style dot notation over `iterrows()` when row-by-row DataFrame iteration is required.
