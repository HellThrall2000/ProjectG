## 2024-04-28 - Avoid Pandas optimization journaling
**Learning:** Standard pandas `itertuples` improvements shouldn't be journaled unless there are edge cases or surprising failures.
**Action:** Do not log routine pandas optimizations like `iterrows` to `itertuples`.
