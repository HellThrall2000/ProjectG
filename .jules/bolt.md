## 2026-04-12 - Redundant Kaggle API Authentication
**Learning:** Instantiating and authenticating `KaggleApi` inside a loop leads to redundant network overhead.
**Action:** Instantiate and authenticate `KaggleApi` once before loops, and pass it via dependency injection to functions that require it.
