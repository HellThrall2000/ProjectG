## 2024-06-11 - [Optimize Kaggle API Authentication]
**Learning:** Redundant authentication overhead occurs when creating a new `KaggleApi` instance inside a loop (e.g., `download_kaggle_dataset`). This adds significant network and I/O latency for multiple dataset downloads.
**Action:** Instantiate and authenticate `KaggleApi` once outside the loop and pass it to functions via dependency injection to reuse the authenticated session.
