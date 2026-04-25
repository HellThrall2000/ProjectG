## 2026-04-25 - Redundant Kaggle API Authentication Overhead
**Learning:** In scripts like `ingestion/download_data.py`, initializing and authenticating the `KaggleApi` client inside a loop causes a significant, redundant network/initialization overhead on every iteration.
**Action:** Use dependency injection. Instantiate and authenticate the `KaggleApi` object once outside the loop and pass the authenticated instance into the function downloading the data.
