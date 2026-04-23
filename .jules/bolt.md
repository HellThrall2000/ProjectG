## 2026-04-23 - Avoid redundant Kaggle API authentication
**Learning:** Instantiating and authenticating `KaggleApi` inside a loop for batch downloads causes redundant network calls and authentication overhead, creating a bottleneck.
**Action:** Instantiate and authenticate `KaggleApi` once outside the loop and pass it via dependency injection to functions needing API access.
