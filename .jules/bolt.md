## 2026-04-17 - Optimize Kaggle API Authentication
**Learning:** Instantiating and authenticating API clients like KaggleApi inside loops causes redundant network/authentication overhead during batch operations.
**Action:** Use dependency injection to instantiate and authenticate the API client once, passing it to the processing functions.
