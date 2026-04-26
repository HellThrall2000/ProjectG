## 2024-05-18 - Kaggle API Authentication Overhead
**Learning:** Instantiating and authenticating `KaggleApi` within a loop creates massive network overhead. Each `api.authenticate()` call blocks synchronously and delays the overall batch download execution time significantly when scaled across many datasets.
**Action:** When performing batch downloads using external APIs like Kaggle, always authenticate exactly once outside the loop. Use dependency injection to pass the authenticated client directly to the operational functions inside the loop.
