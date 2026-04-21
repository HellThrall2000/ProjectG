## 2024-05-18 - [Kaggle API Authentication Overhead]
**Learning:** Instantiating and authenticating the Kaggle API object repeatedly within a loop creates unnecessary network requests and overhead, slowing down batch dataset processing.
**Action:** Use dependency injection to pass an authenticated `KaggleApi` instance to functions within loops instead of authenticating inside the function.