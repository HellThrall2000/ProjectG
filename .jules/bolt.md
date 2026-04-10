## 2024-04-10 - Kaggle API Dependency Injection Optimization
**Learning:** Found that `KaggleApi().authenticate()` takes significant time when called repeatedly within loops for batch data downloads, as it re-evaluates configuration and potentially checks network each time.
**Action:** Always instantiate and authenticate external APIs exactly once before loops and pass the instance to internal functions using dependency injection.
