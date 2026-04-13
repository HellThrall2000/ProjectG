## 2024-05-18 - Prevent redundant API authentication in loops
**Learning:** Instantiating and authenticating an API client inside a loop causes redundant network calls and significantly degrades performance. Dependency injection provides a clean way to pass authenticated clients to functions without breaking their standalone usage.
**Action:** Use optional dependency injection parameters (e.g., `api: Client = None`) to allow passing an authenticated client into functions that interact with external APIs, especially if they are called iteratively.
