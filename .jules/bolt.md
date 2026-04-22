## 2024-06-25 - Avoid redundant API authentication in loops
**Learning:** Initializing and authenticating external API clients (like `KaggleApi`) inside a function called within a loop causes significant redundant network overhead.
**Action:** Use dependency injection. Instantiate and authenticate the API client once outside the loop, and pass the authenticated instance to the function to avoid redundant authentication calls.
