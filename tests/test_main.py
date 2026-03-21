import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_app_initialization():
    """Test that the FastAPI app initializes correctly."""
    # Test that the app is a FastAPI instance
    assert app.title == "FastAPI"
    assert app.version == "0.1.0"

    # Test that a request to an undefined route returns 404 Not Found
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

    # Test that the automatic OpenAPI docs generation is active
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()
