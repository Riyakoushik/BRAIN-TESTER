import sys
from unittest.mock import MagicMock

# 1. Mock the 'chat' module so we don't load heavyweight ML models during testing.
mock_chat = MagicMock()
mock_chat_system_instance = MagicMock()

# Setup dummy values for ChatSystem mock
mock_chat_system_instance.generate_response.return_value = ["Mocked response 1", "Mocked response 2", "Mocked response 3"]
mock_chat_system_instance.orchestrator.get_interaction_count.return_value = 42

mock_chat.ChatSystem.return_value = mock_chat_system_instance
sys.modules['chat'] = mock_chat

# 2. Now import server and config safely
import server
from server import app
from config import config
from fastapi.testclient import TestClient

client = TestClient(app)

# Ensure the chat system global is initialized in server.py
server.chat_system = mock_chat_system_instance

def test_health_endpoint_unauthenticated():
    """Verify that /health is public and does not require an API key."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_endpoints_unauthenticated():
    """Verify that sensitive endpoints return 401 Unauthorized when API key is missing."""
    endpoints = [
        ("post", "/chat", {"message": "hello"}),
        ("post", "/learn", {"message": "hello"}),
        ("post", "/learn/select", {"message": "hello", "choice": 1}),
        ("post", "/evolve", {}),
        ("get", "/stats", None),
    ]

    for method, path, payload in endpoints:
        if method == "post":
            response = client.post(path, json=payload)
        else:
            response = client.get(path)

        assert response.status_code == 401, f"Endpoint {path} did not return 401 when unauthenticated"
        assert response.json()["detail"] == "API Key missing"

def test_endpoints_invalid_key():
    """Verify that sensitive endpoints return 403 Forbidden when an incorrect API key is provided."""
    headers = {"X-API-Key": "invalid-key"}
    endpoints = [
        ("post", "/chat", {"message": "hello"}),
        ("post", "/learn", {"message": "hello"}),
        ("post", "/learn/select", {"message": "hello", "choice": 1}),
        ("post", "/evolve", {}),
        ("get", "/stats", None),
    ]

    for method, path, payload in endpoints:
        if method == "post":
            response = client.post(path, json=payload, headers=headers)
        else:
            response = client.get(path, headers=headers)

        assert response.status_code == 403, f"Endpoint {path} did not return 403 when using invalid key"
        assert response.json()["detail"] == "Invalid API Key"

def test_endpoints_valid_key():
    """Verify that sensitive endpoints return 200 OK when the correct API key is provided."""
    # Use the configured api key from config
    headers = {"X-API-Key": config.api_key}

    # Test /stats GET
    response = client.get("/stats", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"interactions": 42, "model_loaded": True}

    # Test /chat POST
    response = client.post("/chat", json={"message": "hello"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked response 1"}

    # Test /learn POST
    response = client.post("/learn", json={"message": "hello"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"options": ["Mocked response 1", "Mocked response 2", "Mocked response 3"]}

    # Test /learn/select POST
    # We first populate the learn cache so that learn_select succeeds
    server._learn_cache["hello"] = (1.0, ["Mocked response 1", "Mocked response 2", "Mocked response 3"])
    response = client.post("/learn/select", json={"message": "hello", "choice": 1}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"saved": "Mocked response 1"}

    # Test /evolve POST
    response = client.post("/evolve", headers=headers)
    assert response.status_code == 200
    assert "Evolution started in background" in response.json()["status"]
