"""
tests/api/test_health.py

Tests for GET /health and GET / on the real FastAPI app
(api/salary_api.py), via TestClient. The app's
`model_loader` is pre-loaded with a real fitted pipeline (see
tests/api/conftest.py) so startup never touches real MLflow.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_body(self, client):
        assert client.get("/").json() == {
            "service": "salary-prediction-api",
            "status": "running",
        }


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        body = client.get("/health").json()
        assert set(body.keys()) == {
            "status",
            "model_loaded",
            "registered_model_name",
            "model_alias",
        }

    def test_health_reports_healthy_when_model_is_loaded(self, client, api_module):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["registered_model_name"] == api_module.serving_config.registered_model_name
        assert body["model_alias"] == api_module.serving_config.model_alias

    def test_health_field_types(self, client):
        body = client.get("/health").json()
        assert isinstance(body["status"], str)
        assert isinstance(body["model_loaded"], bool)
        assert isinstance(body["registered_model_name"], str)
        assert isinstance(body["model_alias"], str)