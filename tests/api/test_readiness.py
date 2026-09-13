"""
tests/api/test_readiness.py

NOTE ON SCOPE
-------------
The current `salary_api.py` does not expose a dedicated `/readiness`
route -- only `/health`, whose `model_loaded` flag and `degraded` status
already serve that purpose (this is stated as a known simplification of
the current serving layer, not an assumption made here). If a dedicated
`/readiness` endpoint is added later, point these tests at it instead.

The app's `lifespan` handler calls `model_loader.load()` synchronously at
startup and re-raises on failure (fail-fast: the app will not finish
starting without a model). That means:

  - A live, started TestClient always reports `model_loaded=True` --
    there's no way to reach a "degraded" `/health` response through a
    normal request once the app is up.
  - The "degraded" response *shape* is still real, reachable code
    (`health()`'s `if not model_loader.is_loaded:` branch), so it's
    tested here by calling the route function directly rather than
    through a live app.
  - The "model cannot be loaded" failure mode is tested at the level
    where it actually manifests today: app startup itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


class TestReadySignalViaHealthEndpoint:
    def test_ready_when_model_is_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True


class TestDegradedResponseShape:
    def test_degraded_shape_when_model_not_loaded(self, api_module):
        # Exercises health()'s "not loaded" branch directly -- the live
        # app's fail-fast lifespan makes this otherwise unreachable via
        # a real HTTP request (see module docstring).
        api_module.model_loader._model = None
        body = api_module.health()

        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["registered_model_name"] == api_module.serving_config.registered_model_name
        assert body["model_alias"] == api_module.serving_config.model_alias


class TestStartupFailureBehavior:
    def test_app_refuses_to_start_when_model_cannot_be_loaded(self, monkeypatch, api_module):
        api_module.model_loader._model = None

        def _raise_load_error():
            raise RuntimeError(
                f"Unable to load the registered salary model from URI: "
                f"{api_module.serving_config.model_uri}"
            )

        monkeypatch.setattr(api_module.model_loader, "load", _raise_load_error)

        with pytest.raises(RuntimeError):
            with TestClient(api_module.app):
                pass  # lifespan startup must raise before this body ever runs
