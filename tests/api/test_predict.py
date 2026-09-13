"""
tests/api/test_predict.py

Tests for POST /api/v1/predict on the real FastAPI app, exercising the
real request validation (SalaryPredictionRequest), the real
SalaryInferenceFeatureBuilder, and a real (fake-production) fitted
sklearn pipeline as the model -- see tests/api/conftest.py.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.api


# ======================================================================
# Valid requests
# ======================================================================


class TestValidPrediction:
    def test_valid_request_returns_200(self, client, valid_payload):
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 200

    def test_response_schema(self, client, valid_payload):
        body = client.post("/api/v1/predict", json=valid_payload).json()
        assert set(body.keys()) == {
            "predicted_annual_salary",
            "predicted_log_salary",
            "model_name",
            "model_alias",
            "registered_model_name",
        }

    def test_prediction_values_are_finite(self, client, valid_payload):
        body = client.post("/api/v1/predict", json=valid_payload).json()
        assert isinstance(body["predicted_annual_salary"], float)
        assert isinstance(body["predicted_log_salary"], float)
        assert math.isfinite(body["predicted_annual_salary"])
        assert math.isfinite(body["predicted_log_salary"])
        assert body["predicted_annual_salary"] > 0

    def test_annual_salary_is_expm1_of_log_salary(self, client, valid_payload):
        body = client.post("/api/v1/predict", json=valid_payload).json()
        assert body["predicted_annual_salary"] == pytest.approx(
            math.expm1(body["predicted_log_salary"]), rel=1e-3
        )

    def test_model_metadata_fields(self, client, valid_payload, api_module):
        body = client.post("/api/v1/predict", json=valid_payload).json()
        assert body["model_name"] == "ridge"
        assert body["model_alias"] == api_module.serving_config.model_alias
        assert body["registered_model_name"] == api_module.serving_config.registered_model_name

    def test_optional_fields_can_be_omitted(self, client):
        payload = {
            "title": "Backend Developer",
            "skill_list": "Java|SQL",
            "formatted_experience_level": "Associate",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200

    def test_optional_fields_explicit_null_is_accepted(self, client, valid_payload):
        payload = dict(valid_payload, company_state=None, company_country=None, top_industry=None)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200


# ======================================================================
# Request validation errors (422 -- pydantic)
# ======================================================================


class TestRequestValidationErrors:
    @pytest.mark.parametrize(
        "missing_field",
        ["title", "skill_list", "formatted_experience_level"],
    )
    def test_missing_required_field_returns_422(self, client, valid_payload, missing_field):
        payload = dict(valid_payload)
        del payload[missing_field]
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_empty_title_returns_422(self, client, valid_payload):
        payload = dict(valid_payload, title="")
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_field_type_returns_422(self, client, valid_payload):
        payload = dict(valid_payload, title=12345)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_unknown_extra_field_is_rejected(self, client, valid_payload):
        # SalaryPredictionRequest uses model_config = ConfigDict(extra="forbid").
        payload = dict(valid_payload, not_a_real_field="nope")
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_malformed_json_body_returns_422(self, client):
        response = client.post(
            "/api/v1/predict",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ======================================================================
# Domain validation errors surfaced as 400
# (SalaryInferenceFeatureBuilder / SalaryInferenceService raise ValueError,
#  which the route maps to HTTP 400)
# ======================================================================


class TestDomainValidationErrors:
    def test_whitespace_only_title_returns_400(self, client, valid_payload):
        # Passes pydantic's min_length=1 (3 chars) but fails
        # SalaryInferenceFeatureBuilder._validate_required_text after
        # stripping -> ValueError -> HTTP 400.
        payload = dict(valid_payload, title="   ")
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400

    def test_non_finite_model_output_returns_400(self, monkeypatch, client, valid_payload, api_module):
        monkeypatch.setattr(
            api_module.model_loader, "predict", lambda features: [float("nan")]
        )
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 400

    def test_non_finite_inverse_transform_returns_400(self, monkeypatch, client, valid_payload, api_module):
        # A very large log-salary prediction overflows expm1 to inf.
        monkeypatch.setattr(
            api_module.model_loader, "predict", lambda features: [1000.0]
        )
        response = client.post("/api/v1/predict", json=valid_payload)
        assert response.status_code == 400


# ======================================================================
# Dependency / model failures surfaced as 500
# ======================================================================


class TestDependencyFailures:
    def test_unexpected_model_failure_returns_500(self, monkeypatch, client, valid_payload, api_module):
        def _raise(*_args, **_kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(api_module.model_loader, "predict", _raise)
        response = client.post("/api/v1/predict", json=valid_payload)

        assert response.status_code == 500
        assert response.json()["detail"] == "Prediction failed."
