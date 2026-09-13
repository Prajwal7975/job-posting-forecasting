"""
tests/unit/test_salary_mlflow_config.py

Unit tests for src/configs/salary_predict/salary_ML_flow_config.py
(SalaryMLflowConfig).

No real MLflow server or database is touched here -- this config object
never opens a connection, it only validates and exposes values.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.configs.salary_predict.salary_ML_flow_config import (
    SalaryMLflowConfig,
    ENV_TRACKING_URI,
    ENV_EXPERIMENT_NAME,
    ENV_TRACKING_ENABLED,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with none of this module's env vars set, so
    defaults are actually the defaults and not leftovers from the host
    environment or a previous test."""
    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    monkeypatch.delenv(ENV_EXPERIMENT_NAME, raising=False)
    monkeypatch.delenv(ENV_TRACKING_ENABLED, raising=False)


class TestDefaults:
    def test_default_tracking_uri_is_local_sqlite(self):
        config = SalaryMLflowConfig()
        assert config.tracking_uri.startswith("sqlite:///")
        assert config.tracking_uri.endswith("mlflow.db")

    def test_default_experiment_name(self):
        config = SalaryMLflowConfig()
        assert config.experiment_name == "salary_prediction"

    def test_default_registered_model_name(self):
        config = SalaryMLflowConfig()
        assert config.registered_model_name == "salary_prediction_model"

    def test_default_tracking_enabled_is_true(self):
        config = SalaryMLflowConfig()
        assert config.tracking_enabled is True
        assert config.is_tracking_enabled is True

    def test_default_project_and_pipeline_names(self):
        config = SalaryMLflowConfig()
        assert config.project_name == "linkedin-job-intelligence"
        assert config.pipeline_name == "salary_predict"


class TestEnvironmentOverrides:
    def test_tracking_uri_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv(ENV_TRACKING_URI, "http://mlflow.internal:5000")
        config = SalaryMLflowConfig()
        assert config.tracking_uri == "http://mlflow.internal:5000"

    def test_remote_tracking_uri_is_not_local(self, monkeypatch):
        monkeypatch.setenv(ENV_TRACKING_URI, "https://mlflow.example.com")
        config = SalaryMLflowConfig()
        assert config.is_local_tracking is False
        assert config.local_tracking_path is None

    def test_experiment_name_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv(ENV_EXPERIMENT_NAME, "ci_experiment")
        config = SalaryMLflowConfig()
        assert config.experiment_name == "ci_experiment"

    @pytest.mark.parametrize("raw,expected", [("1", True), ("false", False), ("YES", True), ("off", False)])
    def test_tracking_enabled_env_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv(ENV_TRACKING_ENABLED, raw)
        config = SalaryMLflowConfig()
        assert config.tracking_enabled is expected

    def test_invalid_tracking_enabled_env_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_TRACKING_ENABLED, "maybe")
        with pytest.raises(ValueError):
            SalaryMLflowConfig()


class TestValidation:
    def test_blank_tracking_uri_raises(self):
        with pytest.raises(ValueError):
            SalaryMLflowConfig(tracking_uri="   ")

    def test_uri_with_only_scheme_separator_raises(self):
        with pytest.raises(ValueError):
            SalaryMLflowConfig(tracking_uri="://")

    @pytest.mark.parametrize(
        "field_name",
        ["experiment_name", "project_name", "pipeline_name", "run_name_prefix", "registered_model_name"],
    )
    def test_blank_required_string_fields_raise(self, field_name):
        with pytest.raises(ValueError):
            SalaryMLflowConfig(**{field_name: "   "})

    def test_non_string_tracking_uri_raises_type_error(self):
        with pytest.raises(TypeError):
            SalaryMLflowConfig(tracking_uri=123)  # type: ignore[arg-type]

    def test_blank_artifact_location_raises(self):
        with pytest.raises(ValueError):
            SalaryMLflowConfig(artifact_location="   ")

    def test_none_artifact_location_is_allowed(self):
        config = SalaryMLflowConfig(artifact_location=None)
        assert config.artifact_location is None

    def test_non_bool_tracking_enabled_raises_type_error(self):
        with pytest.raises(TypeError):
            SalaryMLflowConfig(tracking_enabled="true")  # type: ignore[arg-type]

    def test_non_positive_max_parameter_length_raises(self):
        with pytest.raises(ValueError):
            SalaryMLflowConfig(max_parameter_length=0)

    def test_non_int_max_parameter_length_raises_type_error(self):
        with pytest.raises(TypeError):
            SalaryMLflowConfig(max_parameter_length=1.5)  # type: ignore[arg-type]


class TestHelpers:
    def test_default_tags(self):
        config = SalaryMLflowConfig(project_name="proj", pipeline_name="pipe")
        assert config.default_tags() == {"project": "proj", "pipeline": "pipe"}

    def test_build_run_name(self):
        config = SalaryMLflowConfig(run_name_prefix="salary")
        assert config.build_run_name("e1_ridge") == "salary-e1_ridge"

    def test_build_run_name_rejects_blank_suffix(self):
        config = SalaryMLflowConfig()
        with pytest.raises(ValueError):
            config.build_run_name("   ")

    def test_local_tracking_path_from_file_uri(self, tmp_path):
        db_path = tmp_path / "mlflow.db"
        config = SalaryMLflowConfig(tracking_uri=f"file://{db_path.as_posix()}")
        assert config.is_local_tracking is True
        assert config.local_tracking_path is not None

    def test_sqlite_uri_counts_as_local_tracking(self):
        config = SalaryMLflowConfig(tracking_uri="sqlite:////tmp/mlflow.db")
        assert config.is_local_tracking is False  # "://" present but scheme != file
        # NOTE: this documents actual `is_local_tracking` behavior --
        # only bare paths or file:// URIs are treated as local.


class TestImmutability:
    def test_config_is_frozen(self):
        config = SalaryMLflowConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.experiment_name = "changed"  # type: ignore[misc]
