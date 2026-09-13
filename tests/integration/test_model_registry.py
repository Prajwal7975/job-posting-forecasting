"""
tests/integration/test_model_registry.py

Integration tests for
src/components/salary_predict/salary_model_registry.py
(SalaryModelRegistry).

Every test points at an isolated, temporary SQLite-backed MLflow store
under `tmp_path`, never the project's real mlflow.db or model registry.
Persisted state (model versions, tags, aliases) is verified through
`MlflowClient` directly.
"""

from __future__ import annotations

import uuid

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge

from src.components.salary_model_registry import SalaryModelRegistry
from src.configs.salary_model_registry_config import SalaryModelRegistryConfig
from src.entity.salary_model_registry_entity import SalaryModelRegistryResult

pytestmark = pytest.mark.integration


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    db_path = tmp_path / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


@pytest.fixture
def mlflow_run(tracking_uri, tmp_path):
    """
    Logs a tiny, real sklearn model to a real MLflow run and yields its
    (run_id, model_uri) -- SalaryModelRegistry.register() expects an
    already-logged model URI, exactly as SalaryMLflowTracker.log_final_model
    would hand it.
    """
    mlflow.set_tracking_uri(tracking_uri)
    artifact_root = tmp_path / "mlruns"
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_name = f"registry_test_{uuid.uuid4().hex[:8]}"
    # Path.as_uri() gives a correct file:// URI on both Windows and POSIX;
    # a hand-built "file://" + as_posix() string is malformed on Windows.
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=artifact_root.as_uri()
    )

    model = Ridge(alpha=1.0).fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model_info = mlflow.sklearn.log_model(model, name="model")

    return run.info.run_id, model_info.model_uri


@pytest.fixture
def registry_config(tracking_uri) -> SalaryModelRegistryConfig:
    return SalaryModelRegistryConfig(
        registered_model_name=f"salary_test_model_{uuid.uuid4().hex[:8]}",
        tracking_uri=tracking_uri,
        production_alias="production",
        allow_existing_model=True,
    )


@pytest.fixture
def registry(registry_config) -> SalaryModelRegistry:
    return SalaryModelRegistry(registry_config)


@pytest.fixture
def client(tracking_uri) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


# ======================================================================
# Successful registration + promotion
# ======================================================================


class TestSuccessfulRegistration:
    def test_first_registration_creates_model_version_and_promotes(
        self, registry, registry_config, client, mlflow_run
    ):
        run_id, model_uri = mlflow_run

        result = registry.register(
            model_uri=model_uri,
            source_run_id=run_id,
            validation_passed=True,
            validation_metrics={"rmse": 12000.0},
            test_metrics={"rmse": 12500.0},
            promote_to_production=True,
        )

        assert isinstance(result, SalaryModelRegistryResult)
        assert result.success is True
        assert str(result.model_version) == "1"
        assert result.alias_updated is True
        assert str(result.promoted_model_version) == "1"
        assert result.promotion_approved is True
        assert result.previous_production_version is None  # first-ever registration

        alias_version = client.get_model_version_by_alias(
            registry_config.registered_model_name, registry_config.production_alias
        )
        assert str(alias_version.version) == "1"

    def test_tags_persist_lineage_and_metrics(self, registry, registry_config, client, mlflow_run):
        run_id, model_uri = mlflow_run
        registry.register(
            model_uri=model_uri,
            source_run_id=run_id,
            validation_passed=True,
            validation_metrics={"rmse": 12000.0},
            test_metrics={"rmse": 12500.0},
            metadata={"feature_experiment_id": "E1"},
            promote_to_production=True,
        )

        version = client.get_model_version(registry_config.registered_model_name, "1")
        assert version.tags.get("source_run_id") == run_id
        assert version.tags.get("validation_passed") == "true"
        assert version.tags.get("validation_rmse") == "12000.0"
        assert version.tags.get("test_rmse") == "12500.0"
        assert version.tags.get("feature_experiment_id") == "E1"

    def test_second_registration_reports_and_replaces_previous_production_version(
        self, registry, registry_config, client, tracking_uri, tmp_path
    ):
        mlflow.set_tracking_uri(tracking_uri)
        artifact_root = tmp_path / "mlruns2"
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            f"exp_{uuid.uuid4().hex[:8]}", artifact_location=artifact_root.as_uri()
        )

        def _log_new_model() -> tuple[str, str]:
            model = Ridge(alpha=1.0).fit([[1.0], [2.0]], [1.0, 2.0])
            with mlflow.start_run(experiment_id=experiment_id) as run:
                info = mlflow.sklearn.log_model(model, name="model")
            return run.info.run_id, info.model_uri

        run_id_1, uri_1 = _log_new_model()
        first_result = registry.register(
            model_uri=uri_1, source_run_id=run_id_1, validation_passed=True, promote_to_production=True
        )
        assert str(first_result.model_version) == "1"

        run_id_2, uri_2 = _log_new_model()
        second_result = registry.register(
            model_uri=uri_2, source_run_id=run_id_2, validation_passed=True, promote_to_production=True
        )

        assert str(second_result.model_version) == "2"
        assert str(second_result.previous_production_version) == "1"
        assert str(second_result.promoted_model_version) == "2"

        alias_version = client.get_model_version_by_alias(
            registry_config.registered_model_name, registry_config.production_alias
        )
        assert str(alias_version.version) == "2"

    def test_registration_without_promotion_does_not_assign_alias(
        self, registry, registry_config, client, mlflow_run
    ):
        run_id, model_uri = mlflow_run
        result = registry.register(
            model_uri=model_uri,
            source_run_id=run_id,
            validation_passed=True,
            promote_to_production=False,
        )

        assert result.success is True
        assert result.alias_updated is False
        assert result.promoted_model_version is None
        assert result.promotion_approved is False
        assert result.production_alias is None

        with pytest.raises(MlflowException):
            client.get_model_version_by_alias(
                registry_config.registered_model_name, registry_config.production_alias
            )

    def test_registered_model_is_reused_not_recreated(self, registry, mlflow_run, client, registry_config):
        run_id, model_uri = mlflow_run
        registry.register(model_uri=model_uri, source_run_id=run_id, validation_passed=True)
        # A second registry instance targeting the SAME already-existing
        # registered model must not fail on "model already exists".
        second_registry = SalaryModelRegistry(registry_config)
        result = second_registry.register(
            model_uri=model_uri, source_run_id=run_id, validation_passed=True
        )
        assert result.success is True
        assert str(result.model_version) == "2"


# ======================================================================
# Validation gate (raises before any MLflow call)
# ======================================================================


class TestValidationGate:
    def test_validation_not_passed_raises_value_error(self, registry, mlflow_run):
        _, model_uri = mlflow_run
        with pytest.raises(ValueError):
            registry.register(model_uri=model_uri, validation_passed=False)

    def test_missing_model_uri_raises_value_error(self, registry):
        with pytest.raises(ValueError):
            registry.register(model_uri="", validation_passed=True)

    def test_failed_validation_gate_creates_no_model_version(
        self, registry, registry_config, client, mlflow_run
    ):
        _, model_uri = mlflow_run
        with pytest.raises(ValueError):
            registry.register(model_uri=model_uri, validation_passed=False)

        with pytest.raises(MlflowException):
            client.get_registered_model(registry_config.registered_model_name)


# ======================================================================
# Failure handling (caught internally, returned as a failed result)
# ======================================================================


class TestFailureHandling:
    def test_mlflow_client_failure_returns_failed_result_without_raising(
        self, registry, mlflow_run, monkeypatch
    ):
        # Empirically, this MLflow version doesn't validate `source`
        # existence at registration time -- neither a fake "runs:/..."
        # URI nor a nonexistent local path caused create_model_version to
        # fail. Force the failure deterministically instead, at the
        # actual MLflow client call the registry makes internally.
        run_id, model_uri = mlflow_run

        def _raise(*_args, **_kwargs):
            raise RuntimeError("simulated MLflow client failure")

        monkeypatch.setattr(registry.client, "create_model_version", _raise)

        result = registry.register(model_uri=model_uri, source_run_id=run_id, validation_passed=True)

        assert result.success is False
        assert result.error is not None
        assert "simulated MLflow client failure" in result.error
        assert result.model_version is None

    def test_allow_existing_model_false_returns_failed_result_for_new_model(
        self, tracking_uri, mlflow_run
    ):
        run_id, model_uri = mlflow_run
        config = SalaryModelRegistryConfig(
            registered_model_name=f"brand_new_model_{uuid.uuid4().hex[:8]}",
            tracking_uri=tracking_uri,
            allow_existing_model=False,
        )
        registry = SalaryModelRegistry(config)

        result = registry.register(model_uri=model_uri, source_run_id=run_id, validation_passed=True)

        assert result.success is False
        assert "does not exist" in result.error
