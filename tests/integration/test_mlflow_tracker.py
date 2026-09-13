"""
tests/integration/test_mlflow_tracker.py

Integration tests for
src/components/salary_predict/salary_mlflow_tracker.py
(SalaryMLflowTracker).

Every test points at an isolated, temporary SQLite-backed MLflow store
under `tmp_path` (both the tracking backend AND the artifact root, via
`SalaryMLflowConfig.artifact_location`), so nothing here ever touches
the project's real mlflow.db or a real MLflow server. Persisted state is
verified through `MlflowClient` directly, not just through the
tracker's return values.
"""

from __future__ import annotations

import math

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge

from src.components.salary_predict.salary_mlflow_tracker import (
    SalaryMLflowTracker,
    SalaryMLflowRunInfo,
)
from src.configs.salary_predict.salary_ML_flow_config import SalaryMLflowConfig
from src.configs.salary_predict.salary_experiment_config import build_e1_config
from src.components.salary_predict.salary_single_experiment_runner import SalaryTrainingResult
from src.exception import CustomException

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_config(tmp_path) -> SalaryMLflowConfig:
    db_path = tmp_path / "mlflow.db"
    artifact_root = tmp_path / "mlruns"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return SalaryMLflowConfig(
        tracking_uri=f"sqlite:///{db_path.as_posix()}",
        # Path.as_uri() (not a hand-built "file://" + as_posix() string)
        # is required for a correct file URI on Windows, where it must
        # come out as "file:///C:/..." (three slashes before the drive
        # letter), not "file://C:/...".
        artifact_location=artifact_root.as_uri(),
        experiment_name="salary_test_experiment",
        registered_model_name="salary_test_model",
        tracking_enabled=True,
    )


@pytest.fixture
def tracker(mlflow_config) -> SalaryMLflowTracker:
    return SalaryMLflowTracker(mlflow_config)


@pytest.fixture
def client(mlflow_config) -> MlflowClient:
    return MlflowClient(tracking_uri=mlflow_config.tracking_uri)


def _make_training_result(experiment_id: str = "E1") -> SalaryTrainingResult:
    return SalaryTrainingResult(
        experiment_id=experiment_id,
        experiment_name="structured_ridge",
        model_name="ridge",
        config_signature="test-signature",
        train_row_count=40,
        validation_row_count=15,
        raw_feature_columns=("formatted_experience_level", "company_state"),
        raw_feature_count=2,
        transformed_feature_count=12,
        training_seconds=0.05,
        validation_prediction_seconds=0.01,
        log_metrics={"mae": 0.15, "rmse": 0.22, "r2": 0.81},
        annual_metrics={"mae": 9500.0, "rmse": 14200.0, "r2": 0.77, "median_ape": 8.4},
        fitted_workflow=None,
    )


# ======================================================================
# track_training_result -- experiment/run creation
# ======================================================================


class TestTrackTrainingResult:
    def test_creates_experiment_and_finished_run(self, tracker, client, mlflow_config):
        experiment_config = build_e1_config()
        training_result = _make_training_result(experiment_id=experiment_config.experiment_id)

        run_info = tracker.track_training_result(experiment_config, training_result)

        assert isinstance(run_info, SalaryMLflowRunInfo)
        assert run_info.status == "FINISHED"
        assert run_info.run_id is not None

        experiment = client.get_experiment_by_name(mlflow_config.experiment_name)
        assert experiment is not None

        run = client.get_run(run_info.run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags.get("experiment_id") == experiment_config.experiment_id
        assert run.data.tags.get("model_family") == experiment_config.model_name
        assert run.data.params.get("experiment_id") == experiment_config.experiment_id
        assert run.data.metrics.get("validation.log_mae") == pytest.approx(0.15)
        assert run.data.metrics.get("validation.annual_r2") == pytest.approx(0.77)
        assert run.data.metrics.get("data.train_row_count") == pytest.approx(40)

    def test_repeated_calls_reuse_the_same_mlflow_experiment(self, tracker, client, mlflow_config):
        experiment_config = build_e1_config()

        first_run = tracker.track_training_result(
            experiment_config, _make_training_result(experiment_config.experiment_id)
        )
        second_run = tracker.track_training_result(
            experiment_config, _make_training_result(experiment_config.experiment_id)
        )

        assert first_run.mlflow_experiment_id == second_run.mlflow_experiment_id
        assert first_run.run_id != second_run.run_id

    def test_experiment_id_mismatch_raises_custom_exception(self, tracker):
        experiment_config = build_e1_config()
        mismatched_result = _make_training_result(experiment_id="E2")  # config is E1

        with pytest.raises(CustomException):
            tracker.track_training_result(experiment_config, mismatched_result)

    def test_invalid_argument_types_raise_custom_exception(self, tracker):
        experiment_config = build_e1_config()
        with pytest.raises(CustomException):
            tracker.track_training_result(experiment_config, {"not": "a result"})  # type: ignore[arg-type]

    def test_tracking_disabled_returns_disabled_status_without_creating_a_run(
        self, tmp_path, client
    ):
        config = SalaryMLflowConfig(
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow_disabled.db').as_posix()}",
            experiment_name="disabled_experiment",
            tracking_enabled=False,
        )
        tracker = SalaryMLflowTracker(config)
        experiment_config = build_e1_config()

        run_info = tracker.track_training_result(
            experiment_config, _make_training_result(experiment_config.experiment_id)
        )

        assert run_info.status == "DISABLED"
        assert run_info.run_id is None
        assert run_info.mlflow_experiment_id is None


# ======================================================================
# start_run context manager
# ======================================================================


class TestStartRun:
    def test_creates_and_finishes_a_run(self, tracker, client, mlflow_config):
        with tracker.start_run(run_name="manual_run") as run:
            assert run is not None
            run_id = run.info.run_id
            tracker.log_params({"alpha": 1.0})
            tracker.log_metrics({"rmse": 0.5})
            tracker.log_tags({"stage": "manual_test"})

        persisted_run = client.get_run(run_id)
        assert persisted_run.info.status == "FINISHED"
        assert persisted_run.data.params.get("alpha") == "1.0"
        assert persisted_run.data.metrics.get("rmse") == pytest.approx(0.5)
        assert persisted_run.data.tags.get("stage") == "manual_test"

    def test_disabled_tracking_yields_none(self, tmp_path):
        config = SalaryMLflowConfig(
            tracking_uri=f"sqlite:///{(tmp_path / 'disabled.db').as_posix()}",
            tracking_enabled=False,
        )
        tracker = SalaryMLflowTracker(config)
        with tracker.start_run(run_name="anything") as run:
            assert run is None


# ======================================================================
# Metric / param / tag logging edge cases
# ======================================================================


class TestLoggingEdgeCases:
    def test_non_finite_and_none_metrics_are_filtered_out(self, tracker, client):
        with tracker.start_run(run_name="metrics_run") as run:
            run_id = run.info.run_id
            tracker.log_metrics({"good": 1.0, "bad_nan": float("nan"), "bad_inf": float("inf"), "skip": None})

        persisted = client.get_run(run_id)
        assert persisted.data.metrics.get("good") == pytest.approx(1.0)
        assert "bad_nan" not in persisted.data.metrics
        assert "bad_inf" not in persisted.data.metrics
        assert "skip" not in persisted.data.metrics

    def test_logging_without_an_active_run_is_a_safe_no_op(self, tracker):
        # No exception should be raised even though no run is active.
        tracker.log_params({"x": 1})
        tracker.log_metrics({"y": 1.0})
        tracker.log_tags({"z": "v"})

    def test_log_artifact_missing_file_raises_file_not_found(self, tracker, tmp_path):
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(tmp_path / "does_not_exist.txt")

    def test_log_artifact_and_log_artifacts_persist_files(self, tracker, client, tmp_path):
        single_file = tmp_path / "note.txt"
        single_file.write_text("hello")

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "a.json").write_text("{}")

        with tracker.start_run(run_name="artifact_run") as run:
            run_id = run.info.run_id
            tracker.log_artifact(single_file, artifact_folder="misc")
            tracker.log_artifacts(bundle_dir, artifact_folder="bundle")

        artifact_paths = {a.path for a in client.list_artifacts(run_id)}
        assert "misc" in artifact_paths
        assert "bundle" in artifact_paths


# ======================================================================
# Final model logging / registry interaction
# ======================================================================


class TestLogFinalModel:
    def test_returns_a_loadable_model_uri_without_registering(self, tracker, mlflow_config):
        model = Ridge(alpha=1.0).fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])

        with tracker.start_run(run_name="final_model_run"):
            model_uri = tracker.log_final_model(model, register_model=False)

        assert model_uri is not None
        # Don't assert a specific URI scheme prefix ("runs:/" vs
        # "models:/" etc. can vary by MLflow version/config) -- the
        # meaningful check is that the URI is genuinely loadable.
        assert isinstance(model_uri, str) and len(model_uri) > 0

        loaded = mlflow.sklearn.load_model(model_uri)
        # Ridge(alpha=1.0) regularizes toward zero, so an exact-fit
        # assertion here would be testing Ridge's math, not the
        # save/load round-trip. The actual intent is just: does the
        # loaded model produce a finite, sane prediction.
        prediction = loaded.predict([[4.0]])[0]
        assert math.isfinite(prediction)
        assert 0.0 < prediction < 10.0

    def test_registers_model_when_requested(self, tracker, client, mlflow_config):
        model = Ridge(alpha=1.0).fit([[1.0], [2.0]], [1.0, 2.0])
        registered_name = "salary_test_model_registered"

        with tracker.start_run(run_name="register_run"):
            tracker.log_final_model(model, registered_model_name=registered_name, register_model=True)

        versions = client.search_model_versions(f"name='{registered_name}'")
        assert len(versions) == 1

        latest = tracker.get_latest_model_version(registered_model_name=registered_name)
        # This MLflow version returns `.version` inconsistently typed
        # across call paths (int in some, str in others) -- compare as
        # strings so the test doesn't depend on which one it is.
        assert str(latest) == str(versions[0].version)

    def test_none_workflow_raises_value_error(self, tracker):
        with pytest.raises(ValueError):
            tracker.log_final_model(None)

    def test_get_latest_model_version_returns_none_when_unregistered(self, tracker):
        assert tracker.get_latest_model_version(registered_model_name="never_registered_model") is None

    def test_get_latest_model_version_rejects_blank_name(self, tracker):
        with pytest.raises(ValueError):
            tracker.get_latest_model_version(registered_model_name="   ")

    def test_disabled_tracking_skips_model_logging(self, tmp_path):
        config = SalaryMLflowConfig(
            tracking_uri=f"sqlite:///{(tmp_path / 'disabled_model.db').as_posix()}",
            tracking_enabled=False,
        )
        tracker = SalaryMLflowTracker(config)
        model = Ridge().fit([[1.0]], [1.0])
        assert tracker.log_final_model(model) is None
        assert tracker.get_latest_model_version() is None