"""
tests/integration/test_salary_pipeline.py

Integration tests for
src/components/salary_predict/salary_model_pipeline_orchestrator.py
(SalaryModelPipelineOrchestrator).

Strategy
--------
SalaryModelPipelineOrchestrator.__init__ accepts explicit injection for
every heavy downstream stage: feature_runner, model_family_runner,
tuning_runner, final_model_trainer, mlflow_tracker, and model_registry.
Those stages already have (or will have) their own dedicated unit/
integration coverage elsewhere -- re-verifying their internals here
would duplicate that coverage and make this test slow and brittle.

Per the project's own testing brief ("where the architecture allows
individual stages to be mocked/stubbed, keep the integration test
lightweight while still verifying orchestration order and stage
interactions"), this file mocks exactly those six injectable
dependencies and lets the REAL, cheap parts of the orchestrator run
unmocked:

    - the real SalaryFeatureEngineeringConfig/file-existence check
    - the real SalaryPreprocessorBuilder (Stage 4 transformation)
    - the real joblib load/verify/promote logic (Stage 9/10)
    - the real SalaryModelPipelineResult assembly

`self.feature_engineer` and `self.splitter` are not constructor
parameters (by the real orchestrator's own design), so they are
replaced as plain instance attributes after construction -- which is
safe because Python attribute assignment doesn't care how an object
was built.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.pipelines.salary_model_pipeline_orchestrator import (
    SalaryModelPipelineOrchestrator,
    SalaryModelPipelineResult,
)
from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)
from src.configs.salary_predict.salary_experiment_config import build_e1_config
from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)
from src.configs.salary_model_registry_config import SalaryModelRegistryConfig
from src.configs.salary_predict.salary_ML_flow_config import SalaryMLflowConfig
from src.entity.salary_model_registry_entity import SalaryModelRegistryResult
from src.exception import CustomException

pytestmark = pytest.mark.integration


FEATURE_COLUMNS = [
    "formatted_experience_level",
    "company_state",
    "company_country",
    "top_industry",
    "skill_count",
]


def _make_split_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual = rng.uniform(50_000, 200_000, size=n)
    return pd.DataFrame(
        {
            "formatted_experience_level": rng.choice(
                ["Entry level", "Associate", "Mid-Senior level"], size=n
            ),
            "company_state": rng.choice(["CA", "NY", "TX"], size=n),
            "company_country": ["US"] * n,
            "top_industry": rng.choice(["Tech", "Finance"], size=n),
            "skill_count": rng.integers(0, 5, size=n).astype(float),
            "target_log_salary": np.log1p(annual),
        }
    )


@pytest.fixture
def split_frames():
    return (
        _make_split_frame(30, seed=1),
        _make_split_frame(10, seed=2),
        _make_split_frame(10, seed=3),
    )


def _fake_final_model_result(tmp_path, train_df, artifact_dir_name, validation_passed=True):
    """
    Builds a REAL fitted preprocessor+model pipeline (mirroring what
    SalaryFinalModelTrainer actually produces) and persists it to disk,
    so Stage 9/10's real joblib.load()/predict()/verify logic has a
    genuine artifact to work with instead of a mock standing in for it.
    """
    feature_config = build_e1_config()
    preprocessor = SalaryPreprocessorBuilder().build(feature_config)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["target_log_salary"])

    artifact_dir = tmp_path / artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "final_model.joblib"
    joblib.dump(pipeline, model_path)

    return SimpleNamespace(
        success=True,
        error=None,
        model_name="ridge",
        model_class_name="Ridge",
        feature_experiment_id="E1",
        model_experiment_id="M1",
        model_artifact_path=str(model_path),
        artifact_directory=str(artifact_dir),
        validation_metrics={"RMSE": 0.35, "MAE": 0.28},
        validation_metric="RMSE",
        validation_passed=validation_passed,
        registered_model_uri="runs:/fake-run-id/final_model",
        mlflow_run_id="fake-run-id",
    )


@pytest.fixture
def orchestrator_factory(tmp_path, split_frames):
    """Returns a function that builds a fully-wired orchestrator with
    every heavy stage mocked, plus the split parquet files it needs."""

    train_df, validation_df, test_df = split_frames

    def _build(validation_passed: bool = True, feature_runner_error: Exception | None = None):
        feature_engineering_config = SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)
        # The orchestrator checks feature_store_path.exists() itself,
        # BEFORE ever calling feature_engineer -- so this file must exist
        # regardless of feature_engineer being mocked below. Its content
        # is irrelevant since the mocked feature_engineer never reads it.
        feature_engineering_config.feature_store_path.parent.mkdir(parents=True, exist_ok=True)
        feature_engineering_config.feature_store_path.write_text("placeholder")

        train_path = tmp_path / "train.parquet"
        validation_path = tmp_path / "validation.parquet"
        test_path = tmp_path / "test.parquet"
        train_df.to_parquet(train_path)
        validation_df.to_parquet(validation_path)
        test_df.to_parquet(test_path)

        mock_feature_runner = MagicMock()
        if feature_runner_error is not None:
            mock_feature_runner.run.side_effect = feature_runner_error
        else:
            mock_feature_runner.run.return_value = SimpleNamespace(
                winner_config=build_e1_config(),
                best_experiment_id="E1",
            )

        mock_model_family_runner = MagicMock()
        mock_model_family_runner.run_experiments.return_value = SimpleNamespace(
            winner_model_name="ridge",
            winner_experiment_id="M1",
            winner_config_signature="model-sig",
        )

        mock_tuning_runner = MagicMock()
        mock_tuning_runner.run_tuning.return_value = SimpleNamespace(
            model_name="ridge",
            preferred_params={"alpha": 1.0},
            tuning_improved_baseline=False,
            best_config_signature="tune-sig",
            preferred_config_signature="tune-sig-preferred",
        )

        mock_final_model_trainer = MagicMock()
        mock_final_model_trainer.run.return_value = _fake_final_model_result(
            tmp_path, train_df, "final_model_artifacts", validation_passed=validation_passed
        )

        mock_model_registry = MagicMock()
        mock_model_registry.register.return_value = SalaryModelRegistryResult(
            success=True,
            registered_model_name="salary_prediction_model",
            model_version="1",
            model_uri="models:/salary_prediction_model/1",
            source_run_id="fake-run-id",
            production_alias="production",
            alias_updated=True,
            promoted_model_version="1",
            promotion_approved=True,
            validation_passed=True,
        )

        mlflow_tracker = MagicMock()
        mlflow_tracker.config = SalaryMLflowConfig(
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db", tracking_enabled=True
        )

        orchestrator = SalaryModelPipelineOrchestrator(
            feature_engineering_config=feature_engineering_config,
            model_registry_config=SalaryModelRegistryConfig(tracking_uri=None),
            feature_runner=mock_feature_runner,
            model_family_runner=mock_model_family_runner,
            tuning_runner=mock_tuning_runner,
            final_model_trainer=mock_final_model_trainer,
            mlflow_tracker=mlflow_tracker,
            model_registry=mock_model_registry,
        )

        # `feature_engineer` and `splitter` are not constructor parameters
        # on the real class (only the six heavy ML stages are injectable)
        # -- replace them as plain instance attributes instead. Their
        # real behavior is already covered by
        # test_salary_feature_engineering.py; re-running the real
        # component here would just require a fully valid raw feature
        # store, which the mocked splitter never actually reads anyway.
        dummy_feature_store_output = tmp_path / "salary_modeling_dataset.parquet"
        train_df.to_parquet(dummy_feature_store_output)  # content is irrelevant; only existence matters

        mock_feature_engineer = MagicMock()
        mock_feature_engineer.initiate_salary_feature_engineering.return_value = SimpleNamespace(
            salary_modeling_dataset_path=str(dummy_feature_store_output),
            metadata_path=None,
            report_path=None,
            schema_fingerprint_path=None,
            summary=None,
        )
        orchestrator.feature_engineer = mock_feature_engineer

        mock_splitter = MagicMock()
        mock_splitter.initiate_dataset_splitting.return_value = SimpleNamespace(
            train_dataset_path=str(train_path),
            validation_dataset_path=str(validation_path),
            test_dataset_path=str(test_path),
            status="EXECUTED",
        )
        orchestrator.splitter = mock_splitter

        return orchestrator, mock_model_registry, mock_final_model_trainer

    return _build


# ======================================================================
# Successful end-to-end orchestration
# ======================================================================


class TestSuccessfulPipeline:
    def test_successful_run_returns_success_result(self, orchestrator_factory):
        orchestrator, model_registry, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        assert isinstance(result, SalaryModelPipelineResult)
        assert result.success is True
        assert result.feature_experiment_id == "E1"
        assert result.model_name == "ridge"
        assert result.validation_passed is True

    def test_final_model_artifact_exists_after_promotion(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        assert result.final_model_path is not None
        from pathlib import Path

        assert Path(result.final_model_path).exists()
        assert Path(result.final_model_path).name == "model.joblib"

    def test_validation_and_test_metrics_are_populated(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        assert result.validation_metrics
        assert result.test_metrics
        assert set(result.test_metrics.keys()) == {"MAE", "RMSE", "R2"}

    def test_exactly_one_registry_version_created_and_promoted(self, orchestrator_factory):
        orchestrator, model_registry, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        model_registry.register.assert_called_once()
        assert result.registered_model_uri == "models:/salary_prediction_model/1"
        assert result.registered_model_version == "1"

    def test_production_alias_points_to_new_version_on_success(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        assert result.production_alias == "production"
        assert result.alias_updated is True

    def test_stage_times_recorded_for_every_stage(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        expected_stages = {
            "salary_feature_engineering",
            "dataset_splitting",
            "feature_experiments",
            "winning_feature_preparation",
            "model_family_comparison",
            "hyperparameter_tuning",
            "final_model_training",
            "test_evaluation",
            "model_promotion",
            "model_registry_promotion",
        }
        assert expected_stages.issubset(result.stage_times.keys())
        assert result.total_execution_seconds >= 0

    def test_pipeline_summary_json_is_persisted(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=True)
        result = orchestrator.run(force_rebuild=True)

        from pathlib import Path

        summary_path = Path(result.final_model_artifact_directory) / "pipeline_summary.json"
        assert summary_path.exists()
        saved = json.loads(summary_path.read_text())
        assert saved["success"] is True


# ======================================================================
# Failed validation gate -- no promotion
# ======================================================================


class TestFailedValidationGate:
    def test_failed_validation_returns_unsuccessful_result(self, orchestrator_factory):
        orchestrator, model_registry, _ = orchestrator_factory(validation_passed=False)
        result = orchestrator.run(force_rebuild=True)

        assert result.success is False
        assert "validation quality gate" in result.error.lower()

    def test_failed_validation_never_calls_model_registry(self, orchestrator_factory):
        orchestrator, model_registry, _ = orchestrator_factory(validation_passed=False)
        orchestrator.run(force_rebuild=True)

        model_registry.register.assert_not_called()

    def test_failed_validation_has_no_test_metrics(self, orchestrator_factory):
        orchestrator, _, _ = orchestrator_factory(validation_passed=False)
        result = orchestrator.run(force_rebuild=True)

        # Test evaluation is Stage 9, which never runs when Stage 8 fails.
        assert result.test_metrics is None
        assert result.registered_model_version is None


# ======================================================================
# Required-stage failure
# ======================================================================


class TestRequiredStageFailure:
    def test_feature_runner_failure_propagates_as_custom_exception(self, orchestrator_factory):
        orchestrator, model_registry, _ = orchestrator_factory(
            feature_runner_error=RuntimeError("feature experiment runner exploded")
        )

        with pytest.raises(CustomException):
            orchestrator.run(force_rebuild=True)

        model_registry.register.assert_not_called()

    def test_missing_feature_store_raises_custom_exception(self, tmp_path, split_frames):
        # feature_store_path was never created -> Stage 1 must fail fast,
        # before any injected mock is even touched.
        feature_engineering_config = SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)
        orchestrator = SalaryModelPipelineOrchestrator(
            feature_engineering_config=feature_engineering_config,
            model_registry_config=SalaryModelRegistryConfig(tracking_uri=None),
            feature_runner=MagicMock(),
            model_family_runner=MagicMock(),
            tuning_runner=MagicMock(),
            final_model_trainer=MagicMock(),
            mlflow_tracker=MagicMock(config=SalaryMLflowConfig(tracking_enabled=False)),
            model_registry=MagicMock(),
        )
        with pytest.raises(CustomException):
            orchestrator.run(force_rebuild=False)


# ======================================================================
# Constructor-level configuration validation
# ======================================================================


class TestConstructorValidation:
    def test_mismatched_tracking_uris_raise_in_production_environment(self, tmp_path):
        mlflow_tracker = MagicMock(
            config=SalaryMLflowConfig(tracking_uri=f"sqlite:///{tmp_path}/a.db")
        )
        mismatched_registry_config = SalaryModelRegistryConfig(
            tracking_uri=f"sqlite:///{tmp_path}/b.db", environment="production"
        )

        with pytest.raises(RuntimeError):
            SalaryModelPipelineOrchestrator(
                mlflow_tracker=mlflow_tracker,
                model_registry_config=mismatched_registry_config,
                model_registry=MagicMock(),
                feature_runner=MagicMock(),
                model_family_runner=MagicMock(),
                tuning_runner=MagicMock(),
                final_model_trainer=MagicMock(),
            )

    def test_mismatched_tracking_uris_only_warn_outside_production(self, tmp_path):
        mlflow_tracker = MagicMock(
            config=SalaryMLflowConfig(tracking_uri=f"sqlite:///{tmp_path}/a.db")
        )
        mismatched_registry_config = SalaryModelRegistryConfig(
            tracking_uri=f"sqlite:///{tmp_path}/b.db", environment="local"
        )

        # Must NOT raise -- local/dev environments only get a warning.
        SalaryModelPipelineOrchestrator(
            mlflow_tracker=mlflow_tracker,
            model_registry_config=mismatched_registry_config,
            model_registry=MagicMock(),
            feature_runner=MagicMock(),
            model_family_runner=MagicMock(),
            tuning_runner=MagicMock(),
            final_model_trainer=MagicMock(),
        )
