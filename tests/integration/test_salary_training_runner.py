"""
tests/integration/test_salary_training_runner.py

Integration tests for
src/components/salary_predict/salary_training_runner.py
(SalaryTrainingRunner).

These tests write small, deterministic train/validation parquet files to
tmp_path and run the real training workflow end-to-end (preprocessing +
model fit + validation prediction + metrics). No MLflow tracking is
involved here -- SalaryTrainingRunner itself never touches MLflow, per
its own docstring ("Out of scope: ... log to MLflow").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.components.salary_predict.salary_single_experiment_runner import (
    SalaryTrainingRunner,
    SalaryTrainingResult,
)
from src.configs.salary_predict.salary_experiment_config import (
    build_e0_config,
    build_e1_config,
    build_e2_config,
    build_e3a_config,
    build_e3b_config,
)
from src.exception import CustomException

pytestmark = pytest.mark.integration


TITLE_CHOICES = (
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
    "Machine Learning Engineer",
    "Software Engineer",
)


SKILL_LIST_CHOICES = (
    "python|sql",
    "python|aws",
    "sql|docker",
    "aws|docker",
    "python|docker",
)


def _make_split_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    experience_levels = ["Entry level", "Associate", "Mid-Senior level", "Director"]
    states = ["CA", "NY", "TX", "WA"]
    industries = ["Tech", "Finance", "Retail"]

    annual_salary = rng.uniform(50_000, 200_000, size=n)
    return pd.DataFrame(
        {
            # Repeated, overlapping words ("Data", "Engineer") so
            # TfidfVectorizer's min_df/max_df thresholds both have terms
            # to work with -- unique-per-row titles either fall below
            # min_df (rare tokens) or above max_df (constant tokens).
            "title": rng.choice(TITLE_CHOICES, size=n),
            "formatted_experience_level": rng.choice(experience_levels, size=n),
            "company_state": rng.choice(states, size=n),
            "company_country": ["US"] * n,
            "top_industry": rng.choice(industries, size=n),
            "skill_count": rng.integers(0, 6, size=n).astype(float),
            # Same overlap reasoning as `title`, for E3A's skill TF-IDF
            # and E3B's skill multi-hot encoding.
            "skill_list": rng.choice(SKILL_LIST_CHOICES, size=n),
            "target_annual_salary": annual_salary,
            "target_log_salary": np.log1p(annual_salary),
        }
    )


@pytest.fixture
def split_paths(tmp_path):
    train_df = _make_split_df(40, seed=1)
    validation_df = _make_split_df(15, seed=2)

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    train_df.to_parquet(train_path)
    validation_df.to_parquet(validation_path)

    return train_path, validation_path, train_df, validation_df


@pytest.fixture
def runner(split_paths) -> SalaryTrainingRunner:
    train_path, validation_path, _, _ = split_paths
    return SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)


# ======================================================================
# Successful runs
# ======================================================================


class TestSuccessfulRuns:
    def test_e1_structured_experiment_produces_complete_result(self, runner, split_paths):
        _, _, train_df, validation_df = split_paths
        result = runner.run(build_e1_config())

        assert isinstance(result, SalaryTrainingResult)
        assert result.experiment_id == "E1"
        assert result.model_name == "ridge"
        assert result.train_row_count == len(train_df)
        assert result.validation_row_count == len(validation_df)
        assert result.raw_feature_count == 5  # 4 categorical + 1 numeric
        assert result.transformed_feature_count is not None
        assert result.transformed_feature_count > 0
        assert result.training_seconds >= 0
        assert result.validation_prediction_seconds >= 0
        assert set(result.log_metrics.keys()) == {"mae", "rmse", "r2"}
        assert set(result.annual_metrics.keys()) == {"mae", "rmse", "r2", "median_ape"}
        assert hasattr(result.fitted_workflow, "predict")

    def test_e0_dummy_baseline_has_no_preprocessor_or_features(self, runner):
        result = runner.run(build_e0_config())

        assert result.raw_feature_count == 0
        assert result.raw_feature_columns == ()
        assert result.transformed_feature_count is None
        # No preprocessor step -- fitted_workflow IS the bare estimator.
        assert type(result.fitted_workflow).__name__ == "DummyRegressor"

    def test_e2_title_tfidf_experiment_runs_successfully(self, runner):
        result = runner.run(build_e2_config())
        assert result.raw_feature_count == 6  # 5 (E1) + title
        assert "title" in result.raw_feature_columns
        assert result.transformed_feature_count is not None

    def test_e3a_skill_tfidf_experiment_runs_successfully(self, runner):
        # Ported from the original test_salary_feature_exp_runner_smoke.py
        # (which also covered E3A/E3B, but against real production
        # artifacts/salary_dataset_splits/latest/*.parquet -- unsafe for
        # CI per the "never use production data" rule). Same assertions,
        # small synthetic data instead.
        result = runner.run(build_e3a_config())
        assert result.raw_feature_count == 7  # 5 (E1) + title + skill_list
        assert {"title", "skill_list"}.issubset(result.raw_feature_columns)
        assert result.transformed_feature_count is not None
        assert result.transformed_feature_count > 0

        fitted_preprocessor = result.fitted_workflow.named_steps["preprocessor"]
        feature_names = fitted_preprocessor.get_feature_names_out()
        assert len(feature_names) == result.transformed_feature_count

    def test_e3b_skill_multihot_experiment_runs_successfully(self, runner):
        result = runner.run(build_e3b_config())
        assert result.raw_feature_count == 7  # 5 (E1) + title + skill_list
        assert {"title", "skill_list"}.issubset(result.raw_feature_columns)
        assert result.transformed_feature_count is not None
        assert result.transformed_feature_count > 0

        fitted_preprocessor = result.fitted_workflow.named_steps["preprocessor"]
        feature_names = fitted_preprocessor.get_feature_names_out()
        assert len(feature_names) == result.transformed_feature_count

        fitted_model = result.fitted_workflow.named_steps["model"]
        assert hasattr(fitted_model, "n_features_in_")

    def test_fitted_pipeline_can_predict_on_held_out_rows(self, runner, split_paths):
        _, validation_path, _, validation_df = split_paths
        result = runner.run(build_e1_config())

        feature_cols = list(result.raw_feature_columns)
        predictions = result.fitted_workflow.predict(validation_df[feature_cols])
        assert predictions.shape[0] == len(validation_df)
        assert np.isfinite(predictions).all()

    def test_metrics_are_deterministic_across_repeated_runs(self, split_paths):
        train_path, validation_path, _, _ = split_paths
        runner_a = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)
        runner_b = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)

        result_a = runner_a.run(build_e1_config())
        result_b = runner_b.run(build_e1_config())

        assert result_a.log_metrics == pytest.approx(result_b.log_metrics)
        assert result_a.annual_metrics == pytest.approx(result_b.annual_metrics)


# ======================================================================
# Failure modes
# ======================================================================


class TestFailureModes:
    def test_missing_train_file_raises_custom_exception(self, tmp_path):
        validation_df = _make_split_df(5, seed=3)
        validation_path = tmp_path / "validation.parquet"
        validation_df.to_parquet(validation_path)

        runner = SalaryTrainingRunner(
            train_path=tmp_path / "does_not_exist.parquet",
            validation_path=validation_path,
        )
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_missing_validation_file_raises_custom_exception(self, tmp_path):
        train_df = _make_split_df(5, seed=4)
        train_path = tmp_path / "train.parquet"
        train_df.to_parquet(train_path)

        runner = SalaryTrainingRunner(
            train_path=train_path,
            validation_path=tmp_path / "does_not_exist.parquet",
        )
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_empty_train_dataset_raises_custom_exception(self, tmp_path):
        empty_train = _make_split_df(0, seed=5)
        validation_df = _make_split_df(5, seed=6)
        train_path = tmp_path / "train.parquet"
        validation_path = tmp_path / "validation.parquet"
        empty_train.to_parquet(train_path)
        validation_df.to_parquet(validation_path)

        runner = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_missing_target_column_raises_custom_exception(self, tmp_path):
        train_df = _make_split_df(10, seed=7).drop(columns=["target_log_salary"])
        validation_df = _make_split_df(5, seed=8)
        train_path = tmp_path / "train.parquet"
        validation_path = tmp_path / "validation.parquet"
        train_df.to_parquet(train_path)
        validation_df.to_parquet(validation_path)

        runner = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_non_positive_annual_target_in_validation_raises_custom_exception(self, tmp_path):
        train_df = _make_split_df(10, seed=9)
        validation_df = _make_split_df(5, seed=10)
        validation_df.loc[0, "target_annual_salary"] = -1.0
        train_path = tmp_path / "train.parquet"
        validation_path = tmp_path / "validation.parquet"
        train_df.to_parquet(train_path)
        validation_df.to_parquet(validation_path)

        runner = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_missing_feature_column_raises_custom_exception(self, tmp_path):
        train_df = _make_split_df(10, seed=11).drop(columns=["company_state"])
        validation_df = _make_split_df(5, seed=12)
        train_path = tmp_path / "train.parquet"
        validation_path = tmp_path / "validation.parquet"
        train_df.to_parquet(train_path)
        validation_df.to_parquet(validation_path)

        runner = SalaryTrainingRunner(train_path=train_path, validation_path=validation_path)
        with pytest.raises(CustomException):
            runner.run(build_e1_config())

    def test_invalid_config_type_raises_custom_exception(self, runner):
        with pytest.raises(CustomException):
            runner.run({"model_name": "ridge"})  # type: ignore[arg-type]