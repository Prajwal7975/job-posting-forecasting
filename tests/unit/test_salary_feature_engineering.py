"""
tests/unit/test_salary_feature_engineering.py

Unit tests for
src/components/salary_predict/salary_feature_engineering.py
(SalaryFeatureEngineering).

This component reads a feature-store CSV from disk and writes its
outputs to disk, so every test configures `base_artifacts_dir` to
pytest's `tmp_path` -- nothing here ever touches a real project
artifact directory.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.components.salary_predict.salary_feature_engineering import (
    SalaryFeatureEngineering,
)
from src.configs.salary_predict.salary_feature_engineering_config import (
    SalaryFeatureEngineeringConfig,
)
from src.entity.salary_predict.salary_feature_engineering_entity import (
    SalaryFeatureEngineeringResult,
)
from src.exception import CustomException

pytestmark = pytest.mark.unit


def _write_feature_store(tmp_path, df: pd.DataFrame, config: SalaryFeatureEngineeringConfig) -> None:
    path = config.feature_store_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def base_config(tmp_path) -> SalaryFeatureEngineeringConfig:
    return SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)


# ======================================================================
# Input validation / failure modes
# ======================================================================


class TestInputValidation:
    def test_missing_feature_store_file_raises_custom_exception(self, base_config):
        engine = SalaryFeatureEngineering(base_config)
        with pytest.raises(CustomException):
            engine.initiate_salary_feature_engineering()

    def test_missing_required_target_construction_column_raises(
        self, tmp_path, base_config, feature_store_df
    ):
        df = feature_store_df.drop(columns=["currency"])
        _write_feature_store(tmp_path, df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        with pytest.raises(CustomException):
            engine.initiate_salary_feature_engineering()

    def test_leakage_column_configured_as_predictor_raises(
        self, tmp_path, feature_store_df
    ):
        config = SalaryFeatureEngineeringConfig(
            base_artifacts_dir=tmp_path,
            candidate_predictor_columns=["min_salary"],  # min_salary is a leakage column
        )
        _write_feature_store(tmp_path, feature_store_df, config)
        engine = SalaryFeatureEngineering(config)
        with pytest.raises(CustomException):
            engine.initiate_salary_feature_engineering()

    def test_zero_valid_salary_rows_raises_custom_exception(self, tmp_path, base_config):
        df = pd.DataFrame(
            {
                "min_salary": [None],
                "med_salary": [None],
                "max_salary": [None],
                "pay_period": ["YEARLY"],
                "currency": ["USD"],
                "title": ["x"],
                "company_name": ["y"],
                "location": ["z"],
            }
        )
        _write_feature_store(tmp_path, df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        with pytest.raises(CustomException):
            engine.initiate_salary_feature_engineering()


# ======================================================================
# Target construction funnel -- exercises every branch in one fixture
# ======================================================================


class TestTargetConstructionFunnel:
    def test_funnel_counts_match_expected_breakdown(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)

        result = engine.initiate_salary_feature_engineering()
        summary = result.summary

        assert summary.input_row_count == 7
        assert summary.salary_candidate_count == 7
        assert summary.valid_usd_salary_count == 5  # excludes the EUR row
        assert summary.rows_removed_missing_salary == 0
        assert summary.rows_removed_currency == 1  # the EUR row
        assert summary.rows_removed_unsupported_pay_period == 0
        assert summary.rows_removed_out_of_bounds == 1  # the $5,000/yr row
        assert summary.final_row_count == 4
        assert summary.target_coverage_pct == pytest.approx(4 / 7 * 100, rel=1e-4)

    def test_range_midpoint_preferred_when_both_min_and_max_present(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        row = result.dataframe.loc[result.dataframe["title"] == "Data Scientist"].iloc[0]
        assert row["target_annual_salary"] == pytest.approx(100000.0)
        assert row["salary_target_source"] == "RANGE_MIDPOINT"

    def test_median_fallback_used_when_no_valid_range(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        row = result.dataframe.loc[result.dataframe["title"] == "ML Engineer"].iloc[0]
        assert row["target_annual_salary"] == pytest.approx(85000.0)
        assert row["salary_target_source"] == "MEDIAN_SALARY"

    def test_hourly_rate_is_correctly_annualized(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        row = result.dataframe.loc[result.dataframe["title"] == "Backend Developer"].iloc[0]
        assert row["target_annual_salary"] == pytest.approx(50 * 2080.0)

    def test_suspiciously_high_hourly_rate_is_rejected(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        titles_kept = set(result.dataframe["title"])
        assert "Bad Hourly Row" not in titles_kept

    def test_out_of_bounds_annual_salary_excluded(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        assert "Too Low Salary" not in set(result.dataframe["title"])

    def test_non_usd_currency_excluded(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        assert "Sales Rep" not in set(result.dataframe["title"])

    def test_log_target_matches_log1p_of_annual_target(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        expected = np.log1p(result.dataframe["target_annual_salary"])
        np.testing.assert_allclose(result.dataframe["target_log_salary"], expected)


# ======================================================================
# Predictor resolution
# ======================================================================


class TestPredictorResolution:
    def test_missing_candidate_predictor_column_is_skipped_not_fatal(self, tmp_path):
        config = SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)
        df = pd.DataFrame(
            {
                "min_salary": [80000],
                "med_salary": [None],
                "max_salary": [90000],
                "pay_period": ["YEARLY"],
                "currency": ["USD"],
                "title": ["Data Scientist"],
                "formatted_experience_level": ["Entry level"],
                # every other candidate_predictor_column intentionally absent
            }
        )
        _write_feature_store(tmp_path, df, config)
        engine = SalaryFeatureEngineering(config)
        result = engine.initiate_salary_feature_engineering()

        assert "formatted_experience_level" in result.summary.feature_columns
        assert "company_state" not in result.summary.feature_columns

    def test_negative_numeric_predictor_masked_before_log_transform(self, tmp_path):
        config = SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)
        df = pd.DataFrame(
            {
                "min_salary": [80000, 90000],
                "med_salary": [None, None],
                "max_salary": [90000, 100000],
                "pay_period": ["YEARLY", "YEARLY"],
                "currency": ["USD", "USD"],
                "title": ["a", "b"],
                "company_employee_count": [-5, 100],
            }
        )
        _write_feature_store(tmp_path, df, config)
        engine = SalaryFeatureEngineering(config)
        result = engine.initiate_salary_feature_engineering()

        assert pd.isna(result.dataframe.loc[0, "log_company_employee_count"])
        assert result.dataframe.loc[1, "log_company_employee_count"] == pytest.approx(
            np.log1p(100)
        )


# ======================================================================
# Posting group id
# ======================================================================


class TestPostingGroupId:
    def test_identical_title_company_location_yields_same_group_id(self, tmp_path):
        config = SalaryFeatureEngineeringConfig(base_artifacts_dir=tmp_path)
        df = pd.DataFrame(
            {
                "min_salary": [80000, 80000],
                "med_salary": [None, None],
                "max_salary": [90000, 90000],
                "pay_period": ["YEARLY", "YEARLY"],
                "currency": ["USD", "USD"],
                "title": ["Data Scientist", "Data Scientist"],
                "company_name": ["Acme", "Acme"],
                "location": ["SF, CA", "SF, CA"],
            }
        )
        _write_feature_store(tmp_path, df, config)
        engine = SalaryFeatureEngineering(config)
        result = engine.initiate_salary_feature_engineering()

        group_ids = result.dataframe["posting_group_id"].tolist()
        assert group_ids[0] == group_ids[1]


# ======================================================================
# Output artifacts
# ======================================================================


class TestOutputArtifacts:
    def test_result_is_correct_entity_type(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()
        assert isinstance(result, SalaryFeatureEngineeringResult)

    def test_dataset_and_metadata_files_are_written_and_parseable(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        assert result.salary_modeling_dataset_path.exists()
        assert result.metadata_path.exists()
        assert result.report_path.exists()
        assert result.schema_fingerprint_path.exists()

        reloaded = pd.read_parquet(result.salary_modeling_dataset_path)
        assert len(reloaded) == result.summary.final_row_count

        metadata = json.loads(result.metadata_path.read_text())
        assert metadata["row_count"] == result.summary.final_row_count

    def test_archive_snapshot_written_by_default(self, tmp_path, base_config, feature_store_df):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()

        assert result.archive_dataset_path is not None
        assert result.archive_dataset_path.exists()

    def test_archive_disabled_via_config(self, tmp_path, feature_store_df):
        config = SalaryFeatureEngineeringConfig(
            base_artifacts_dir=tmp_path, keep_archive_snapshots=False
        )
        _write_feature_store(tmp_path, feature_store_df, config)
        engine = SalaryFeatureEngineering(config)
        result = engine.initiate_salary_feature_engineering()

        assert result.archive_dataset_path is None

    def test_source_feature_store_file_is_never_modified(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        original_bytes = base_config.feature_store_path.read_bytes()

        engine = SalaryFeatureEngineering(base_config)
        engine.initiate_salary_feature_engineering()

        assert base_config.feature_store_path.read_bytes() == original_bytes

    def test_integrity_passed_flag_is_true_on_success(
        self, tmp_path, base_config, feature_store_df
    ):
        _write_feature_store(tmp_path, feature_store_df, base_config)
        engine = SalaryFeatureEngineering(base_config)
        result = engine.initiate_salary_feature_engineering()
        assert result.summary.integrity_passed is True
