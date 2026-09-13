"""
tests/unit/test_data_cleaning.py

Unit tests for src/components/data_cleaning.py (DataCleaning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import time
import sys

from src.components.data_cleaning import (
    DataCleaning,
    TableCleaningReport,
    DataCleaningSummary,
    DataCleaningResult,
)
from src.configs.data_cleaning_config import (
    DataCleaningConfig,
    TableCleaningRule,
    NullPolicy,
    DuplicatePolicy,
    ForeignKeyPolicy,
    InvalidValuePolicy,
)
from src.exception import CustomException

pytestmark = pytest.mark.unit


@pytest.fixture
def cleaner() -> DataCleaning:
    return DataCleaning()


# ======================================================================
# String trimming / whitespace / lowercase / empty-string handling
# ======================================================================


class TestStringCleaning:
    def test_trims_whitespace_and_normalizes_internal_spaces(self, cleaner):
        df = pd.DataFrame({"job_id": [1], "name": ["  Acme   Inc  "]})
        cleaned, count = cleaner._clean_strings(df, TableCleaningRule())
        assert cleaned.loc[0, "name"] == "Acme Inc"
        assert count == 1

    def test_lowercase_column_rule_applied(self, cleaner):
        rule = TableCleaningRule(lowercase_columns=("industry_name",))
        df = pd.DataFrame({"industry_name": ["Software", "FINANCE"]})
        cleaned, _ = cleaner._clean_strings(df, rule)
        assert list(cleaned["industry_name"]) == ["software", "finance"]

    def test_empty_string_converted_to_null(self, cleaner):
        df = pd.DataFrame({"name": ["   ", "Acme"]})
        cleaned, count = cleaner._clean_strings(df, TableCleaningRule())
        assert pd.isna(cleaned.loc[0, "name"])
        assert cleaned.loc[1, "name"] == "Acme"
        assert count == 1  # one cell was modified and converted to null

    def test_string_cleaning_is_idempotent(self, cleaner):
        df = pd.DataFrame({"name": ["  Acme  Inc "]})
        once, _ = cleaner._clean_strings(df, TableCleaningRule())
        twice, second_pass_count = cleaner._clean_strings(once, TableCleaningRule())
        assert once.equals(twice)
        assert second_pass_count == 0


# ======================================================================
# Null handling
# ======================================================================


class TestNullHandling:
    def test_drop_row_policy(self, cleaner):
        rule = TableCleaningRule(null_policies={"company_id": NullPolicy.DROP_ROW})
        df = pd.DataFrame({"company_id": [1, None, 3]})
        cleaned, filled = cleaner._handle_nulls(df, rule)
        assert len(cleaned) == 2
        assert filled == 0

    def test_fill_default_policy(self, cleaner):
        rule = TableCleaningRule(
            null_policies={"name": NullPolicy.FILL_DEFAULT},
            default_fill_values={"name": "Unknown Company"},
        )
        df = pd.DataFrame({"name": [None, "Acme"]})
        cleaned, filled = cleaner._handle_nulls(df, rule)
        assert cleaned.loc[0, "name"] == "Unknown Company"
        assert filled == 1

    def test_leave_policy_is_a_no_op(self, cleaner):
        rule = TableCleaningRule(null_policies={"x": NullPolicy.LEAVE})
        df = pd.DataFrame({"x": [None, 1]})
        cleaned, filled = cleaner._handle_nulls(df, rule)
        assert cleaned["x"].isna().sum() == 1
        assert filled == 0


# ======================================================================
# Dtype conversion (delegates to SchemaAlignmentConfig canonical dtypes)
# ======================================================================


class TestDtypeConversion:
    def test_converts_column_to_canonical_dtype(self, cleaner):
        df = pd.DataFrame({"job_id": ["1", "2", "3"]})  # postings.job_id -> Int64
        cleaned, conversions = cleaner._convert_dtypes(df, "postings", "Version_9")
        assert str(cleaned["job_id"].dtype) == "Int64"
        assert conversions == 1

    def test_non_numeric_value_coerced_to_na_for_int64_target(self, cleaner):
        df = pd.DataFrame({"job_id": ["1", "not-a-number"]})
        cleaned, _ = cleaner._convert_dtypes(df, "postings", "Version_9")
        assert cleaned["job_id"].isna().iloc[1]

    def test_unconfigured_table_is_a_no_op(self, cleaner):
        df = pd.DataFrame({"anything": [1, 2]})
        cleaned, conversions = cleaner._convert_dtypes(df, "not_a_real_table", "Version_9")
        assert conversions == 0
        pd.testing.assert_frame_equal(cleaned, df)


# ======================================================================
# Invalid value handling
# ======================================================================


class TestInvalidValueHandling:
    def test_drop_row_policy_for_out_of_bounds_values(self, cleaner):
        rule = TableCleaningRule(
            numeric_min_bounds={"min_salary": 0.0},
            invalid_value_policy=InvalidValuePolicy.DROP_ROW,
        )
        df = pd.DataFrame({"min_salary": [-100, 50000]})
        cleaned, invalid_count = cleaner._handle_invalid_values(df, rule)
        assert len(cleaned) == 1
        assert invalid_count == 1

    def test_nullify_policy_for_out_of_bounds_values(self, cleaner):
        rule = TableCleaningRule(
            numeric_min_bounds={"min_salary": 0.0},
            invalid_value_policy=InvalidValuePolicy.NULLIFY,
        )
        df = pd.DataFrame({"min_salary": [-100, 50000]})
        cleaned, invalid_count = cleaner._handle_invalid_values(df, rule)
        assert cleaned["min_salary"].isna().iloc[0]
        assert invalid_count == 1

    def test_leave_policy_is_a_no_op(self, cleaner):
        rule = TableCleaningRule(
            numeric_min_bounds={"min_salary": 0.0},
            invalid_value_policy=InvalidValuePolicy.LEAVE,
        )
        df = pd.DataFrame({"min_salary": [-100, 50000]})
        cleaned, invalid_count = cleaner._handle_invalid_values(df, rule)
        assert invalid_count == 0
        pd.testing.assert_frame_equal(cleaned, df)


# ======================================================================
# Duplicate handling
# ======================================================================


class TestDuplicateHandling:
    def test_keep_first_policy(self, cleaner):
        rule = TableCleaningRule(primary_keys=("job_id",), duplicate_policy=DuplicatePolicy.KEEP_FIRST)
        df = pd.DataFrame({"job_id": [1, 1, 2], "title": ["a", "b", "c"]})
        cleaned, removed = cleaner._remove_duplicates(df, rule)
        assert len(cleaned) == 2
        assert cleaned.iloc[0]["title"] == "a"
        assert removed == 1

    def test_drop_all_policy(self, cleaner):
        rule = TableCleaningRule(primary_keys=("job_id",), duplicate_policy=DuplicatePolicy.DROP_ALL)
        df = pd.DataFrame({"job_id": [1, 1, 2]})
        cleaned, removed = cleaner._remove_duplicates(df, rule)
        assert list(cleaned["job_id"]) == [2]
        assert removed == 2


# ======================================================================
# Foreign key / orphan handling
# ======================================================================


class TestOrphanHandling:
    def test_drop_child_removes_orphan_rows(self, cleaner):
        rule = TableCleaningRule(
            foreign_keys={"company_id": ("companies", "company_id")},
            fk_policy=ForeignKeyPolicy.DROP_CHILD,
        )
        df = pd.DataFrame({"company_id": [1, 999]})
        version_data = {"companies": pd.DataFrame({"company_id": [1, 2]})}
        cleaned, orphans_removed = cleaner._remove_orphans(df, rule, version_data)

        assert list(cleaned["company_id"]) == [1]
        assert orphans_removed == 1

    def test_null_foreign_key_values_are_kept(self, cleaner):
        rule = TableCleaningRule(
            foreign_keys={"company_id": ("companies", "company_id")},
            fk_policy=ForeignKeyPolicy.DROP_CHILD,
        )
        df = pd.DataFrame({"company_id": [1, None]})
        version_data = {"companies": pd.DataFrame({"company_id": [1]})}
        cleaned, _ = cleaner._remove_orphans(df, rule, version_data)
        assert len(cleaned) == 2

    def test_leave_policy_is_a_no_op(self, cleaner):
        rule = TableCleaningRule(
            foreign_keys={"company_id": ("companies", "company_id")},
            fk_policy=ForeignKeyPolicy.LEAVE,
        )
        df = pd.DataFrame({"company_id": [1, 999]})
        version_data = {"companies": pd.DataFrame({"company_id": [1]})}
        cleaned, orphans_removed = cleaner._remove_orphans(df, rule, version_data)
        assert orphans_removed == 0
        assert len(cleaned) == 2


# ======================================================================
# Full pipeline orchestration
# ======================================================================


class TestFullCleaningPipeline:
    def test_end_to_end_cleaning_for_postings_and_companies(self, cleaner):
        validated_data = {
            "Version_9": {
                "postings": pd.DataFrame(
                    {
                        "job_id": [1, 2, 3],
                        "company_id": [10, None, 20],
                        "title": ["  Data Scientist ", "ML Eng", "Backend Dev"],
                    }
                ),
                "companies": pd.DataFrame(
                    {"company_id": [10, 20], "name": [None, "Beta LLC"]}
                ),
            }
        }
        result = cleaner.initiate_data_cleaning(validated_data)

        assert isinstance(result, DataCleaningResult)
        assert isinstance(result.summary, DataCleaningSummary)
        cleaned_postings = result.cleaned_data["Version_9"]["postings"]
        cleaned_companies = result.cleaned_data["Version_9"]["companies"]

        # company_id null-row was dropped per postings' null_policy.
        assert len(cleaned_postings) == 2
        assert cleaned_companies.loc[cleaned_companies["company_id"] == 10, "name"].iloc[0] == "Unknown Company"
        assert result.summary.tables_processed == 2
        assert result.summary.tables_failed == 0

    def test_input_dataframes_are_not_mutated(self, cleaner):
        postings = pd.DataFrame({"job_id": [1], "company_id": [10], "title": [" x "]})
        companies = pd.DataFrame({"company_id": [10], "name": ["Acme"]})
        original_postings = postings.copy(deep=True)
        original_companies = companies.copy(deep=True)

        cleaner.initiate_data_cleaning(
            {"Version_9": {"postings": postings, "companies": companies}}
        )

        pd.testing.assert_frame_equal(postings, original_postings)
        pd.testing.assert_frame_equal(companies, original_companies)

    def test_empty_dataframe_does_not_crash(self, cleaner):
        empty_postings = pd.DataFrame({"job_id": [], "company_id": [], "title": []})
        result = cleaner.initiate_data_cleaning({"Version_9": {"postings": empty_postings}})
        assert result.summary.total_rows_before == 0
        assert result.summary.total_rows_after == 0
        assert result.summary.percentage_rows_removed == 0.0

    def test_continue_on_error_true_records_failure_and_keeps_going(self):
        config = DataCleaningConfig(continue_on_error=True)
        cleaner = DataCleaning(config)

        # A table with no rule falls back to TableCleaningRule() (a no-op
        # default) so cleaning cannot fail from bad config alone here;
        # instead we simulate a hostile input type to force an internal
        # exception during string cleaning.
        bad_table = object()  # not a DataFrame -> .copy(deep=True) will fail
        validated_data = {
            "Version_9": {
                "postings": pd.DataFrame({"job_id": [1], "company_id": [1], "title": ["a"]}),
                "broken_table": bad_table,
            }
        }
        result = cleaner.initiate_data_cleaning(validated_data)  # type: ignore[arg-type]
        assert result.summary.tables_failed == 1
        assert result.summary.tables_passed == 1

    def test_continue_on_error_false_raises_custom_exception(self):
        config = DataCleaningConfig(continue_on_error=False)
        cleaner = DataCleaning(config)

        validated_data = {"Version_9": {"broken_table": object()}}
        with pytest.raises(CustomException):
            cleaner.initiate_data_cleaning(validated_data)  # type: ignore[arg-type]

    def test_report_serialization(self, cleaner):
        validated_data = {
            "Version_9": {"postings": pd.DataFrame({"job_id": [1], "company_id": [1], "title": ["a"]})}
        }
        result = cleaner.initiate_data_cleaning(validated_data)
        report = result.reports[0]
        assert isinstance(report, TableCleaningReport)
        assert report.table == "postings"
        assert report.rows_before == 1
