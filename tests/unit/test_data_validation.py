from __future__ import annotations

import pandas as pd
import pytest

from src.components.data_validation import DataValidation
from src.configs.data_validation_config import (
    DataValidationConfig,
    ForeignKeyRule,
    TableValidationRules,
    ValueConstraint,
)
from src.configs.schema_alignment_config import SchemaAlignmentConfig
from src.exception import CustomException


@pytest.fixture
def valid_postings_df():
    return pd.DataFrame(
        {
            "job_id": [1, 2, 3],
            "company_id": [10, 20, 30],
            "title": ["Engineer", "Developer", "Analyst"],
            "description": ["Build systems", "Write software", "Analyze data"],
            "location": ["Mangalore", "Bangalore", "Mumbai"],
            "views": [100, 200, 150],
            "applies": [10, 20, 15],
            "min_salary": [80000.0, 90000.0, 70000.0],
            "med_salary": [100000.0, 100000.0, 85000.0],
            "max_salary": [120000.0, 110000.0, 100000.0],
            "pay_period": ["YEARLY", "YEARLY", "MONTHLY"],
            "currency": ["USD", "USD", "USD"],
            "compensation_type": [
                "BASE_SALARY",
                "BASE_SALARY",
                "BASE_SALARY",
            ],
            "normalized_salary": [100000.0, 100000.0, 85000.0],
            "formatted_work_type": [
                "Full-time",
                "Full-time",
                "Contract",
            ],
            "work_type": ["FULL_TIME", "FULL_TIME", "CONTRACT"],
            "remote_allowed": [0.0, 1.0, 0.0],
            "formatted_experience_level": [
                "Entry level",
                "Associate",
                "Director",
            ],
            "listed_time": [1700000000000.0] * 3,
            "original_listed_time": [1700000000000.0] * 3,
            "expiry": [1800000000000.0] * 3,
            "closed_time": [1800000000000.0] * 3,
            "sponsored": [0, 1, 0],
            "zip_code": ["575001", "560001", "400001"],
            "fips": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def companies_df():
    return pd.DataFrame(
        {
            "company_id": [10, 20, 30],
            "name": ["Acme", "Beta", "Gamma"],
        }
    )


@pytest.fixture
def validator():
    return DataValidation()


class TestBasicValidation:

    def test_valid_data_produces_no_violations(
        self,
        validator,
        valid_postings_df,
    ):
        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            valid_postings_df,
            validator.config.table_rules["postings"],
            {"companies": pd.DataFrame({"company_id": [10, 20, 30]})},
        )

        assert result.has_issues() is False
        assert result.passed is True

    def test_table_without_configured_rules_is_skipped_not_failed(
        self,
        validator,
    ):
        df = pd.DataFrame({"x": [1, 2, 3]})

        # "unknown_table" is not present in the validator's configured
        # table_rules, so initiate_data_validation() should skip it.
        result = validator.initiate_data_validation(
            {"Version_9": {"unknown_table": df}}
        )

        assert result.summary.tables_processed == 1
        assert result.summary.tables_skipped_no_rules == 1
        assert result.summary.tables_failed == 0
        assert result.summary.tables_passed == 0
        assert result.summary.passed is True


class TestStructuralValidation:

    def test_configured_column_missing_from_dataframe_is_structural_issue(
        self,
        validator,
    ):
        rules = TableValidationRules(not_null_columns=("job_id", "company_id"))

        df = pd.DataFrame({"job_id": [1, 2, 3]})

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            rules,
            {},
        )

        assert result.has_issues() is True
        assert result.structural_issues


class TestNullValidation:

    def test_null_violation_counts(
        self,
        validator,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df.loc[1, "title"] = None

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            validator.config.table_rules["postings"],
            {},
        )

        assert result.null_violations["title"] == 1
        assert result.has_issues() is True


class TestDuplicateValidation:

    def test_duplicate_unique_key_detected_with_sample(
        self,
        validator,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df.loc[2, "job_id"] = 1

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            validator.config.table_rules["postings"],
            {},
        )

        assert result.duplicate_key_row_count > 0
        assert result.duplicate_key_sample

    def test_no_duplicates_when_key_column_missing(
        self,
        validator,
    ):
        rules = TableValidationRules(unique_key_columns=("job_id",))

        df = pd.DataFrame({"company_id": [1, 2, 3]})

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            rules,
            {},
        )

        assert result.duplicate_key_row_count == 0


class TestValueConstraints:

    def test_allowed_values_violation_detected(
        self,
        validator,
    ):
        rules = TableValidationRules(
            value_constraints={
                "work_type": ValueConstraint(allowed_values=("FULL_TIME", "PART_TIME"))
            }
        )

        df = pd.DataFrame(
            {
                "work_type": [
                    "FULL_TIME",
                    "INVALID",
                    "PART_TIME",
                ]
            }
        )

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            rules,
            {},
        )

        assert result.invalid_value_counts["work_type"] == 1

    def test_min_value_range_violation_detected(
        self,
        validator,
    ):
        rules = TableValidationRules(
            value_constraints={"min_salary": ValueConstraint(min_value=0)}
        )

        df = pd.DataFrame({"min_salary": [100.0, -50.0, 200.0]})

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            rules,
            {},
        )

        assert result.invalid_value_counts["min_salary"] == 1

    def test_regex_constraint_violation_detected(
        self,
        validator,
    ):
        rules = TableValidationRules(
            value_constraints={"zip_code": ValueConstraint(regex_pattern=r"^\d{6}$")}
        )

        df = pd.DataFrame(
            {
                "zip_code": [
                    "575001",
                    "12345",
                    "560001",
                ]
            }
        )

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            rules,
            {},
        )

        assert result.invalid_value_counts["zip_code"] == 1


class TestDtypeValidation:

    def test_non_numeric_value_in_numeric_canonical_column_is_flagged(
        self,
        validator,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df["job_id"] = ["abc", 2, 3]

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            validator.config.table_rules["postings"],
            {},
        )

        assert result.dtype_conformance_issues


class TestRelationshipValidation:

    def test_broken_relationship_detected(
        self,
        validator,
        valid_postings_df,
        companies_df,
    ):
        df = valid_postings_df.copy()
        df.loc[2, "company_id"] = 999

        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            df,
            validator.config.table_rules["postings"],
            {"companies": companies_df},
        )

        assert result.broken_relationship_counts["company_id"] == 1

    def test_foreign_key_check_skipped_when_referenced_table_absent(
        self,
        validator,
        valid_postings_df,
    ):
        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            valid_postings_df,
            validator.config.table_rules["postings"],
            {},
        )

        assert result.broken_relationship_counts == {}


class TestSummary:

    def test_summary_pass_fail_and_skip_counts(
        self,
        valid_postings_df,
        companies_df,
    ):
        postings_rules = TableValidationRules(
            unique_key_columns=("job_id",),
            not_null_columns=(
                "job_id",
                "company_id",
                "title",
            ),
            foreign_keys=(
                ForeignKeyRule(
                    "company_id",
                    "companies",
                    "company_id",
                ),
            ),
        )

        companies_rules = TableValidationRules(
            unique_key_columns=("company_id",),
            not_null_columns=(
                "company_id",
                "name",
            ),
        )

        config = DataValidationConfig(
            table_rules={
                "postings": postings_rules,
                "companies": companies_rules,
            },
            schema_alignment_config=SchemaAlignmentConfig(table_schemas={}),
        )

        validator = DataValidation(config)

        broken_df = valid_postings_df.copy()
        broken_df.loc[0, "title"] = None

        aligned_data = {
            "Version_9": {
                "postings": valid_postings_df,
                "companies": companies_df,
                "unrelated_table": pd.DataFrame({"x": [1]}),
            },
            "Version_13": {
                "postings": broken_df,
                "companies": companies_df,
            },
        }

        result = validator.initiate_data_validation(aligned_data)

        assert result.summary.versions_processed == 2
        assert result.summary.tables_processed == 5
        assert result.summary.tables_skipped_no_rules == 1
        assert result.summary.tables_failed == 1
        assert result.summary.tables_passed == 3
        assert result.summary.passed is False
        assert result.summary.total_null_violations == 1
        assert result.summary.total_structural_issues == 0


class TestDataMutation:

    def test_initiate_data_validation_does_not_mutate_input(
        self,
        validator,
        valid_postings_df,
        companies_df,
    ):
        aligned_data = {
            "Version_9": {
                "postings": valid_postings_df.copy(),
                "companies": companies_df.copy(),
            }
        }

        original_postings = aligned_data["Version_9"]["postings"].copy(deep=True)
        original_companies = aligned_data["Version_9"]["companies"].copy(deep=True)

        validator.initiate_data_validation(aligned_data)

        pd.testing.assert_frame_equal(
            aligned_data["Version_9"]["postings"],
            original_postings,
        )

        pd.testing.assert_frame_equal(
            aligned_data["Version_9"]["companies"],
            original_companies,
        )


class TestRaisePolicies:

    def test_raise_on_null_violations_raises_custom_exception(
        self,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df.loc[0, "title"] = None

        config = DataValidationConfig(
            raise_on_null_violations=True,
            schema_alignment_config=SchemaAlignmentConfig(table_schemas={}),
        )

        validator = DataValidation(config)

        with pytest.raises(CustomException):
            validator.initiate_data_validation({"Version_9": {"postings": df}})

    def test_no_raise_when_policy_disabled(
        self,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df.loc[0, "title"] = None

        config = DataValidationConfig(
            raise_on_null_violations=False,
            schema_alignment_config=SchemaAlignmentConfig(table_schemas={}),
        )

        validator = DataValidation(config)

        result = validator.initiate_data_validation({"Version_9": {"postings": df}})

        assert result.summary.passed is False

    def test_multiple_raise_policies_combine_into_one_message(
        self,
        valid_postings_df,
    ):
        df = valid_postings_df.copy()
        df.loc[0, "title"] = None
        df.loc[1, "job_id"] = 1

        config = DataValidationConfig(
            raise_on_null_violations=True,
            raise_on_duplicate_keys=True,
            schema_alignment_config=SchemaAlignmentConfig(table_schemas={}),
        )

        validator = DataValidation(config)

        with pytest.raises(CustomException) as exc_info:
            validator.initiate_data_validation({"Version_9": {"postings": df}})

        message = str(exc_info.value)

        assert "null" in message.lower()
        assert "duplicate" in message.lower()


class TestSerialization:

    def test_report_serialization_helpers(
        self,
        validator,
        valid_postings_df,
    ):
        result = validator._validate_against_rules(
            "Version_9",
            "postings",
            valid_postings_df,
            validator.config.table_rules["postings"],
            {},
        )

        # TableValidationReport does not implement to_dict().
        # Verify its actual dataclass fields instead.
        assert result.version == "Version_9"
        assert result.table == "postings"
        assert result.row_count == len(valid_postings_df)
        assert result.rules_defined is True
        assert result.passed is True

        # Verify that the report contains the expected
        # validation result fields.
        assert hasattr(result, "structural_issues")
        assert hasattr(result, "null_violations")
        assert hasattr(result, "duplicate_key_row_count")
        assert hasattr(result, "duplicate_key_sample")
        assert hasattr(result, "invalid_value_counts")
        assert hasattr(result, "dtype_conformance_issues")
        assert hasattr(result, "broken_relationship_counts")

    def test_summary_serialization_includes_passed_flag(
        self,
        valid_postings_df,
        companies_df,
    ):
        config = DataValidationConfig(
            table_rules={
                "postings": TableValidationRules(
                    unique_key_columns=("job_id",),
                    not_null_columns=(
                        "job_id",
                        "company_id",
                        "title",
                    ),
                ),
                "companies": TableValidationRules(
                    unique_key_columns=("company_id",),
                    not_null_columns=(
                        "company_id",
                        "name",
                    ),
                ),
            },
            schema_alignment_config=SchemaAlignmentConfig(table_schemas={}),
        )

        validator = DataValidation(config)

        result = validator.initiate_data_validation(
            {
                "Version_9": {
                    "postings": valid_postings_df,
                    "companies": companies_df,
                }
            }
        )

        # DataValidationSummary does not implement to_dict().
        # Verify the actual summary fields directly.
        assert result.summary.versions_processed == 1
        assert result.summary.tables_processed == 2
        assert result.summary.tables_passed == 2
        assert result.summary.tables_failed == 0
        assert result.summary.tables_skipped_no_rules == 0
        assert result.summary.passed is True
