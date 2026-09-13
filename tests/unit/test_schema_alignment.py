"""
tests/unit/test_schema_alignment.py

Unit tests for src/components/schema_alignment.py (SchemaAlignment).

These tests exercise the real SchemaAlignment class against small,
synthetic DataFrames -- never the production dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.components.schema_alignment import (
    SchemaAlignment,
    TableAlignmentReport,
    AlignmentSummary,
    SchemaAlignmentResult,
)
from src.configs.schema_alignment_config import (
    SchemaAlignmentConfig,
    TableSchema,
)
from src.exception import CustomException

pytestmark = pytest.mark.unit


# ======================================================================
# Fixtures local to this module
# ======================================================================


@pytest.fixture
def aligner() -> SchemaAlignment:
    return SchemaAlignment()


# ======================================================================
# Valid alignment / alias resolution
# ======================================================================


class TestValidAlignment:
    def test_aliases_are_resolved_and_renamed(self, aligner, raw_postings_df):
        aligned_df, report = aligner.align_table(
            "Version_9", "postings", raw_postings_df
        )

        # "id" -> job_id and "job_title" -> title are both known aliases.
        assert "job_id" in aligned_df.columns
        assert "title" in aligned_df.columns
        assert report.renamed_columns.get("job_id") == "id"
        assert report.renamed_columns.get("title") == "job_title"

    def test_case_insensitive_alias_resolution(self, aligner):
        df = pd.DataFrame({"JOB_ID": [1, 2], "TITLE": ["a", "b"]})
        aligned_df, report = aligner.align_table("Version_9", "postings", df)

        assert "job_id" in aligned_df.columns
        assert list(aligned_df["job_id"]) == [1, 2]
        # Case-only differences still count as a rename since the source
        # column name string differs from the canonical name.
        assert report.renamed_columns.get("job_id") == "JOB_ID"

    def test_column_order_matches_schema_column_order(self, aligner, raw_postings_df):
        aligned_df, _ = aligner.align_table("Version_9", "postings", raw_postings_df)

        schema = SchemaAlignmentConfig().table_schemas["postings"]
        canonical_columns = [c for c in aligned_df.columns if c in schema.column_order]
        assert canonical_columns == [c for c in schema.column_order]

    def test_row_count_preserved(self, aligner, raw_postings_df):
        aligned_df, report = aligner.align_table(
            "Version_9", "postings", raw_postings_df
        )
        assert len(aligned_df) == len(raw_postings_df)
        assert report.row_count == len(raw_postings_df)


# ======================================================================
# Missing columns (required vs optional)
# ======================================================================


class TestMissingColumns:
    def test_missing_optional_column_is_created_empty_with_canonical_dtype(
        self, aligner, raw_postings_df
    ):
        # raw_postings_df never provides 'views' (optional, Int64).
        aligned_df, report = aligner.align_table(
            "Version_9", "postings", raw_postings_df
        )

        assert "views" in report.missing_columns
        assert aligned_df["views"].isna().all()
        assert str(aligned_df["views"].dtype) == "Int64"

    def test_missing_required_column_raises_custom_exception(self, aligner):
        df = pd.DataFrame({"title": ["a", "b"]})  # no job_id, no 'id' alias either
        with pytest.raises(CustomException):
            aligner.align_table("Version_9", "postings", df)

    def test_required_column_present_but_entirely_null_is_reported_not_raised(
        self,
    ):
        config = SchemaAlignmentConfig(
            table_schemas={
                "postings": TableSchema(
                    column_order=("job_id", "title"),
                    aliases={
                        "job_id": ("job_id",),
                        "title": ("title",),
                    },
                    canonical_dtypes={
                        "job_id": "Int64",
                        "title": "string",
                    },
                    required_columns=("job_id",),
                )
            }
        )

        aligner = SchemaAlignment(config)

        df = pd.DataFrame(
            {
                "job_id": [None, None],
                "title": ["a", "b"],
            }
        )

        aligned_df, report = aligner.align_table(
            "Version_9",
            "postings",
            df,
        )

        # The required column exists, so alignment does not raise.
        assert "job_id" in aligned_df.columns

        # The required column is entirely NULL, so it is reported.
        assert "job_id" in report.empty_required_columns

        # No canonical columns are missing.
        assert report.missing_columns == ()

        # Therefore there are no structural issues.
        assert report.has_structural_issues() is False


# ======================================================================
# Unmapped / unknown source columns
# ======================================================================


class TestUnmappedColumns:
    def test_unmapped_columns_preserved_by_default_with_prefix(
        self, aligner, raw_postings_df
    ):
        aligned_df, report = aligner.align_table(
            "Version_9", "postings", raw_postings_df
        )

        assert "some_vendor_specific_column" in report.unmapped_source_columns
        assert report.unmapped_columns_preserved is True
        assert "unmapped__some_vendor_specific_column" in aligned_df.columns

    def test_unmapped_columns_dropped_when_configured(self, raw_postings_df):
        config = SchemaAlignmentConfig(preserve_unmapped_columns=False)
        aligner = SchemaAlignment(config)

        aligned_df, report = aligner.align_table(
            "Version_9", "postings", raw_postings_df
        )

        assert report.unmapped_columns_preserved is False
        assert "some_vendor_specific_column" in report.unmapped_source_columns
        assert not any(col.startswith("unmapped__") for col in aligned_df.columns)
        assert "some_vendor_specific_column" not in aligned_df.columns


# ======================================================================
# Tables with no configured schema
# ======================================================================


class TestUnknownTable:
    def test_table_without_configured_schema_passes_through_unaligned(self, aligner):
        df = pd.DataFrame({"anything": [1, 2, 3]})
        aligned_df, report = aligner.align_table("Version_9", "not_a_real_table", df)

        pd.testing.assert_frame_equal(aligned_df, df)
        assert report.missing_columns == ()
        assert report.unmapped_columns_preserved is True
        assert set(report.unmapped_source_columns) == {"anything"}


# ======================================================================
# Immutability & exception handling
# ======================================================================


class TestImmutabilityAndErrors:
    def test_input_dataframe_is_not_mutated(self, aligner, raw_postings_df):
        original = raw_postings_df.copy(deep=True)
        aligner.align_table("Version_9", "postings", raw_postings_df)
        pd.testing.assert_frame_equal(raw_postings_df, original)

    def test_invalid_input_type_raises_custom_exception(self, aligner):
        with pytest.raises(CustomException):
            aligner.align_table("Version_9", "postings", None)  # type: ignore[arg-type]


# ======================================================================
# Report / summary serialization
# ======================================================================


class TestReportsAndSummary:
    def test_report_serialization_helpers(self, aligner, raw_postings_df):
        _, report = aligner.align_table("Version_9", "postings", raw_postings_df)

        assert isinstance(report, TableAlignmentReport)
        assert isinstance(report.as_log_dict(), dict)
        assert isinstance(report.to_json(), str)
        assert "postings" in report.to_json()

    def test_align_version_returns_one_report_per_table(
        self, aligner, raw_postings_df, raw_companies_df
    ):
        version_data = {"postings": raw_postings_df, "companies": raw_companies_df}
        aligned_version, reports = aligner.align_version("Version_9", version_data)

        assert set(aligned_version.keys()) == {"postings", "companies"}
        assert len(reports) == 2
        assert {r.table for r in reports} == {"postings", "companies"}

    def test_initiate_schema_alignment_builds_correct_summary(
        self, aligner, raw_postings_df, raw_companies_df
    ):
        all_versions = {
            "Version_9": {"postings": raw_postings_df, "companies": raw_companies_df}
        }
        result = aligner.initiate_schema_alignment(all_versions)

        assert isinstance(result, SchemaAlignmentResult)
        assert isinstance(result.summary, AlignmentSummary)
        assert result.summary.versions_processed == 1
        assert result.summary.tables_processed == 2
        assert result.summary.columns_renamed >= 2  # job_id + title aliases
        assert "postings" in result.aligned_data["Version_9"]
        assert "companies" in result.aligned_data["Version_9"]

    def test_initiate_schema_alignment_propagates_missing_required_column(
        self, aligner
    ):
        all_versions = {"Version_9": {"postings": pd.DataFrame({"title": ["a"]})}}
        with pytest.raises(CustomException):
            aligner.initiate_schema_alignment(all_versions)
