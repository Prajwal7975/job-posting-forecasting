import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomException
from src.configs.data_cleaning_config import (
    DataCleaningConfig,
    NullPolicy,
    DuplicatePolicy,
    ForeignKeyPolicy,
    InvalidValuePolicy,
    TableCleaningRule,
)


@dataclass
class TableCleaningReport:
    version: str
    table: str
    rows_before: int
    rows_after: int
    rows_removed: int
    nulls_filled: int
    invalid_values_handled: int
    duplicates_removed: int
    orphan_rows_removed: int
    dtype_conversions: int
    strings_modified: int
    execution_time_seconds: float


@dataclass
class DataCleaningSummary:
    versions_processed: int
    tables_processed: int
    tables_passed: int
    tables_failed: int
    total_rows_before: int
    total_rows_after: int
    total_rows_removed: int
    percentage_rows_removed: float
    total_nulls_filled: int
    total_invalid_values_handled: int
    total_duplicates_removed: int
    total_orphan_rows_removed: int
    total_dtype_conversions: int
    execution_time_seconds: float


@dataclass
class DataCleaningResult:
    cleaned_data: Dict[str, Dict[str, pd.DataFrame]]
    reports: List[TableCleaningReport]
    summary: DataCleaningSummary


class DataCleaning:
    def __init__(self, config: Optional[DataCleaningConfig] = None):
        self.config = config or DataCleaningConfig()

    def _clean_strings(
        self, df: pd.DataFrame, rule: TableCleaningRule
    ) -> Tuple[pd.DataFrame, int]:
        """Trims whitespace, normalizes spaces, converts empty strings, and lowercases text safely."""
        df = df.copy()
        strings_modified = 0

        string_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in string_cols:
            original_s = df[col].astype("string")
            clean_s = df[col].astype("string")

            if self.config.trim_strings:
                clean_s = clean_s.str.strip()

            if self.config.normalize_whitespace:
                clean_s = clean_s.str.replace(r"\s+", " ", regex=True)

            if col in rule.lowercase_columns:
                clean_s = clean_s.str.lower()

            if self.config.convert_empty_strings_to_null:
                clean_s = clean_s.replace(r"^\s*$", np.nan, regex=True).replace(
                    {"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan}
                )

            sentinel = "__NULL_SENTINEL__"
            diff_mask = original_s.fillna(sentinel) != clean_s.fillna(sentinel)
            strings_modified += int(diff_mask.sum())

            df[col] = clean_s

        return df, strings_modified

    def _handle_nulls(
        self, df: pd.DataFrame, rule: TableCleaningRule
    ) -> Tuple[pd.DataFrame, int]:
        """Implements table-specific null remediation policies."""
        df = df.copy()
        nulls_filled = 0

        for col, policy in rule.null_policies.items():
            if col not in df.columns:
                continue

            if policy == NullPolicy.DROP_ROW:
                df = df.dropna(subset=[col])
            elif policy == NullPolicy.FILL_DEFAULT:
                fill_val = rule.default_fill_values.get(col)
                if fill_val is not None:
                    null_count = int(df[col].isna().sum())
                    df[col] = df[col].fillna(fill_val)
                    nulls_filled += null_count

        return df, nulls_filled

    def _convert_dtypes(
        self, df: pd.DataFrame, table_name: str, version_name: str
    ) -> Tuple[pd.DataFrame, int]:
        """Aligns datatypes with Schema Alignment as the single source of truth."""
        df = df.copy()
        conversions = 0

        schema_config = self.config.schema_alignment_config
        table_schema = schema_config.table_schemas.get(table_name)

        target_dtypes = table_schema.canonical_dtypes if table_schema else {}

        for col, target_dtype in target_dtypes.items():
            if col in df.columns:
                try:
                    current_dtype = str(df[col].dtype)
                    if current_dtype != target_dtype:
                        if target_dtype == "Int64":
                            df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                                "Int64"
                            )
                        else:
                            df[col] = df[col].astype(target_dtype)
                        conversions += 1
                except Exception as e:
                    logging.warning(
                        f"[{version_name}/{table_name}] Failed converting column '{col}' to {target_dtype}: {e}"
                    )

        return df, conversions

    def _handle_invalid_values(
        self, df: pd.DataFrame, rule: TableCleaningRule
    ) -> Tuple[pd.DataFrame, int]:
        """Handles out-of-bounds or invalid numeric data based on policy configuration."""
        if (
            rule.invalid_value_policy == InvalidValuePolicy.LEAVE
            or not rule.numeric_min_bounds
        ):
            return df, 0

        df = df.copy()
        invalid_count = 0

        for col, min_bound in rule.numeric_min_bounds.items():
            if col not in df.columns:
                continue

            numeric_s = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = (numeric_s < min_bound) & df[col].notna()
            detected_invalids = int(invalid_mask.sum())

            if detected_invalids > 0:
                invalid_count += detected_invalids
                if rule.invalid_value_policy == InvalidValuePolicy.DROP_ROW:
                    df = df[~invalid_mask]
                elif rule.invalid_value_policy == InvalidValuePolicy.NULLIFY:
                    df.loc[invalid_mask, col] = np.nan

        return df, invalid_count

    def _remove_duplicates(
        self, df: pd.DataFrame, rule: TableCleaningRule
    ) -> Tuple[pd.DataFrame, int]:
        """Removes duplicate rows based on primary keys or global exact match."""
        rows_before = len(df)
        df = df.copy()

        subset = list(rule.primary_keys) if rule.primary_keys else None

        if rule.duplicate_policy == DuplicatePolicy.KEEP_FIRST:
            df = df.drop_duplicates(subset=subset, keep="first")
        elif rule.duplicate_policy == DuplicatePolicy.KEEP_LAST:
            df = df.drop_duplicates(subset=subset, keep="last")
        elif rule.duplicate_policy == DuplicatePolicy.DROP_ALL:
            df = df.drop_duplicates(subset=subset, keep=False)

        duplicates_removed = rows_before - len(df)
        return df, duplicates_removed

    def _remove_orphans(
        self,
        df: pd.DataFrame,
        rule: TableCleaningRule,
        version_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, int]:
        """Purges orphan records using relational foreign key definitions."""
        if rule.fk_policy != ForeignKeyPolicy.DROP_CHILD or not rule.foreign_keys:
            return df, 0

        df = df.copy()
        rows_before = len(df)

        for fk_col, (parent_table, parent_key) in rule.foreign_keys.items():
            if fk_col not in df.columns or parent_table not in version_data:
                continue

            parent_df = version_data[parent_table]
            if parent_key not in parent_df.columns:
                continue

            valid_keys = set(parent_df[parent_key].dropna().unique())
            df = df[df[fk_col].isin(valid_keys) | df[fk_col].isna()]

        orphans_removed = rows_before - len(df)
        return df, orphans_removed

    def initiate_data_cleaning(
        self, validated_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> DataCleaningResult:
        """Executes full data cleaning pipeline over all dataset versions."""
        start_time = time.time()
        logging.info("=" * 70)
        logging.info("DATA CLEANING STARTED")
        logging.info("=" * 70)

        cleaned_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        reports: List[TableCleaningReport] = []

        tables_passed = 0
        tables_failed = 0
        total_rows_before = 0
        total_rows_after = 0
        total_nulls_filled = 0
        total_invalid_values_handled = 0
        total_duplicates_removed = 0
        total_orphan_rows_removed = 0
        total_dtype_conversions = 0

        try:
            for version_name, tables in validated_data.items():
                logging.info(f"Cleaning {version_name} ({len(tables)} tables detected)")
                cleaned_data[version_name] = {}

                parent_tables = ["companies", "postings", "industries", "skills"]
                table_order = [t for t in parent_tables if t in tables] + [
                    t for t in tables if t not in parent_tables
                ]

                for table_name in table_order:
                    try:
                        df = tables[table_name].copy(deep=True)
                        t_start = time.time()
                        rows_before = len(df)
                        total_rows_before += rows_before

                        rule = self.config.table_rules.get(
                            table_name, TableCleaningRule()
                        )

                        # Pipeline Operations
                        df, strings_modified = self._clean_strings(df, rule)
                        df, nulls_filled = self._handle_nulls(df, rule)
                        df, dtype_conversions = self._convert_dtypes(
                            df, table_name, version_name
                        )
                        df, invalid_handled = self._handle_invalid_values(df, rule)
                        df, duplicates_removed = self._remove_duplicates(df, rule)
                        df, orphan_rows_removed = self._remove_orphans(
                            df, rule, cleaned_data[version_name]
                        )

                        rows_after = len(df)
                        total_rows_after += rows_after
                        total_nulls_filled += nulls_filled
                        total_invalid_values_handled += invalid_handled
                        total_duplicates_removed += duplicates_removed
                        total_orphan_rows_removed += orphan_rows_removed
                        total_dtype_conversions += dtype_conversions

                        cleaned_data[version_name][table_name] = df
                        t_execution = time.time() - t_start
                        tables_passed += 1

                        report = TableCleaningReport(
                            version=version_name,
                            table=table_name,
                            rows_before=rows_before,
                            rows_after=rows_after,
                            rows_removed=rows_before - rows_after,
                            nulls_filled=nulls_filled,
                            invalid_values_handled=invalid_handled,
                            duplicates_removed=duplicates_removed,
                            orphan_rows_removed=orphan_rows_removed,
                            dtype_conversions=dtype_conversions,
                            strings_modified=strings_modified,
                            execution_time_seconds=round(t_execution, 4),
                        )
                        reports.append(report)

                        logging.info(
                            f"[{version_name}/{table_name}] Cleaned: {rows_before} -> {rows_after} rows "
                            f"(Strings modified: {strings_modified}, Nulls filled: {nulls_filled}, "
                            f"Dtype conversions: {dtype_conversions}, Invalids: {invalid_handled}, "
                            f"Duplicates: {duplicates_removed}, Orphans: {orphan_rows_removed})"
                        )
                    except Exception as e:
                        tables_failed += 1
                        logging.error(
                            f"[{version_name}/{table_name}] Cleaning failed: {e}"
                        )
                        if not self.config.continue_on_error:
                            raise CustomException(e, sys)

            total_execution_time = time.time() - start_time
            total_removed = total_rows_before - total_rows_after
            percentage_removed = (
                round((total_removed / total_rows_before) * 100, 2)
                if total_rows_before > 0
                else 0.0
            )

            summary = DataCleaningSummary(
                versions_processed=len(validated_data),
                tables_processed=len(reports) + tables_failed,
                tables_passed=tables_passed,
                tables_failed=tables_failed,
                total_rows_before=total_rows_before,
                total_rows_after=total_rows_after,
                total_rows_removed=total_removed,
                percentage_rows_removed=percentage_removed,
                total_nulls_filled=total_nulls_filled,
                total_invalid_values_handled=total_invalid_values_handled,
                total_duplicates_removed=total_duplicates_removed,
                total_orphan_rows_removed=total_orphan_rows_removed,
                total_dtype_conversions=total_dtype_conversions,
                execution_time_seconds=round(total_execution_time, 4),
            )

            logging.info("=" * 70)
            logging.info(
                f"Cleaning summary: {summary.tables_passed}/{summary.tables_processed} tables succeeded. "
                f"Total rows removed: {summary.total_rows_removed} ({summary.percentage_rows_removed}%). "
                f"Orphans: {summary.total_orphan_rows_removed}, Nulls filled: {summary.total_nulls_filled}, "
                f"Invalids: {summary.total_invalid_values_handled}, Conversions: {summary.total_dtype_conversions}."
            )
            logging.info("DATA CLEANING COMPLETED")
            logging.info("=" * 70)

            return DataCleaningResult(
                cleaned_data=cleaned_data, reports=reports, summary=summary
            )

        except Exception as e:
            logging.error(f"Critical error occurred during Data Cleaning: {str(e)}")
            raise CustomException(e, sys)
