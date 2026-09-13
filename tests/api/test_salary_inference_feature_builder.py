"""
tests/unit/test_salary_inference_feature_builder.py

NOTE: this file wasn't in the originally-specified test tree -- it
covers SalaryInferenceFeatureBuilder, a component uploaded after that
tree was fixed. It's included because the builder's validation logic is
exactly the kind of pure, isolated unit behavior this layer is for, and
test_predict.py already depends on it indirectly (via the API), so a
failure there is easier to diagnose here first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.serving.salary_predict.salary_inference_feature_builder import (
    SalaryInferenceFeatureBuilder,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def builder() -> SalaryInferenceFeatureBuilder:
    return SalaryInferenceFeatureBuilder()


class TestValidBuild:
    def test_builds_one_row_with_expected_columns_in_order(self, builder):
        features = builder.build(
            title="Data Scientist",
            skill_list="Python|SQL|Docker",
            formatted_experience_level="Mid-Senior level",
            company_state="CA",
            company_country="US",
            top_industry="Technology",
        )
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        assert list(features.columns) == list(builder.MODEL_FEATURE_COLUMNS)

    def test_required_fields_are_trimmed(self, builder):
        features = builder.build(
            title="  Data Scientist  ",
            skill_list="Python|SQL",
            formatted_experience_level="  Entry level  ",
        )
        assert features.loc[0, "title"] == "Data Scientist"
        assert features.loc[0, "formatted_experience_level"] == "Entry level"

    def test_optional_fields_default_to_none_when_omitted(self, builder):
        features = builder.build(
            title="Data Scientist",
            skill_list="Python",
            formatted_experience_level="Entry level",
        )
        assert features.loc[0, "company_state"] is None
        assert features.loc[0, "company_country"] is None
        assert features.loc[0, "top_industry"] is None

    def test_empty_string_optional_field_normalized_to_none(self, builder):
        features = builder.build(
            title="Data Scientist",
            skill_list="Python",
            formatted_experience_level="Entry level",
            company_state="   ",
        )
        assert features.loc[0, "company_state"] is None

    def test_skill_count_counts_unique_case_insensitive_skills(self, builder):
        features = builder.build(
            title="Data Scientist",
            skill_list="Python|python|SQL|  |SQL",
            formatted_experience_level="Entry level",
        )
        assert features.loc[0, "skill_count"] == 2  # {"python", "sql"}


class TestRequiredFieldValidation:
    @pytest.mark.parametrize("field", ["title", "skill_list", "formatted_experience_level"])
    def test_none_required_field_raises_value_error(self, builder, field):
        kwargs = {
            "title": "Data Scientist",
            "skill_list": "Python",
            "formatted_experience_level": "Entry level",
        }
        kwargs[field] = None
        with pytest.raises(ValueError):
            builder.build(**kwargs)

    def test_whitespace_only_required_field_raises_value_error(self, builder):
        with pytest.raises(ValueError):
            builder.build(title="   ", skill_list="Python", formatted_experience_level="Entry level")

    def test_non_string_required_field_raises_value_error(self, builder):
        with pytest.raises(ValueError):
            builder.build(title=123, skill_list="Python", formatted_experience_level="Entry level")  # type: ignore[arg-type]

    def test_non_string_optional_field_raises_value_error(self, builder):
        with pytest.raises(ValueError):
            builder.build(
                title="Data Scientist",
                skill_list="Python",
                formatted_experience_level="Entry level",
                company_state=123,  # type: ignore[arg-type]
            )
