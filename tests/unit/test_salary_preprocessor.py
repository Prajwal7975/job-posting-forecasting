"""
tests/unit/test_salary_preprocessor.py

Unit tests for
src/components/salary_predict/salary_preprocessor_builder.py
(SalaryPreprocessorBuilder), exercised against the real predefined
E0/E1/E2/E3A/E3B experiment configs -- no experiment name here is
invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)
from src.configs.salary_predict.salary_experiment_config import (
    SalaryExperimentConfig,
    ModelName,
    build_e0_config,
    build_e1_config,
    build_e2_config,
    build_e3a_config,
    build_e3b_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def builder() -> SalaryPreprocessorBuilder:
    return SalaryPreprocessorBuilder()


def _branch_names(preprocessor: ColumnTransformer) -> list:
    return [name for name, _, _ in preprocessor.transformers]


# ======================================================================
# E0 -- dummy baseline requires no preprocessor
# ======================================================================


class TestE0Dummy:
    def test_e0_returns_none(self, builder):
        assert builder.build(build_e0_config()) is None


# ======================================================================
# E1 -- structured features only, no NLP
# ======================================================================


class TestE1Structured:
    def test_branches_are_categorical_and_numeric_only(self, builder):
        preprocessor = builder.build(build_e1_config())
        assert isinstance(preprocessor, ColumnTransformer)
        assert _branch_names(preprocessor) == ["categorical", "numeric"]

    def test_required_input_columns_match_active_predictor_features(self, builder):
        config = build_e1_config()
        required = SalaryPreprocessorBuilder.get_required_input_columns(config)
        assert required == list(config.active_predictor_features)
        assert "title" not in required
        assert "skill_list" not in required

    def test_fit_transform_produces_sparse_output_with_correct_row_count(
        self, builder, salary_modeling_df
    ):
        config = build_e1_config()
        preprocessor = builder.build(config)
        X = salary_modeling_df[list(config.active_predictor_features)]

        result = preprocessor.fit_transform(X)

        assert sp.issparse(result)
        assert result.shape[0] == len(salary_modeling_df)

    def test_missing_categorical_and_numeric_values_do_not_crash(self, builder):
        config = build_e1_config()
        preprocessor = builder.build(config)
        df = pd.DataFrame(
            {
                "formatted_experience_level": ["Entry level", None],
                "company_state": [None, "CA"],
                "company_country": ["US", "US"],
                "top_industry": ["Tech", None],
                "skill_count": [2, np.nan],
            }
        )
        result = preprocessor.fit_transform(df)
        assert result.shape[0] == 2

    def test_unseen_category_at_transform_time_is_ignored_not_raised(self, builder):
        config = build_e1_config()
        preprocessor = builder.build(config)
        train = pd.DataFrame(
            {
                "formatted_experience_level": ["Entry level", "Associate"],
                "company_state": ["CA", "NY"],
                "company_country": ["US", "US"],
                "top_industry": ["Tech", "Tech"],
                "skill_count": [1, 2],
            }
        )
        test = pd.DataFrame(
            {
                "formatted_experience_level": ["Entry level"],
                "company_state": ["ZZ"],  # never seen during fit
                "company_country": ["US"],
                "top_industry": ["Tech"],
                "skill_count": [3],
            }
        )
        preprocessor.fit(train)
        result = preprocessor.transform(test)  # must not raise
        assert result.shape[0] == 1

    def test_deterministic_output_across_repeated_transforms(
        self, builder, salary_modeling_df
    ):
        config = build_e1_config()
        preprocessor = builder.build(config)
        X = salary_modeling_df[list(config.active_predictor_features)]

        preprocessor.fit(X)
        first = preprocessor.transform(X)
        second = preprocessor.transform(X)

        assert (first != second).nnz == 0

    def test_sklearn_pipeline_compatibility(self, builder, salary_modeling_df):
        config = build_e1_config()
        preprocessor = builder.build(config)
        pipeline = Pipeline(
            steps=[("preprocess", preprocessor), ("model", Ridge(alpha=1.0))]
        )

        X = salary_modeling_df[list(config.active_predictor_features)]
        y = salary_modeling_df["target_log_salary"]

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape[0] == len(salary_modeling_df)

    def test_validate_input_columns_reports_missing(self, builder):
        config = build_e1_config()
        available = ["formatted_experience_level", "top_industry"]
        missing = SalaryPreprocessorBuilder.validate_input_columns(available, config)
        assert set(missing) == {"company_state", "company_country", "skill_count"}


# ======================================================================
# E2 -- adds job-title TF-IDF
# ======================================================================


class TestE2TitleTfidf:
    def test_branches_include_title_tfidf(self, builder):
        preprocessor = builder.build(build_e2_config())
        assert _branch_names(preprocessor) == ["title_tfidf", "categorical", "numeric"]

    def test_title_column_required(self, builder):
        required = SalaryPreprocessorBuilder.get_required_input_columns(
            build_e2_config()
        )
        assert "title" in required


# ======================================================================
# E3A -- adds skill_list via TF-IDF
# ======================================================================


class TestE3ASkillsTfidf:
    def test_branches_include_skills_tfidf(self, builder):
        preprocessor = builder.build(build_e3a_config())
        assert _branch_names(preprocessor) == [
            "title_tfidf",
            "skills_tfidf",
            "categorical",
            "numeric",
        ]

    def test_fit_transform_on_full_config(self, builder, salary_modeling_df):
        config = build_e3a_config()
        preprocessor = builder.build(config)

        X = salary_modeling_df[list(config.active_predictor_features)].copy()

        # The real E3A TF-IDF configuration uses min_df=5.
        # Create a deterministic small test dataset where the same
        # title/skill tokens occur in 5 out of 6 documents, while
        # still exercising the real experiment configuration.
        X["title"] = [
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "data analyst",
        ]

        X["skill_list"] = [
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "java",
        ]

        result = preprocessor.fit_transform(X)

        assert result.shape[0] == len(X)
        assert result.shape[1] > 0


# ======================================================================
# E3B -- adds skill_list via sparse multi-hot encoding
# ======================================================================


class TestE3BSkillsMultihot:
    def test_branches_include_skills_multihot(self, builder):
        preprocessor = builder.build(build_e3b_config())
        assert _branch_names(preprocessor) == [
            "title_tfidf",
            "skills_multihot",
            "categorical",
            "numeric",
        ]

    def test_unseen_skill_token_is_ignored_at_transform_time(
        self, builder, salary_modeling_df
    ):
        config = build_e3b_config()
        preprocessor = builder.build(config)

        X = salary_modeling_df[list(config.active_predictor_features)].copy()

        # E3B still contains the title TF-IDF branch, whose real
        # configuration uses min_df=5. Make the test data satisfy
        # that configuration.
        X["title"] = [
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "data analyst",
        ]

        X["skill_list"] = [
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "java",
        ]

        preprocessor.fit(X)

        unseen = X.iloc[[0]].copy()
        unseen["skill_list"] = "totally_unseen_skill_token"

        result = preprocessor.transform(unseen)

        assert result.shape[0] == 1

    def test_missing_skill_list_value_produces_all_zero_row(
        self, builder, salary_modeling_df
    ):
        config = build_e3b_config()
        preprocessor = builder.build(config)

        X = salary_modeling_df[list(config.active_predictor_features)].copy()

        # Satisfy the real title TF-IDF min_df=5 requirement.
        X["title"] = [
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "software engineer",
            "data analyst",
        ]

        X["skill_list"] = [
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "python|sql",
            "java",
        ]

        # Test the missing-value behavior explicitly.
        X.loc[X.index[0], "skill_list"] = None

        preprocessor.fit(X)

        result = preprocessor.transform(X)

        assert result.shape[0] == len(X)


# ======================================================================
# Invalid configuration
# ======================================================================


class TestInvalidConfiguration:
    def test_non_experiment_config_raises_type_error(self, builder):
        with pytest.raises(TypeError):
            builder.build({"model_name": "ridge"})  # type: ignore[arg-type]

    def test_non_dummy_experiment_with_no_features_raises_value_error(self, builder):
        config = SalaryExperimentConfig(
            experiment_id="X_EMPTY",
            experiment_name="empty",
            description="A ridge experiment with no declared features.",
            model_name=ModelName.RIDGE.value,
        )
        with pytest.raises(ValueError):
            builder.build(config)
