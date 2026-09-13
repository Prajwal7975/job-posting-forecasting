"""
tests/api/conftest.py

Shared fixtures for the FastAPI salary-prediction API tests.

The real app (`api/salary_api.py`) builds its
`model_loader` and `inference_service` as MODULE-LEVEL globals, and its
`lifespan` handler calls `model_loader.load()` synchronously at startup,
re-raising on failure (fail-fast: the app refuses to come up without a
model). To keep tests fully offline:

- `api_module` pre-populates `model_loader._model` with a REAL, small,
  fitted sklearn pipeline (built with the actual SalaryPreprocessorBuilder
  against the actual E3B experiment config, which matches
  SalaryInferenceFeatureBuilder.MODEL_FEATURE_COLUMNS exactly) so that
  `SalaryModelLoader.load()`'s own short-circuit
  (`if self._model is not None: return self._model`) means startup never
  touches real MLflow.
- `client` then enters the app via `TestClient(...)` as a context manager
  so the (now harmless) lifespan startup/shutdown actually runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.components.salary_predict.salary_preprocessor_builder import (
    SalaryPreprocessorBuilder,
)
from src.configs.salary_predict.salary_experiment_config import build_e3b_config

pytestmark = pytest.mark.api


def _build_fake_production_model():
    """A real, tiny, fitted preprocessor+Ridge pipeline standing in for
    the MLflow-registered production model -- built from the same E3B
    architecture (title + skill_list multihot + core categoricals +
    skill_count) that SalaryInferenceFeatureBuilder's schema matches.

    Needs at least `min_df` (5) rows per repeated term for
    TfidfVectorizer's title branch to build a non-empty vocabulary --
    an earlier 4-row version failed with "max_df corresponds to <
    documents than min_df" for exactly that reason.
    """
    df = pd.DataFrame(
        {
            # "data" appears in exactly 6/8 titles -- inside the safe band
            # between min_df=5 and max_df*n_doc=7.6, so it survives both
            # thresholds (6 rows sharing it, 2 that don't, to avoid the
            # opposite failure of "data" appearing in ALL 8 docs and
            # getting pruned as too common instead).
            "title": [
                "Data Scientist",
                "Data Engineer",
                "Data Analyst",
                "Senior Data Scientist",
                "Lead Data Engineer",
                "Staff Data Analyst",
                "Machine Learning Engineer",
                "Software Engineer",
            ],
            "skill_list": [
                "python|sql",
                "python|aws",
                "sql|docker",
                "aws|docker",
                "python|docker",
                "python|sql|aws",
                "sql|docker|aws",
                "python|docker|sql",
            ],
            "formatted_experience_level": [
                "Entry level",
                "Mid-Senior level",
                "Associate",
                "Entry level",
                "Mid-Senior level",
                "Associate",
                "Entry level",
                "Mid-Senior level",
            ],
            "company_state": ["CA", "NY", None, "TX", "CA", "NY", "TX", "CA"],
            "company_country": ["US"] * 8,
            "top_industry": ["Tech", "Tech", "Tech", "Retail", "Tech", "Tech", "Retail", "Tech"],
            "skill_count": [2, 2, 2, 2, 2, 3, 3, 3],
        }
    )
    y = np.log1p(
        [95000.0, 130000.0, 80000.0, 60000.0, 110000.0, 140000.0, 105000.0, 150000.0]
    )

    preprocessor = SalaryPreprocessorBuilder().build(build_e3b_config())
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])
    pipeline.fit(df, y)
    return pipeline


@pytest.fixture
def api_module():
    from api import salary_api as api_module

    api_module.model_loader._model = _build_fake_production_model()
    yield api_module
    api_module.model_loader._model = None  # keep tests isolated from each other


@pytest.fixture
def client(api_module):
    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return {
        "title": "Data Scientist",
        "skill_list": "Python|SQL|Docker",
        "formatted_experience_level": "Mid-Senior level",
        "company_state": "CA",
        "company_country": "US",
        "top_industry": "Technology",
    }