"""
tests/unit/test_salary_model_factory.py

Unit tests for src/components/salary_predict/salary_model_factory.py
(SalaryModelFactory).

Covers every model family registered in SalaryModelFactory._MODEL_REGISTRY
via the real predefined model-family experiment configs (M0-M5), so no
model name or hyperparameter set here is invented.
"""

from __future__ import annotations

import sys
import types

import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.components.salary_predict.salary_model_factory import SalaryModelFactory
from src.configs.salary_predict.salary_model_factory_config import (
    build_m0_config,
    build_m1_config,
    build_m2_config,
    build_m3_config,
    build_m4_config,
    build_m5_config,
)
from src.exception import CustomException

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> SalaryModelFactory:
    return SalaryModelFactory()


# ======================================================================
# Every registered model family, built from its real predefined config
# ======================================================================


class TestBuildKnownFamilies:
    def test_dummy_from_m0_config(self, factory):
        model = factory.build(build_m0_config())
        assert isinstance(model, DummyRegressor)
        assert model.strategy == "median"
        assert not hasattr(model, "constant_")  # unfitted

    def test_ridge_from_m1_config(self, factory):
        model = factory.build(build_m1_config())
        assert isinstance(model, Ridge)
        assert model.alpha == 1.0
        assert not hasattr(model, "coef_")  # unfitted

    def test_random_forest_from_m2_config(self, factory):
        model = factory.build(build_m2_config())
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 200
        assert model.random_state == 42
        assert not hasattr(model, "estimators_")  # unfitted

    def test_lightgbm_from_m3_config(self, factory):
        pytest.importorskip("lightgbm")
        model = factory.build(build_m3_config())
        assert type(model).__name__ == "LGBMRegressor"
        assert model.get_params()["n_estimators"] == 300

    def test_xgboost_from_m4_config(self, factory):
        pytest.importorskip("xgboost")
        model = factory.build(build_m4_config())
        assert type(model).__name__ == "XGBRegressor"

    def test_catboost_from_m5_config(self, factory):
        pytest.importorskip("catboost")
        model = factory.build(build_m5_config())
        assert type(model).__name__ == "CatBoostRegressor"

    def test_sklearn_clone_is_compatible_with_returned_estimator(self, factory):
        model = factory.build(build_m1_config())
        cloned = clone(model)
        assert isinstance(cloned, Ridge)
        assert cloned.alpha == model.alpha


# ======================================================================
# Missing optional dependency
# ======================================================================


class TestMissingOptionalDependency:
    def test_missing_lightgbm_raises_custom_exception(self, factory, monkeypatch):
        monkeypatch.setitem(sys.modules, "lightgbm", None)
        with pytest.raises(CustomException) as exc_info:
            factory.build(build_m3_config())
        assert "lightgbm" in str(exc_info.value).lower()


# ======================================================================
# Invalid configuration
# ======================================================================


class TestInvalidConfiguration:
    def test_unsupported_model_name_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(
            model_experiment_id="X0",
            model_name="not_a_real_model",
            model_params={},
        )
        with pytest.raises(CustomException) as exc_info:
            factory.build(config)
        assert "Unsupported salary model family" in str(exc_info.value)

    def test_missing_required_attributes_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(model_experiment_id="X0")  # no model_name/model_params
        with pytest.raises(CustomException):
            factory.build(config)

    def test_non_string_model_name_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(model_name=123, model_params={})
        with pytest.raises(CustomException):
            factory.build(config)

    def test_empty_model_name_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(model_name="   ", model_params={})
        with pytest.raises(CustomException):
            factory.build(config)

    def test_non_mapping_model_params_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(model_name="ridge", model_params=["alpha", 1.0])
        with pytest.raises(CustomException):
            factory.build(config)

    def test_non_string_param_keys_raise_custom_exception(self, factory):
        config = types.SimpleNamespace(model_name="ridge", model_params={1: 2})
        with pytest.raises(CustomException):
            factory.build(config)

    def test_invalid_param_for_model_raises_custom_exception(self, factory):
        config = types.SimpleNamespace(
            model_name="ridge", model_params={"not_a_real_param": 1}
        )
        with pytest.raises(CustomException):
            factory.build(config)


# ======================================================================
# Read-only helper methods
# ======================================================================


class TestHelperMethods:
    def test_list_supported_models_contains_all_six_families(self, factory):
        supported = SalaryModelFactory.list_supported_models()
        assert supported == ("catboost", "dummy", "lightgbm", "random_forest", "ridge", "xgboost")

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("ridge", True),
            ("RIDGE", True),  # normalization is case-insensitive
            ("dummy", True),
            ("not_a_model", False),
            ("", False),
        ],
    )
    def test_is_supported_model_family(self, model_name, expected):
        assert SalaryModelFactory.is_supported_model_family(model_name) is expected

    def test_is_supported_model_family_handles_non_string_gracefully(self):
        assert SalaryModelFactory.is_supported_model_family(None) is False
        assert SalaryModelFactory.is_supported_model_family(123) is False
