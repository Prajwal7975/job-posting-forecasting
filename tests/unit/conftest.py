"""
tests/conftest.py

Shared, deterministic pytest fixtures for the salary-prediction test
suite. Nothing here touches production data, production MLflow, or
production filesystem paths -- every fixture that needs disk I/O is
scoped to pytest's `tmp_path`.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ======================================================================
# Schema Alignment / Data Validation / Data Cleaning fixtures
# (raw, pre-alignment "postings" and "companies" tables)
# ======================================================================


@pytest.fixture
def raw_postings_df() -> pd.DataFrame:
    """
    A small, deterministic raw 'postings' table using a mix of canonical
    names and known aliases (id -> job_id, job_title -> title), plus one
    genuinely unmapped source column, to exercise alias resolution and
    unmapped-column handling in the same fixture.
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "job_title": ["Data Scientist", "  ML Engineer  ", "Backend Dev"],
            "company_id": [10, 20, 30],
            "description": ["desc a", "desc b", "desc c"],
            "min_salary": [80000, 90000, None],
            "max_salary": [120000, 110000, None],
            "pay_period": ["YEARLY", "yearly", "MONTHLY"],
            "some_vendor_specific_column": ["x", "y", "z"],
        }
    )


@pytest.fixture
def raw_companies_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": [10, 20, 30],
            "company_name": ["Acme Inc", "Beta LLC", "Gamma Co"],
        }
    )


# ======================================================================
# Common Feature Store fixture
# (mimics the output of CommonFeatureEngineering that
#  SalaryFeatureEngineering consumes)
# ======================================================================


@pytest.fixture
def feature_store_df() -> pd.DataFrame:
    """
    Deterministic synthetic rows covering: a clean range-midpoint salary,
    a median-fallback salary, a non-USD row, an unsupported pay period,
    an absurd hourly rate (data-quality guard), and an out-of-bounds
    annual salary -- so a single fixture exercises every funnel branch
    in SalaryFeatureEngineering._construct_target.
    """
    return pd.DataFrame(
        {
            "title": [
                "Data Scientist",
                "ML Engineer",
                "Backend Developer",
                "Sales Rep",
                "Bad Hourly Row",
                "Too Low Salary",
                "Missing Predictor Row",
            ],
            "description": ["d"] * 7,
            "min_salary": [90000, None, 40, 50000, 1000, 5000, 70000],
            "med_salary": [None, 85000, None, None, None, None, None],
            "max_salary": [110000, None, 60, 55000, 1000, 5000, 90000],
            "pay_period": [
                "YEARLY",
                "YEARLY",
                "HOURLY",
                "YEARLY",
                "HOURLY",
                "YEARLY",
                "YEARLY",
            ],
            "currency": ["USD", "USD", "USD", "EUR", "USD", "USD", "USD"],
            "formatted_experience_level": [
                "Mid-Senior level",
                "Entry level",
                "Associate",
                "Entry level",
                "Entry level",
                "Entry level",
                "Entry level",
            ],
            "formatted_work_type": [
                "FULL_TIME",
                "FULL_TIME",
                "CONTRACT",
                "FULL_TIME",
                "FULL_TIME",
                "FULL_TIME",
                "FULL_TIME",
            ],
            "company_state": ["CA", "NY", "TX", "TX", "CA", "CA", "CA"],
            "company_country": ["US"] * 7,
            "company_size": [5, 3, 2, 4, 5, 5, 5],
            "company_employee_count": [1000, 50, 20, 200, 1000, 1000, 1000],
            "company_follower_count": [500, 10, 5, 80, 500, 500, 500],
            "top_industry": ["Tech", "Tech", "Tech", "Retail", "Tech", "Tech", "Tech"],
            "top_skill": ["python", "python", "sql", "sales", "python", "python", "python"],
            "skill_list": [
                "python|sql",
                "python|pandas",
                "sql",
                "sales|crm",
                "python",
                "python",
                "python",
            ],
            "skill_count": [2, 2, 1, 2, 1, 1, 1],
            "company_name": ["Acme", "Beta", "Gamma", "Delta", "Acme", "Acme", "Acme"],
            "location": ["SF, CA", "NY, NY", "Austin, TX", "Dallas, TX", "SF, CA", "SF, CA", "SF, CA"],
            "original_listed_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "listed_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "dataset_version": ["v1"] * 7,
        }
    )


# ======================================================================
# Salary modeling dataset fixture
# (mimics the output of SalaryFeatureEngineering that the preprocessor
#  and training stages consume)
# ======================================================================


@pytest.fixture
def salary_modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_annual_salary": [90000.0, 105000.0, 60000.0, 130000.0, 75000.0, 88000.0],
            "target_log_salary": [
                11.4076,
                11.5624,
                11.0021,
                11.7753,
                11.2252,
                11.3849,
            ],
            "title": [
                "Data Scientist",
                "Senior ML Engineer",
                "Backend Developer",
                "Staff Data Scientist",
                "Junior Analyst",
                "Data Engineer",
            ],
            "skill_list": [
                "python|sql",
                "python|pandas|sql",
                "java|sql",
                "python|scala",
                None,
                "python|airflow",
            ],
            "formatted_experience_level": [
                "Mid-Senior level",
                "Mid-Senior level",
                "Entry level",
                "Director",
                "Entry level",
                "Associate",
            ],
            "company_state": ["CA", "NY", None, "CA", "TX", "WA"],
            "company_country": ["US", "US", "US", "US", "US", "US"],
            "top_industry": ["Tech", "Tech", "Tech", "Tech", "Retail", "Tech"],
            "skill_count": [2, 3, 2, 2, 0, 2],
            "posting_group_id": [f"grp_{i}" for i in range(6)],
        }
    )
