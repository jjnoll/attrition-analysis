import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "attrition": ["Yes", "No", "Yes", "Yes", "No", "No"],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "monthly_income": [3000, 6000, 4000, 5000, 7000, 8000],
            "job_satisfaction": [1, 3, 2, 2, 3, 4],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(sample_df):
    # 3 leavers out of 6 employees = 50.0%
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_stay():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_correct_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["HR"] == 100.0    # 2 leavers / 2 employees
    assert rates["Sales"] == 50.0  # 1 leaver  / 2 employees
    assert rates["IT"] == 0.0      # 0 leavers / 2 employees


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_correct_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    # Overtime=Yes: employees 1,3,5 — leavers 1,3 → 2/3 = 66.67%
    assert rates["Yes"] == 66.67
    # Overtime=No: employees 2,4,6 — leaver 4 → 1/3 = 33.33%
    assert rates["No"] == 33.33


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correct_values(sample_df):
    result = average_income_by_attrition(sample_df)
    values = result.set_index("attrition")["avg_monthly_income"]
    # Leavers (Yes): 3000, 4000, 5000 → mean = 4000.0
    assert values["Yes"] == 4000.0
    # Stayers (No): 6000, 7000, 8000 → mean = 7000.0
    assert values["No"] == 7000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_calculates_rate_per_group_not_total_leavers(sample_df):
    # Validates the bug fix: denominator must be group size, not total leavers.
    # satisfaction=2: 2 employees (ids 3,4), both left → true rate = 100%
    # If the old bug were present, it would be 2/3 leavers * 100 = 66.67%
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[2] == 100.0


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_zero_attrition_group(sample_df):
    # satisfaction=4 has 1 employee who stayed → rate should be 0.0, not NaN or error
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[4] == 0.0
