import math, pytest
from infra.cost import job_cost, worst_case_cost, max_minutes_for_budget

def test_job_cost():
    assert math.isclose(job_cost(10.0, 1800), 5.0)
    assert math.isclose(job_cost(10.0, 3600, instances=2), 20.0)

def test_budget_cap_never_exceeds_budget():
    for price in (1.4, 4.0, 16.688):
        m = max_minutes_for_budget(price, 10.0)
        assert worst_case_cost(price, m) <= 10.0 < worst_case_cost(price, m + 1)

def test_whole_credit_on_big_instance_is_under_7_hours():
    assert max_minutes_for_budget(16.688, 114.0) < 7 * 60

def test_invalid():
    with pytest.raises(ValueError): job_cost(-1, 10)
    with pytest.raises(ValueError): max_minutes_for_budget(0, 10)
