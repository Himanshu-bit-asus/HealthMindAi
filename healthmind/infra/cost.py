"""Budget helpers. YOU supply price_per_hr (look it up on the AWS pricing page for your
region): this module never guesses prices. Credits may not cover every service, and may expire."""
import math

def job_cost(price_per_hr: float, seconds: float, instances: int = 1) -> float:
    if price_per_hr < 0 or seconds < 0 or instances < 1: raise ValueError("invalid cost inputs")
    return price_per_hr * instances * seconds / 3600

def worst_case_cost(price_per_hr: float, max_minutes: float, instances: int = 1) -> float:
    return job_cost(price_per_hr, max_minutes * 60, instances)

def max_minutes_for_budget(price_per_hr: float, budget_usd: float, instances: int = 1) -> int:
    """Largest whole-minute runtime whose worst-case cost stays within budget."""
    if price_per_hr <= 0 or budget_usd < 0: raise ValueError("invalid cost inputs")
    return math.floor(budget_usd / (price_per_hr * instances) * 60)
