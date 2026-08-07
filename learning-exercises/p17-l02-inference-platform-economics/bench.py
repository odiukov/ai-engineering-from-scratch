"""Входные данные для замера скорости."""

_vendor = {
    "per_mtok": None,
    "per_minute": 0.55,
    "per_prediction": None,
    "tokens_per_minute": 900_000,
    "reserved_minutes_per_day": 1440,
}
_catalog = {f"v{i:04d}": dict(_vendor) for i in range(2000)}

BENCH = {
    "per_token_cost": (100_000_000, 0.90),
    "per_minute_cost": (100_000_000, 900_000, 0.55, 1440),
    "per_prediction_cost": (500_000, 0.006),
    "effective_rate_per_mtok": (792.0, 100_000_000),
    "utilization_breakeven": (0.90, 900_000, 0.55, 1440),
    "blended_rate": (0.90, 0.4, 0.5),
    "selfhosted_breakeven_requests": (0.002, 2000.0, 0.0005),
    "cheapest_vendor": (_catalog, 100_000_000, 500_000),
}
