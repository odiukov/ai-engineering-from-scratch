"""Входные данные для замера скорости."""

_platform = {
    "in_per_mtok": 2.50,
    "out_per_mtok": 10.00,
    "ptu_hourly": 10.0,
    "ptu_tokens_per_hour": 2_000_000,
    "ttft_p99_ondemand_ms": 140.0,
    "ttft_p99_ptu_ms": 57.0,
}
_catalog = {f"p{i:03d}": dict(_platform) for i in range(500)}

BENCH = {
    "ondemand_cost": (30_000_000, 15_000_000, 2.5, 10.0),
    "ptu_units_needed": (45_000_000, 2_000_000, 24),
    "ptu_cost": (45_000_000, 2_000_000, 10.0, 24),
    "ptu_breakeven_utilization": (10.0, 2_000_000, 10.0),
    "cheapest_path": (_platform, 30_000_000, 15_000_000, 24),
    "pick_platform": (_catalog, 30_000_000, 15_000_000, 24, 200.0),
    "redundancy_uplift": (50.0, 3.0, 10.0),
}
