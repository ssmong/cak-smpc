def format_time_stats_ms(stats: dict, decimals: int = 2) -> str:
    """Format controller timing stats from seconds to milliseconds."""
    scale = 1000.0
    mean_ms = float(stats.get("mean", 0.0)) * scale
    max_ms = float(stats.get("max", 0.0)) * scale
    min_ms = float(stats.get("min", 0.0)) * scale
    std_ms = float(stats.get("std", 0.0)) * scale
    count = int(stats.get("count", 0))
    return (
        f"mean={mean_ms:.{decimals}f} ms, max={max_ms:.{decimals}f} ms, "
        f"min={min_ms:.{decimals}f} ms, std={std_ms:.{decimals}f} ms, n={count}"
    )
