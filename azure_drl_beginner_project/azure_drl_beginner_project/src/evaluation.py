from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def trace_to_dataframe(trace: list[dict]) -> pd.DataFrame:
    if not trace:
        return pd.DataFrame()
    rows = []
    for step, item in enumerate(trace):
        rows.append(
            {
                "step": step,
                "demand": item.get("demand", 0.0),
                "power_it": item.get("power_it", 0.0),
                "power_total": item.get("power_total", item.get("power", 0.0)),
                "pue": item.get("pue", 1.0),
                "sla_violation": item.get("sla_violation", 0.0),
                "active_hosts": item.get("active_hosts", 0),
                "sleep_hosts": item.get("sleep_hosts", 0),
                "off_hosts": item.get("off_hosts", 0),
                "dvfs": item.get("dvfs", 0.0),
                "avg_temp": item.get("avg_temp", 0.0),
                "max_temp": item.get("max_temp", 0.0),
                "switches": item.get("switches", 0),
                "migrations": item.get("migrations", 0),
                "mean_host_age": item.get("mean_host_age", 0.0),
            }
        )
    return pd.DataFrame(rows)


def save_trace_artifacts(trace: list[dict], output_dir: str | Path, prefix: str = "run") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = trace_to_dataframe(trace)
    csv_path = output_dir / f"{prefix}_trace.csv"
    df.to_csv(csv_path, index=False)

    charts = [
        ("power_total", "Tổng power theo thời gian"),
        ("pue", "PUE ước lượng theo thời gian"),
        ("dvfs", "DVFS theo thời gian"),
        ("avg_temp", "Nhiệt độ trung bình theo thời gian"),
        ("active_hosts", "Số host active theo thời gian"),
        ("migrations", "Số migration theo thời gian"),
        ("power_it", "IT power theo thời gian"),
        ("off_hosts", "Số host off theo thời gian"),
        ("sleep_hosts", "Số host sleep theo thời gian"),
        ("max_temp", "Nhiệt độ lớn nhất theo thời gian"),
        ("mean_host_age", "Tuổi thọ hao mòn trung bình theo thời gian"),
    ]
    for col, title in charts:
        plt.figure(figsize=(10, 4))
        plt.plot(df["step"], df[col])
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_{col}.png", dpi=150)
        plt.close()
    return df
