from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def trace_to_dataframe(trace: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(trace)


def export_core_plots(trace_df: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    saved = {}

    fig, ax = plt.subplots(figsize=(10, 4))
    trace_df[["demand", "dvfs"]].plot(ax=ax)
    ax.set_title("Demand and DVFS over time")
    ax.set_xlabel("timestep")
    ax.set_ylabel("normalized value")
    path = output / "demand_dvfs_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["demand_dvfs"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    trace_df[["it_power", "facility_power"]].plot(ax=ax)
    ax.set_title("IT power vs Facility power")
    ax.set_xlabel("timestep")
    ax.set_ylabel("Power (W)")
    path = output / "it_vs_facility_power.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["power"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    trace_df[["pue", "mean_temperature_c", "hardware_wear_index"]].plot(ax=ax)
    ax.set_title("PUE, temperature and hardware wear index")
    ax.set_xlabel("timestep")
    path = output / "pue_temperature_wear.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved["pue_temp_wear"] = str(path)

    return saved
