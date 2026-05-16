from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .energy_env import CloudEnergyEnv, EnvConfig
from .baselines import run_policy
from .evaluation import trace_to_dataframe, save_trace_artifacts


POLICY_ORDER = ["Fixed-Keep", "RoundRobin", "Threshold", "BestFit", "PPO"]


def policy_slug(policy_name: str) -> str:
    return (
        str(policy_name)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def compute_sla_metrics(trace_df: pd.DataFrame, threshold: float = 1e-8) -> dict[str, float | int]:
    
    #  Tính SLA metrics từ trace dataframe.
    #  SLA (Service Level Agreement – cam kết chất lượng dịch vụ):
    #  - sla_violation = 0 nghĩa là không vi phạm.
    #  - sla_violation > 0 nghĩa là thiếu capacity so với demand.
    
    if trace_df is None or trace_df.empty or "sla_violation" not in trace_df.columns:
        return {
            "avg_sla_violation": 0.0,
            "max_sla_violation": 0.0,
            "sla_violation_step_rate": 0.0,
            "sla_violation_steps": 0,
        }

    sla_values = trace_df["sla_violation"].astype(float)
    violation_steps = int((sla_values > threshold).sum())
    total_steps = int(len(sla_values))

    return {
        "avg_sla_violation": float(sla_values.mean()),
        "max_sla_violation": float(sla_values.max()),
        "sla_violation_step_rate": float(violation_steps / max(total_steps, 1)),
        "sla_violation_steps": violation_steps,
    }


def mean_col(trace_df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if trace_df is None or trace_df.empty or col not in trace_df.columns:
        return float(default)
    return float(trace_df[col].astype(float).mean())


def max_col(trace_df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if trace_df is None or trace_df.empty or col not in trace_df.columns:
        return float(default)
    return float(trace_df[col].astype(float).max())


def min_col(trace_df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if trace_df is None or trace_df.empty or col not in trace_df.columns:
        return float(default)
    return float(trace_df[col].astype(float).min())


def _ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def evaluate_policy_once(
    *,
    policy_name: str,
    policy: Any,
    workload: np.ndarray,
    config: EnvConfig,
    trace_dir: str | Path,
    figure_dir: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    
    #Chạy một policy trên CloudEnergyEnv, lưu trace CSV và các hình trace chi tiết.

    #Hàm này dùng chung cho PPO và baseline policies.

    trace_dir = Path(trace_dir)
    figure_dir = Path(figure_dir)
    _ensure_dirs(trace_dir, figure_dir)

    env_local = CloudEnergyEnv(workload=workload, config=config)
    metrics = run_policy(env_local, policy)

    prefix = policy_slug(policy_name)
    trace_df = trace_to_dataframe(env_local.trace)
    trace_df.to_csv(trace_dir / f"{prefix}_trace.csv", index=False)

    # Lưu thêm biểu đồ chi tiết từng policy vào figures/<policy_slug>/.
    save_trace_artifacts(env_local.trace, figure_dir / prefix, prefix=prefix)

    sla_metrics = compute_sla_metrics(trace_df)

    row = {
        "policy": policy_name,
        "total_reward": metrics.total_reward,
        "total_energy": metrics.total_energy,
        "total_it_energy": metrics.total_it_energy,
        "avg_power": metrics.avg_power,
        "avg_pue": metrics.avg_pue,

        # SLA (Service Level Agreement – cam kết chất lượng dịch vụ)
        "sla_rate": metrics.sla_rate,
        "avg_sla_violation": sla_metrics["avg_sla_violation"],
        "max_sla_violation": sla_metrics["max_sla_violation"],
        "sla_violation_step_rate": sla_metrics["sla_violation_step_rate"],
        "sla_violation_steps": sla_metrics["sla_violation_steps"],

        # Resource management (quản lý tài nguyên)
        "avg_active_hosts": metrics.avg_active_hosts,
        "avg_sleep_hosts": metrics.avg_sleep_hosts,
        "avg_off_hosts": metrics.avg_off_hosts,
        "avg_dvfs": mean_col(trace_df, "dvfs"),
        "min_dvfs": min_col(trace_df, "dvfs"),
        "max_dvfs": max_col(trace_df, "dvfs"),

        # Thermal / lifetime (nhiệt độ / tuổi thọ phần cứng)
        "avg_temp": metrics.avg_temp,
        "avg_max_temp": mean_col(trace_df, "max_temp"),
        "avg_mean_host_age": mean_col(trace_df, "mean_host_age"),

        # Migration / latency (di chuyển VM / độ trễ)
        "total_switches": metrics.total_switches,
        "total_migrations": metrics.total_migrations,
        "total_migration_cost": getattr(metrics, "total_migration_cost", 0.0),
        "avg_migration_cost_per_step": mean_col(trace_df, "migration_cost"),
        "avg_latency_penalty": mean_col(trace_df, "latency_penalty"),
        "max_latency_penalty": max_col(trace_df, "latency_penalty"),

        "trace_csv": str(trace_dir / f"{prefix}_trace.csv"),
    }
    return row, trace_df


def evaluate_policies(
    *,
    policies: list[tuple[str, Any]],
    workload: np.ndarray,
    config: EnvConfig,
    trace_dir: str | Path,
    figure_dir: str | Path,
    result_csv: str | Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Evaluate PPO + baselines, lưu evaluation_results.csv và trả về trace_dfs.
    """
    rows: list[dict[str, Any]] = []
    trace_dfs: dict[str, pd.DataFrame] = {}

    for name, policy in policies:
        row, trace_df = evaluate_policy_once(
            policy_name=name,
            policy=policy,
            workload=workload,
            config=config,
            trace_dir=trace_dir,
            figure_dir=figure_dir,
        )
        rows.append(row)
        trace_dfs[name] = trace_df

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["total_reward", "avg_sla_violation", "total_energy"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    result_csv = Path(result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(result_csv, index=False)
    return results_df, trace_dfs


def _ordered_policy_df(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    present_order = [p for p in POLICY_ORDER if p in set(df["policy"].astype(str))]
    remaining = [p for p in df["policy"].astype(str).tolist() if p not in present_order]
    order = present_order + remaining

    df["policy"] = pd.Categorical(df["policy"].astype(str), categories=order, ordered=True)
    df = df.sort_values("policy").reset_index(drop=True)
    df["sla_violation_step_rate_pct"] = df["sla_violation_step_rate"].astype(float) * 100.0
    return df


def compute_energy_saving(
    results_df: pd.DataFrame,
    baselines: list[str] | None = None,
) -> pd.DataFrame:
    baselines = baselines or ["Fixed-Keep", "RoundRobin", "Threshold", "BestFit"]
    energy_ref = results_df.set_index("policy")
    if "PPO" not in energy_ref.index:
        raise ValueError("Chưa có dòng PPO trong results_df.")

    ppo_energy = float(energy_ref.loc["PPO", "total_energy"])
    saving_rows = []
    for baseline_name in baselines:
        if baseline_name not in energy_ref.index:
            continue
        baseline_energy = float(energy_ref.loc[baseline_name, "total_energy"])
        saving_pct = (1.0 - ppo_energy / baseline_energy) * 100.0
        saving_rows.append({
            "baseline": baseline_name,
            "baseline_energy": baseline_energy,
            "ppo_energy": ppo_energy,
            "ppo_energy_saving_pct": saving_pct,
        })

    return pd.DataFrame(saving_rows).sort_values("ppo_energy_saving_pct", ascending=False)


def _pad_axis_for_labels(ax, orientation: str = "vertical", pad_ratio: float = 0.08) -> None:
    if orientation == "horizontal":
        x_min, x_max = ax.get_xlim()
        span = x_max - x_min if x_max != x_min else 1.0
        ax.set_xlim(x_min - span * pad_ratio, x_max + span * pad_ratio)
    else:
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - span * pad_ratio, y_max + span * pad_ratio)


def _add_bar_labels(ax, fmt: str = "{:.2f}", rotation: int = 0) -> None:
    for container in ax.containers:
        labels = []
        values = getattr(container, "datavalues", [])
        for value in values:
            try:
                labels.append(fmt.format(value))
            except Exception:
                labels.append(str(value))
        if hasattr(ax, "bar_label"):
            ax.bar_label(container, labels=labels, padding=2, fontsize=8, rotation=rotation)
            orientation = getattr(container, "orientation", "vertical")
            _pad_axis_for_labels(ax, orientation=orientation)
        else:
            for bar, label in zip(container, labels):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.text(x, y, label, ha="center", va="bottom", fontsize=8, rotation=rotation, clip_on=False)
            _pad_axis_for_labels(ax, orientation="vertical")


def _add_line_labels(
    ax,
    y_values: list[float] | np.ndarray,
    fmt: str = "{:.2f}",
    *,
    x_positions: list[int] | np.ndarray | None = None,
    offset: tuple[int, int] = (0, 3),
) -> None:
    if x_positions is None:
        x_positions = list(range(len(y_values)))
    for x, y in zip(x_positions, y_values):
        if y is None or (isinstance(y, float) and not np.isfinite(y)):
            continue
        try:
            label = fmt.format(y)
        except Exception:
            label = str(y)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=8,
        )


def _add_last_point_label(
    ax,
    x_values: list[float] | np.ndarray,
    y_values: list[float] | np.ndarray,
    fmt: str = "{:.2f}",
    *,
    offset: tuple[int, int] = (6, 0),
) -> None:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return
    finite_mask = np.isfinite(y_arr)
    if not finite_mask.any():
        return
    last_idx = int(np.where(finite_mask)[0][-1])
    x = float(x_arr[last_idx])
    y = float(y_arr[last_idx])
    try:
        label = fmt.format(y)
    except Exception:
        label = str(y)
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=offset,
        ha="left",
        va="center",
        fontsize=8,
    )


def _save_fig(fig, path: str | Path, show: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _save_simple_bar(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    path: str | Path,
    fmt: str = "{:.2f}",
    show: bool = False,
):
    if y_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    _add_bar_labels(ax, fmt=fmt)
    return _save_fig(fig, path, show=show)


def save_goal_check_figures(
    *,
    results_df: pd.DataFrame,
    saving_df: pd.DataFrame,
    experiment_label: str,
    figure_dir: str | Path,
    show_main: bool = True,
) -> dict[str, Path]:
    """
    Lưu biểu đồ tổng quan mục tiêu: energy, saving, SLA, PUE, temperature, migration.
    Chỉ hiển thị goal_check_overview.png, các biểu đồ chi tiết còn lại chỉ lưu file.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_df = _ordered_policy_df(results_df)
    outputs: dict[str, Path] = {}

    # Các biểu đồ chi tiết chỉ lưu.
    detail_specs = [
        ("total_energy", "Tổng năng lượng theo policy", "Tổng năng lượng mô phỏng", "compare_total_energy.png", "{:.0f}"),
        ("avg_pue", "PUE trung bình theo policy", "PUE trung bình", "compare_avg_pue.png", "{:.3f}"),
        ("avg_sla_violation", "SLA violation trung bình theo policy", "SLA violation trung bình", "compare_avg_sla_violation.png", "{:.5f}"),
        ("sla_violation_step_rate_pct", "Tỷ lệ timestep vi phạm SLA theo policy", "Timestep vi phạm SLA (%)", "compare_sla_violation_step_rate.png", "{:.2f}%"),
        ("avg_temp", "Nhiệt độ trung bình theo policy", "Nhiệt độ trung bình (°C)", "compare_avg_temp.png", "{:.2f}"),
        ("avg_dvfs", "DVFS trung bình theo policy", "DVFS trung bình", "compare_avg_dvfs.png", "{:.2f}"),
        ("avg_mean_host_age", "Hao mòn host trung bình theo policy", "Hao mòn host trung bình", "compare_avg_mean_host_age.png", "{:.3f}"),
        ("avg_latency_penalty", "Latency penalty trung bình theo policy", "Latency penalty trung bình", "compare_avg_latency_penalty.png", "{:.3f}"),
    ]
    for y_col, title_prefix, ylabel, filename, fmt in detail_specs:
        out = _save_simple_bar(
            plot_df,
            x_col="policy",
            y_col=y_col,
            title=f"{title_prefix} - {experiment_label}",
            ylabel=ylabel,
            path=figure_dir / filename,
            fmt=fmt,
            show=False,
        )
        if out:
            outputs[filename] = out

    # Host state stacked chi tiết.
    fig, ax = plt.subplots(figsize=(10, 5))
    policies = plot_df["policy"].astype(str)
    ax.bar(policies, plot_df["avg_active_hosts"], label="Active")
    ax.bar(policies, plot_df["avg_sleep_hosts"], bottom=plot_df["avg_active_hosts"], label="Sleep")
    ax.bar(
        policies,
        plot_df["avg_off_hosts"],
        bottom=plot_df["avg_active_hosts"] + plot_df["avg_sleep_hosts"],
        label="Off",
    )
    ax.set_title(f"Trạng thái host Active/Sleep/Off trung bình - {experiment_label}")
    ax.set_ylabel("Số host trung bình")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    _add_bar_labels(ax, fmt="{:.1f}")
    outputs["compare_host_state_stacked.png"] = _save_fig(fig, figure_dir / "compare_host_state_stacked.png", show=False)

    # Migration log-scale chi tiết.
    if "total_migrations" in plot_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        values = plot_df["total_migrations"].astype(float).clip(lower=1e-6)
        ax.barh(plot_df["policy"].astype(str), values)
        ax.set_xscale("log")
        ax.set_title(f"Tổng số VM migration theo policy - {experiment_label}")
        ax.set_xlabel("Tổng VM migration (log scale)")
        ax.grid(axis="x", alpha=0.25)
        _add_bar_labels(ax, fmt="{:.0f}")
        outputs["compare_total_migrations_log.png"] = _save_fig(fig, figure_dir / "compare_total_migrations_log.png", show=False)

    # Energy saving detail.
    if not saving_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(saving_df["baseline"].astype(str), saving_df["ppo_energy_saving_pct"].astype(float))
        ax.axhline(15, linestyle="--", linewidth=1, label="Mục tiêu 15%")
        ax.axhline(30, linestyle="--", linewidth=1, label="Mục tiêu 30%")
        ax.axhline(0, linestyle="-", linewidth=0.8)
        ax.set_title(f"Tỷ lệ tiết kiệm năng lượng của PPO - {experiment_label}")
        ax.set_ylabel("Energy saving (%)")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.25)
        _add_bar_labels(ax, fmt="{:.2f}%")
        outputs["ppo_energy_saving_vs_baselines.png"] = _save_fig(fig, figure_dir / "ppo_energy_saving_vs_baselines.png", show=False)

    # Figure chính hiển thị trong notebook.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    axes[0].bar(plot_df["policy"].astype(str), plot_df["total_energy"].astype(float))
    axes[0].set_title("Tổng năng lượng")
    axes[0].set_ylabel("energy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[0], fmt="{:.0f}")

    if not saving_df.empty:
        axes[1].bar(saving_df["baseline"].astype(str), saving_df["ppo_energy_saving_pct"].astype(float))
        axes[1].axhline(15, linestyle="--", linewidth=1, label="15%")
        axes[1].axhline(30, linestyle="--", linewidth=1, label="30%")
        axes[1].axhline(0, linewidth=0.8)
        axes[1].set_title("PPO energy saving")
        axes[1].set_ylabel("%")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(loc="best")
        axes[1].grid(axis="y", alpha=0.25)
        _add_bar_labels(axes[1], fmt="{:.2f}%")
    else:
        axes[1].axis("off")

    axes[2].bar(plot_df["policy"].astype(str), plot_df["avg_sla_violation"].astype(float))
    axes[2].set_title("SLA violation trung bình")
    axes[2].set_ylabel("avg SLA")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[2], fmt="{:.5f}")

    axes[3].bar(plot_df["policy"].astype(str), plot_df["avg_pue"].astype(float))
    axes[3].set_title("PUE trung bình")
    axes[3].set_ylabel("PUE")
    axes[3].tick_params(axis="x", rotation=20)
    axes[3].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[3], fmt="{:.3f}")

    axes[4].bar(plot_df["policy"].astype(str), plot_df["avg_temp"].astype(float))
    axes[4].set_title("Nhiệt độ trung bình")
    axes[4].set_ylabel("°C")
    axes[4].tick_params(axis="x", rotation=20)
    axes[4].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[4], fmt="{:.2f}")

    axes[5].bar(plot_df["policy"].astype(str), plot_df["total_migrations"].astype(float))
    axes[5].set_title("Tổng VM migration")
    axes[5].set_ylabel("migrations")
    axes[5].tick_params(axis="x", rotation=20)
    axes[5].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[5], fmt="{:.0f}")

    fig.suptitle(f"Tổng quan kiểm tra mục tiêu - {experiment_label}", y=1.02)
    outputs["goal_check_overview.png"] = _save_fig(fig, figure_dir / "goal_check_overview.png", show=show_main)
    return outputs


def _align_trace_lengths(trace_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    required = [df for df in trace_dfs.values() if df is not None and not df.empty]
    if not required:
        return trace_dfs
    min_len = min(len(df) for df in required)
    return {
        name: (df.iloc[:min_len].reset_index(drop=True) if df is not None and not df.empty else df)
        for name, df in trace_dfs.items()
    }


def save_monitoring_dashboard(
    *,
    trace_dfs: dict[str, pd.DataFrame],
    experiment_label: str,
    summary_dir: str | Path,
    figure_dir: str | Path,
    show_main: bool = True,
    rolling_window: int = 24,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """
    Lưu ppo_monitoring_dashboard_timeseries.csv và dashboard chính:
    demand, active hosts, Active/Sleep/Off, power, cumulative energy saving.
    """
    summary_dir = Path(summary_dir)
    figure_dir = Path(figure_dir)
    _ensure_dirs(summary_dir, figure_dir)

    if "PPO" not in trace_dfs or trace_dfs["PPO"].empty:
        raise ValueError("trace_dfs cần có trace PPO không rỗng.")

    aligned = _align_trace_lengths(trace_dfs)
    ppo_trace = aligned["PPO"].copy()
    fixed_trace = aligned.get("Fixed-Keep", pd.DataFrame()).copy()
    rr_trace = aligned.get("RoundRobin", pd.DataFrame()).copy()
    bf_trace = aligned.get("BestFit", pd.DataFrame()).copy()

    def smooth_series(s: pd.Series, window: int = rolling_window) -> pd.Series:
        return s.astype(float).rolling(window=window, min_periods=1).mean()

    dashboard_df = ppo_trace.copy()
    dashboard_df = dashboard_df.rename(columns={"power_total": "ppo_power_total"})

    if not fixed_trace.empty:
        dashboard_df["fixed_power_total"] = fixed_trace["power_total"].to_numpy()
        dashboard_df["energy_saved_vs_fixed"] = dashboard_df["fixed_power_total"] - dashboard_df["ppo_power_total"]
        dashboard_df["cumulative_energy_saved_vs_fixed"] = dashboard_df["energy_saved_vs_fixed"].cumsum()
    else:
        dashboard_df["energy_saved_vs_fixed"] = np.nan
        dashboard_df["cumulative_energy_saved_vs_fixed"] = np.nan

    if not rr_trace.empty:
        dashboard_df["roundrobin_power_total"] = rr_trace["power_total"].to_numpy()
        dashboard_df["energy_saved_vs_roundrobin"] = dashboard_df["roundrobin_power_total"] - dashboard_df["ppo_power_total"]
        dashboard_df["cumulative_energy_saved_vs_roundrobin"] = dashboard_df["energy_saved_vs_roundrobin"].cumsum()
    else:
        dashboard_df["energy_saved_vs_roundrobin"] = np.nan
        dashboard_df["cumulative_energy_saved_vs_roundrobin"] = np.nan

    if not bf_trace.empty:
        dashboard_df["bestfit_power_total"] = bf_trace["power_total"].to_numpy()
        dashboard_df["energy_saved_vs_bestfit"] = dashboard_df["bestfit_power_total"] - dashboard_df["ppo_power_total"]
        dashboard_df["cumulative_energy_saved_vs_bestfit"] = dashboard_df["energy_saved_vs_bestfit"].cumsum()
    else:
        dashboard_df["energy_saved_vs_bestfit"] = np.nan
        dashboard_df["cumulative_energy_saved_vs_bestfit"] = np.nan

    dashboard_df.to_csv(summary_dir / "ppo_monitoring_dashboard_timeseries.csv", index=False)

    outputs: dict[str, Path] = {}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax0 = axes[0]
    steps = ppo_trace["step"].astype(float).to_numpy()
    demand_vals = ppo_trace["demand"].astype(float).to_numpy()
    ax0.plot(steps, demand_vals, label="demand", linewidth=1.5)
    ax0.set_ylabel("demand")
    ax0.set_title("Tải hệ thống và số host Active theo thời gian")
    ax0.grid(alpha=0.25)
    ax0b = ax0.twinx()
    active_vals = smooth_series(ppo_trace["active_hosts"]).to_numpy()
    ax0b.plot(steps, active_vals, label="Active hosts", linestyle="--", alpha=0.8)
    ax0b.set_ylabel("Active hosts")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    ax0.legend(lines0 + lines0b, labels0 + labels0b, loc="best")
    _add_last_point_label(ax0, steps, demand_vals, fmt="{:.3f}", offset=(6, 0))
    _add_last_point_label(ax0b, steps, active_vals, fmt="{:.1f}", offset=(6, 0))

    active_hosts_vals = smooth_series(ppo_trace["active_hosts"]).to_numpy()
    sleep_hosts_vals = smooth_series(ppo_trace["sleep_hosts"]).to_numpy()
    off_hosts_vals = smooth_series(ppo_trace["off_hosts"]).to_numpy()
    axes[1].plot(steps, active_hosts_vals, label="Active")
    axes[1].plot(steps, sleep_hosts_vals, label="Sleep")
    axes[1].plot(steps, off_hosts_vals, label="Off")
    axes[1].set_ylabel("Số host")
    axes[1].set_title("Trạng thái host Active/Sleep/Off của PPO")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    _add_last_point_label(axes[1], steps, active_hosts_vals, fmt="{:.1f}", offset=(6, 6))
    _add_last_point_label(axes[1], steps, sleep_hosts_vals, fmt="{:.1f}", offset=(6, -6))
    _add_last_point_label(axes[1], steps, off_hosts_vals, fmt="{:.1f}", offset=(6, 14))

    ppo_power_vals = smooth_series(ppo_trace["power_total"]).to_numpy()
    axes[2].plot(steps, ppo_power_vals, label="PPO")
    if not rr_trace.empty:
        rr_steps = rr_trace["step"].astype(float).to_numpy()
        rr_power_vals = smooth_series(rr_trace["power_total"]).to_numpy()
        axes[2].plot(rr_steps, rr_power_vals, label="RoundRobin", alpha=0.75)
    if not bf_trace.empty:
        bf_steps = bf_trace["step"].astype(float).to_numpy()
        bf_power_vals = smooth_series(bf_trace["power_total"]).to_numpy()
        axes[2].plot(bf_steps, bf_power_vals, label="BestFit", alpha=0.75)
    axes[2].set_ylabel("Power")
    axes[2].set_title("Power theo thời gian")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    _add_last_point_label(axes[2], steps, ppo_power_vals, fmt="{:.0f}", offset=(6, 0))
    if not rr_trace.empty:
        _add_last_point_label(axes[2], rr_steps, rr_power_vals, fmt="{:.0f}", offset=(6, -10))
    if not bf_trace.empty:
        _add_last_point_label(axes[2], bf_steps, bf_power_vals, fmt="{:.0f}", offset=(6, 10))

    y_save = "cumulative_energy_saved_vs_roundrobin"
    if dashboard_df[y_save].notna().any():
        save_rr_vals = dashboard_df[y_save].astype(float).to_numpy()
        axes[3].plot(steps, save_rr_vals, label="Cumulative saving vs RoundRobin")
    y_save_bf = "cumulative_energy_saved_vs_bestfit"
    if dashboard_df[y_save_bf].notna().any():
        save_bf_vals = dashboard_df[y_save_bf].astype(float).to_numpy()
        axes[3].plot(steps, save_bf_vals, label="Cumulative saving vs BestFit", alpha=0.8)
    axes[3].axhline(0, linestyle="--", linewidth=1)
    axes[3].set_ylabel("Energy saving")
    axes[3].set_xlabel("step")
    axes[3].set_title("Năng lượng tiết kiệm tích lũy theo thời gian")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.25)
    if dashboard_df[y_save].notna().any():
        _add_last_point_label(axes[3], steps, save_rr_vals, fmt="{:.0f}", offset=(6, 0))
    if dashboard_df[y_save_bf].notna().any():
        _add_last_point_label(axes[3], steps, save_bf_vals, fmt="{:.0f}", offset=(6, -10))

    fig.suptitle(f"Monitoring dashboard chính - {experiment_label}", y=1.01)
    outputs["monitoring_main_dashboard.png"] = _save_fig(fig, figure_dir / "monitoring_main_dashboard.png", show=show_main)
    return dashboard_df, outputs


def save_relationship_outputs(
    *,
    trace_dfs: dict[str, pd.DataFrame],
    summary_dir: str | Path,
    figure_dir: str | Path,
    experiment_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    summary_dir = Path(summary_dir)
    figure_dir = Path(figure_dir)
    _ensure_dirs(summary_dir, figure_dir)

    aligned = _align_trace_lengths(trace_dfs)
    ppo_trace = aligned["PPO"].copy()
    fixed_trace = aligned.get("Fixed-Keep", pd.DataFrame()).copy()
    rr_trace = aligned.get("RoundRobin", pd.DataFrame()).copy()
    bf_trace = aligned.get("BestFit", pd.DataFrame()).copy()

    relationship_df = pd.DataFrame({
        "step": ppo_trace["step"],
        "demand": ppo_trace["demand"],
        "active_hosts": ppo_trace["active_hosts"],
        "sleep_hosts": ppo_trace["sleep_hosts"],
        "off_hosts": ppo_trace["off_hosts"],
        "dvfs": ppo_trace["dvfs"],
        "ppo_power_total": ppo_trace["power_total"],
        "sla_violation": ppo_trace["sla_violation"],
        "avg_temp": ppo_trace["avg_temp"],
    })

    if not fixed_trace.empty:
        relationship_df["fixed_power_total"] = fixed_trace["power_total"].to_numpy()
        relationship_df["energy_saved_vs_fixed"] = relationship_df["fixed_power_total"] - relationship_df["ppo_power_total"]
    if not rr_trace.empty:
        relationship_df["roundrobin_power_total"] = rr_trace["power_total"].to_numpy()
        relationship_df["energy_saved_vs_roundrobin"] = relationship_df["roundrobin_power_total"] - relationship_df["ppo_power_total"]
    if not bf_trace.empty:
        relationship_df["bestfit_power_total"] = bf_trace["power_total"].to_numpy()
        relationship_df["energy_saved_vs_bestfit"] = relationship_df["bestfit_power_total"] - relationship_df["ppo_power_total"]

    relationship_df["phase"] = pd.cut(
        relationship_df["demand"],
        bins=[0.0, 0.40, 0.75, 2.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    relationship_df.to_csv(summary_dir / "demand_active_energy_relationship.csv", index=False)

    agg_spec = dict(
        count=("demand", "count"),
        demand_mean=("demand", "mean"),
        active_hosts_mean=("active_hosts", "mean"),
        sleep_hosts_mean=("sleep_hosts", "mean"),
        off_hosts_mean=("off_hosts", "mean"),
        dvfs_mean=("dvfs", "mean"),
        ppo_power_mean=("ppo_power_total", "mean"),
        sla_violation_mean=("sla_violation", "mean"),
        avg_temp_mean=("avg_temp", "mean"),
    )
    if "energy_saved_vs_fixed" in relationship_df.columns:
        agg_spec["energy_saved_vs_fixed_mean"] = ("energy_saved_vs_fixed", "mean")
    if "energy_saved_vs_roundrobin" in relationship_df.columns:
        agg_spec["energy_saved_vs_roundrobin_mean"] = ("energy_saved_vs_roundrobin", "mean")
    if "energy_saved_vs_bestfit" in relationship_df.columns:
        agg_spec["energy_saved_vs_bestfit_mean"] = ("energy_saved_vs_bestfit", "mean")

    relationship_by_phase = relationship_df.groupby("phase", observed=False).agg(**agg_spec).reset_index()
    relationship_by_phase.to_csv(summary_dir / "demand_active_energy_relationship_by_phase.csv", index=False)

    outputs: dict[str, Path] = {}

    def save_hexbin_chart(x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, filename: str):
        if y_col not in relationship_df.columns:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        hb = ax.hexbin(
            relationship_df[x_col].astype(float),
            relationship_df[y_col].astype(float),
            gridsize=35,
            mincnt=1,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
        fig.colorbar(hb, ax=ax, label="Số điểm")
        outputs[filename] = _save_fig(fig, figure_dir / filename, show=False)

    save_hexbin_chart(
        "demand", "active_hosts",
        f"Mật độ quan hệ demand và số host Active - {experiment_label}",
        "demand", "active_hosts",
        "relationship_demand_active_hosts_hexbin.png",
    )
    save_hexbin_chart(
        "demand", "ppo_power_total",
        f"Mật độ quan hệ demand và PPO total power - {experiment_label}",
        "demand", "PPO power_total",
        "relationship_demand_power_hexbin.png",
    )
    if "energy_saved_vs_roundrobin" in relationship_df.columns:
        save_hexbin_chart(
            "demand", "energy_saved_vs_roundrobin",
            f"Mật độ quan hệ demand và energy saving so với RoundRobin - {experiment_label}",
            "demand", "energy_saved_vs_roundrobin",
            "relationship_demand_energy_saved_vs_roundrobin_hexbin.png",
        )

    return relationship_df, relationship_by_phase, outputs


def save_phase_analysis(
    *,
    trace_dfs: dict[str, pd.DataFrame],
    summary_dir: str | Path,
    figure_dir: str | Path,
    experiment_label: str,
    show_main: bool = True,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    summary_dir = Path(summary_dir)
    figure_dir = Path(figure_dir)
    _ensure_dirs(summary_dir, figure_dir)

    ppo_trace = trace_dfs["PPO"].copy()
    ppo_phase_df = ppo_trace.copy()
    ppo_phase_df["phase"] = pd.cut(
        ppo_phase_df["demand"],
        bins=[0.0, 0.40, 0.75, 2.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    phase_metrics = ppo_phase_df.groupby("phase", observed=False).agg(
        count=("demand", "count"),
        demand_mean=("demand", "mean"),
        power_mean=("power_total", "mean"),
        pue_mean=("pue", "mean"),
        sla_mean=("sla_violation", "mean"),
        sla_max=("sla_violation", "max"),
        active_hosts_mean=("active_hosts", "mean"),
        sleep_hosts_mean=("sleep_hosts", "mean"),
        off_hosts_mean=("off_hosts", "mean"),
        dvfs_mean=("dvfs", "mean"),
        avg_temp_mean=("avg_temp", "mean"),
        max_temp_mean=("max_temp", "mean"),
        latency_penalty_mean=("latency_penalty", "mean"),
        migrations_sum=("migrations", "sum"),
        migration_cost_sum=("migration_cost", "sum"),
    ).reset_index()

    phase_metrics.to_csv(summary_dir / "ppo_phase_analysis.csv", index=False)

    phase_order = ["low", "medium", "high"]
    phase_plot = phase_metrics.copy()
    phase_plot["phase"] = pd.Categorical(phase_plot["phase"].astype(str), categories=phase_order, ordered=True)
    phase_plot = phase_plot.sort_values("phase")

    outputs: dict[str, Path] = {}

    def save_phase_line(y_col: str, ylabel: str, title: str, filename: str, fmt: str = "{:.3f}"):
        fig, ax = plt.subplots(figsize=(8, 5))
        y_vals = phase_plot[y_col].astype(float).to_numpy()
        ax.plot(phase_plot["phase"].astype(str), y_vals, marker="o")
        ax.set_title(title)
        ax.set_xlabel("phase")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        _add_line_labels(ax, y_vals, fmt=fmt)
        outputs[filename] = _save_fig(fig, figure_dir / filename, show=False)

    for y_col, ylabel, title, filename, fmt in [
        ("power_mean", "Power trung bình", f"PPO power theo demand phase - {experiment_label}", "ppo_power_by_phase.png", "{:.2f}"),
        ("pue_mean", "PUE trung bình", f"PPO PUE theo demand phase - {experiment_label}", "ppo_pue_by_phase.png", "{:.3f}"),
        ("sla_mean", "SLA violation trung bình", f"PPO SLA violation theo demand phase - {experiment_label}", "ppo_sla_by_phase.png", "{:.5f}"),
        ("avg_temp_mean", "Nhiệt độ trung bình (°C)", f"PPO nhiệt độ theo demand phase - {experiment_label}", "ppo_avg_temp_by_phase.png", "{:.2f}"),
    ]:
        if y_col in phase_plot.columns:
            save_phase_line(y_col, ylabel, title, filename, fmt=fmt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    power_vals = phase_plot["power_mean"].astype(float).to_numpy()
    axes[0].plot(phase_plot["phase"].astype(str), power_vals, marker="o")
    axes[0].set_title("Power trung bình theo demand phase")
    axes[0].set_ylabel("power_mean")
    axes[0].grid(alpha=0.25)
    _add_line_labels(axes[0], power_vals, fmt="{:.2f}")

    phase_labels = phase_plot["phase"].astype(str)
    axes[1].bar(phase_labels, phase_plot["active_hosts_mean"], label="Active")
    axes[1].bar(phase_labels, phase_plot["sleep_hosts_mean"], bottom=phase_plot["active_hosts_mean"], label="Sleep")
    axes[1].bar(
        phase_labels,
        phase_plot["off_hosts_mean"],
        bottom=phase_plot["active_hosts_mean"] + phase_plot["sleep_hosts_mean"],
        label="Off",
    )
    axes[1].set_title("Host state Active/Sleep/Off theo phase")
    axes[1].set_ylabel("Số host trung bình")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    _add_bar_labels(axes[1], fmt="{:.1f}")

    dvfs_vals = phase_plot["dvfs_mean"].astype(float).to_numpy()
    axes[2].plot(phase_plot["phase"].astype(str), dvfs_vals, marker="o")
    axes[2].set_title("DVFS trung bình theo demand phase")
    axes[2].set_ylabel("DVFS")
    axes[2].grid(alpha=0.25)
    _add_line_labels(axes[2], dvfs_vals, fmt="{:.3f}")

    sla_mean_vals = phase_plot["sla_mean"].astype(float).to_numpy()
    sla_max_vals = phase_plot["sla_max"].astype(float).to_numpy()
    axes[3].plot(phase_plot["phase"].astype(str), sla_mean_vals, marker="o", label="SLA mean")
    axes[3].plot(phase_plot["phase"].astype(str), sla_max_vals, marker="o", label="SLA max", alpha=0.8)
    axes[3].set_title("SLA violation theo demand phase")
    axes[3].set_ylabel("SLA violation")
    axes[3].legend(loc="best")
    axes[3].grid(alpha=0.25)
    _add_line_labels(axes[3], sla_mean_vals, fmt="{:.5f}", offset=(0, 3))
    _add_line_labels(axes[3], sla_max_vals, fmt="{:.5f}", offset=(0, -10))

    fig.suptitle(f"PPO theo low / medium / high demand - {experiment_label}", y=1.02)
    outputs["ppo_phase_goal_check.png"] = _save_fig(fig, figure_dir / "ppo_phase_goal_check.png", show=show_main)

    return phase_metrics, outputs


def save_artifact_index(
    *,
    figure_dir: str | Path,
    summary_dir: str | Path,
    trace_dir: str | Path,
) -> pd.DataFrame:
    figure_dir = Path(figure_dir)
    summary_dir = Path(summary_dir)
    trace_dir = Path(trace_dir)

    figure_files = sorted([p.name for p in figure_dir.glob("*.png")])
    summary_files = sorted([p.name for p in summary_dir.glob("*.csv")])
    trace_files = sorted([p.name for p in trace_dir.glob("*.csv")])

    artifact_index = pd.DataFrame({
        "category": (["figure"] * len(figure_files)) + (["summary"] * len(summary_files)) + (["trace"] * len(trace_files)),
        "filename": figure_files + summary_files + trace_files,
    })
    artifact_index.to_csv(summary_dir / "artifact_index.csv", index=False)
    return artifact_index


def create_experiment_report(
    *,
    results_df: pd.DataFrame,
    trace_dfs: dict[str, pd.DataFrame],
    experiment_label: str,
    summary_dir: str | Path,
    figure_dir: str | Path,
    trace_dir: str | Path,
    saving_csv: str | Path,
    show_main_figures: bool = True,
) -> dict[str, Any]:
    
    # Tạo toàn bộ summary/figure cần cho Notebook 02/03.

    # Hàm này thay phần code dài trong notebook:
    # - energy saving
    # - biểu đồ mục tiêu
    # - monitoring dashboard
    # - demand-active-energy relationship
    # - PPO phase analysis
    # - artifact index
    
    summary_dir = Path(summary_dir)
    figure_dir = Path(figure_dir)
    trace_dir = Path(trace_dir)
    _ensure_dirs(summary_dir, figure_dir, trace_dir)

    saving_df = compute_energy_saving(results_df)
    saving_csv = Path(saving_csv)
    saving_df.to_csv(saving_csv, index=False)

    plot_df = _ordered_policy_df(results_df)
    plot_df.to_csv(summary_dir / "evaluation_results_by_policy.csv", index=False)

    goal_figures = save_goal_check_figures(
        results_df=results_df,
        saving_df=saving_df,
        experiment_label=experiment_label,
        figure_dir=figure_dir,
        show_main=show_main_figures,
    )

    dashboard_df, dashboard_figures = save_monitoring_dashboard(
        trace_dfs=trace_dfs,
        experiment_label=experiment_label,
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        show_main=show_main_figures,
    )

    relationship_df, relationship_by_phase, relationship_figures = save_relationship_outputs(
        trace_dfs=trace_dfs,
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        experiment_label=experiment_label,
    )

    phase_metrics, phase_figures = save_phase_analysis(
        trace_dfs=trace_dfs,
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        experiment_label=experiment_label,
        show_main=show_main_figures,
    )

    artifact_index = save_artifact_index(
        figure_dir=figure_dir,
        summary_dir=summary_dir,
        trace_dir=trace_dir,
    )

    return {
        "saving_df": saving_df,
        "plot_df": plot_df,
        "dashboard_df": dashboard_df,
        "relationship_df": relationship_df,
        "relationship_by_phase": relationship_by_phase,
        "phase_metrics": phase_metrics,
        "artifact_index": artifact_index,
        "figures": {
            **goal_figures,
            **dashboard_figures,
            **relationship_figures,
            **phase_figures,
        },
    }
