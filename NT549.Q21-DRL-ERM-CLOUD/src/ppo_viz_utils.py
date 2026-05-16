"""
ppo_viz_utils.py

Các hàm vẽ biểu đồ và chẩn đoán PPO cho đồ án DRL Cloud Energy Optimization.

Mục tiêu:
- Đưa các cell biểu đồ ra khỏi notebook để notebook gọn hơn.
- Giữ phần train/evaluation trong notebook.
- Các hàm chỉ phụ thuộc vào DataFrame đầu vào và thư mục output.

Cách dùng trong notebook:
    from src.ppo_viz_utils import (
        load_tensorboard_scalars,
        plot_learning_convergence,
        plot_reward_curve,
        plot_goal_overview,
        plot_ppo_behavior_dashboard,
        diagnose_ppo_behavior,
        plot_phase_analysis,
        plot_reward_groups,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_df(df: pd.DataFrame) -> None:
    """Display đẹp trong notebook nếu IPython có sẵn; fallback sang print."""
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df)


def _get_x(trace: pd.DataFrame) -> pd.Series | np.ndarray:
    if "step" in trace.columns:
        return trace["step"]
    return np.arange(len(trace))


def _sort_phase_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "phase" in out.columns:
        out = out.set_index("phase", drop=True)

    phase_order = ["low", "medium", "high"]
    existing = [p for p in phase_order if p in out.index]
    remaining = [p for p in out.index if p not in phase_order]
    return out.loc[existing + remaining]


def load_tensorboard_scalars(tb_root: Path | str) -> pd.DataFrame:
    """
    Đọc TensorBoard scalar event files thành DataFrame.

    Returns columns:
        file, tag, step, value
    """
    tb_root = Path(tb_root)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print("Không đọc được TensorBoard vì thiếu package tensorboard:", e)
        return pd.DataFrame(columns=["file", "tag", "step", "value"])

    rows = []
    event_files = list(tb_root.rglob("events.out.tfevents.*"))
    print("Found event files:", len(event_files))

    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                for ev in ea.Scalars(tag):
                    rows.append(
                        {
                            "file": str(event_file),
                            "tag": tag,
                            "step": ev.step,
                            "value": ev.value,
                        }
                    )
        except Exception as e:
            print("Skip event file:", event_file, e)

    return pd.DataFrame(rows, columns=["file", "tag", "step", "value"])


def _plot_scalar(ax, df: pd.DataFrame, tag: str, title: str, ylabel: Optional[str] = None) -> None:
    sub = df[df["tag"] == tag].sort_values("step") if "tag" in df.columns else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, f"Thiếu dữ liệu\n{tag}", ha="center", va="center")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return

    ax.plot(sub["step"], sub["value"])
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_learning_convergence(
    scalars_df: pd.DataFrame,
    fig_dir: Path | str,
    version_name: str = "PPO",
    filename: str = "01_learning_reward_va_hoi_tu.png",
) -> Path:
    """Vẽ 2x3: reward, value loss, entropy, KL, clip fraction, explained variance."""
    fig_dir = _ensure_dir(fig_dir)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    plots = [
        ("rollout/ep_rew_mean", "Learning reward curve", "Reward trung bình"),
        ("train/value_loss", "Value loss", None),
        ("train/entropy_loss", "Entropy loss", None),
        ("train/approx_kl", "Approx KL", None),
        ("train/clip_fraction", "Clip fraction", None),
        ("train/explained_variance", "Explained variance", None),
    ]

    for ax, (tag, title, ylabel) in zip(axes.ravel(), plots):
        _plot_scalar(ax, scalars_df, tag, title, ylabel=ylabel)

    fig.suptitle(f"Hội tụ huấn luyện {version_name} - reward, loss, entropy, KL", y=1.02)
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path


def plot_reward_curve(
    scalars_df: pd.DataFrame,
    fig_dir: Path | str,
    reward_tag: str = "rollout/ep_rew_mean",
    filename: str = "01a_learning_reward_mean.png",
) -> Optional[Path]:
    """Vẽ riêng learning reward curve cho báo cáo."""
    fig_dir = _ensure_dir(fig_dir)

    reward_sub = scalars_df[scalars_df["tag"] == reward_tag].sort_values("step").copy()
    if reward_sub.empty:
        print(f"Không có {reward_tag} trong TensorBoard scalars.")
        return None

    reward_sub["rolling_mean"] = reward_sub["value"].rolling(10, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(reward_sub["step"], reward_sub["value"], alpha=0.35, label="raw")
    ax.plot(reward_sub["step"], reward_sub["rolling_mean"], linewidth=2.0, label="rolling mean")
    ax.set_title(f"Learning Reward — {reward_tag}")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path


def plot_goal_overview(
    results_df: pd.DataFrame,
    energy_saving_df: pd.DataFrame,
    fig_dir: Path | str,
    ppo_policy: str = "PPO_v8_1",
    ppo_label: str = "PPO",
    title: str = "Tổng quan kiểm tra mục tiêu - Experiment B - Azure-derived Multiphase Workload",
    filename: str = "02_tong_quan_kiem_tra_muc_tieu.png",
) -> Path:
    """Vẽ dashboard 2x3 tổng quan energy, saving, SLA, PUE, temperature, migration."""
    fig_dir = _ensure_dir(fig_dir)

    plot_df = results_df.copy()
    plot_df["policy_label"] = plot_df["policy"].replace({ppo_policy: ppo_label})

    energy_order = ["Fixed-Keep", "RoundRobin", "Threshold", "BestFit", ppo_label, "RandomValid"]
    energy_plot = plot_df.set_index("policy_label").reindex(energy_order)

    saving_map = {row["baseline"]: row["energy_saving_pct"] for _, row in energy_saving_df.iterrows()}
    saving_plot = pd.Series(
        {
            "RoundRobin": saving_map.get("RoundRobin", np.nan),
            "Threshold": saving_map.get("Threshold", np.nan),
            "BestFit": saving_map.get("BestFit", np.nan),
            "Fixed-Keep": saving_map.get("Fixed-Keep", np.nan),
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def annotate_bars(ax, fmt: str, fontsize: int = 9) -> None:
        for p in ax.patches:
            h = p.get_height()
            if pd.isna(h):
                continue
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=fontsize)

    # 1. Total energy
    energy_plot["total_energy"].plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Tổng năng lượng")
    axes[0, 0].set_ylabel("energy")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[0, 0], "{:.0f}")

    # 2. Energy saving
    saving_plot.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("PPO energy saving")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].axhline(15, linestyle="--", linewidth=1, label="15%")
    axes[0, 1].axhline(30, linestyle="--", linewidth=1, label="30%")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[0, 1], "{:.2f}%")

    # 3. SLA
    energy_plot["avg_sla"].plot(kind="bar", ax=axes[0, 2])
    axes[0, 2].set_title("SLA violation trung bình")
    axes[0, 2].set_ylabel("avg SLA")
    axes[0, 2].tick_params(axis="x", rotation=20)
    axes[0, 2].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[0, 2], "{:.5f}", fontsize=8)

    # 4. PUE
    energy_plot["avg_pue"].plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("PUE trung bình")
    axes[1, 0].set_ylabel("PUE")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[1, 0], "{:.3f}")

    # 5. Temperature
    energy_plot["avg_temp"].plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Nhiệt độ trung bình")
    axes[1, 1].set_ylabel("°C")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[1, 1], "{:.2f}")

    # 6. Migration
    energy_plot["total_migrations"].plot(kind="bar", ax=axes[1, 2])
    axes[1, 2].set_title("Tổng VM migration")
    axes[1, 2].set_ylabel("migrations")
    axes[1, 2].tick_params(axis="x", rotation=20)
    axes[1, 2].grid(True, axis="y", alpha=0.3)
    annotate_bars(axes[1, 2], "{:.0f}", fontsize=8)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path


def plot_ppo_behavior_dashboard(
    trace: pd.DataFrame,
    fig_dir: Path | str,
    policy_name: str = "PPO",
    filename: str = "03_hanh_vi_agent_ppo.png",
) -> Path:
    """Vẽ dashboard hành vi agent PPO trên full trace."""
    fig_dir = _ensure_dir(fig_dir)
    df = trace.copy()
    x = _get_x(df)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    if "demand" in df.columns:
        axes[0, 0].plot(x, df["demand"], label="Demand")
    if "active_hosts" in df.columns:
        axes[0, 0].plot(x, df["active_hosts"], label="Active hosts")
    axes[0, 0].set_title("Demand và số host active")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for col, label in [("active_hosts", "Active"), ("sleep_hosts", "Sleep"), ("off_hosts", "Off")]:
        if col in df.columns:
            axes[0, 1].plot(x, df[col], label=label)
    axes[0, 1].set_title("Trạng thái host theo thời gian")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if "dvfs" in df.columns:
        axes[1, 0].plot(x, df["dvfs"], label="DVFS")
    if "sla_violation" in df.columns:
        axes[1, 0].plot(x, df["sla_violation"], label="SLA")
    axes[1, 0].set_title("DVFS và SLA violation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    action_col = "action_name" if "action_name" in df.columns else "action"
    if action_col in df.columns:
        action_counts = df[action_col].value_counts().sort_values(ascending=False)
        action_counts.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Phân bố action của PPO")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Hành vi của agent {policy_name} trên toàn workload", y=1.02)
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path


def diagnose_ppo_behavior(
    trace: pd.DataFrame,
    out_dir: Path | str,
    policy_name: str = "PPO",
    max_hosts: int = 8,
    filename: str = "03b_ppo_behavior_diagnostic.png",
) -> dict[str, pd.DataFrame | Path | float | int]:
    """
    In bảng chẩn đoán nhanh và vẽ figure compact.
    Trả về dict gồm action_counts, phase_diag, fig_path, keep_pct, distinct_actions.
    """
    out_dir = _ensure_dir(out_dir)
    trace = trace.copy()
    print(f"Trace {policy_name}: {len(trace)} timesteps")

    core_cols = [
        "demand",
        "phase",
        "action_name",
        "active_hosts",
        "sleep_hosts",
        "off_hosts",
        "dvfs",
        "power_total",
        "pue",
        "avg_temp",
        "max_temp",
        "sla_violation",
        "migrations",
    ]
    core_cols = [c for c in core_cols if c in trace.columns]

    if core_cols:
        print("Summary statistics:")
        _display_df(trace[core_cols].describe(include="all").T)

    action_counts = pd.DataFrame()
    keep_pct = 0.0
    distinct_actions = 0
    action_col = "action_name" if "action_name" in trace.columns else "action"

    if action_col in trace.columns:
        action_counts = trace[action_col].value_counts().rename_axis("action_name").reset_index(name="count")
        action_counts["pct"] = action_counts["count"] / max(len(trace), 1) * 100
        print("Action distribution:")
        _display_df(action_counts)

        if (action_counts["action_name"] == "KEEP").any():
            keep_pct = float(action_counts.loc[action_counts["action_name"] == "KEEP", "pct"].iloc[0])
        distinct_actions = int(action_counts.shape[0])
        power_off_count = int((trace[action_col] == "POWER_OFF_ONE").sum())

        print(f"KEEP rate = {keep_pct:.2f}%")
        print(f"Distinct actions = {distinct_actions}")
        print(f"POWER_OFF_ONE count = {power_off_count}")

        if keep_pct > 95:
            print("[WARN] KEEP quá cao: agent có dấu hiệu bảo thủ.")
        elif keep_pct > 90:
            print("[NOTE] KEEP hơi cao, cần xem phase behavior trước khi kết luận.")
        else:
            print("[OK] KEEP không quá cao.")

        action_counts.to_csv(out_dir / "ppo_action_distribution.csv", index=False)

    phase_diag = pd.DataFrame()
    required_phase_cols = {
        "phase",
        "demand",
        "active_hosts",
        "sleep_hosts",
        "off_hosts",
        "dvfs",
        "sla_violation",
        "avg_temp",
        "pue",
        "migrations",
    }
    if required_phase_cols.issubset(trace.columns):
        phase_diag = (
            trace.groupby("phase")
            .agg(
                steps=("phase", "size"),
                demand_mean=("demand", "mean"),
                active_mean=("active_hosts", "mean"),
                sleep_mean=("sleep_hosts", "mean"),
                off_mean=("off_hosts", "mean"),
                dvfs_mean=("dvfs", "mean"),
                sla_mean=("sla_violation", "mean"),
                pue_mean=("pue", "mean"),
                temp_mean=("avg_temp", "mean"),
                migrations_sum=("migrations", "sum"),
            )
            .reset_index()
        )
        phase_diag["phase"] = pd.Categorical(phase_diag["phase"], categories=["low", "medium", "high"], ordered=True)
        phase_diag = phase_diag.sort_values("phase")

        print("Phase diagnostic:")
        _display_df(phase_diag)

        out_csv = out_dir / "ppo_phase_diagnostic_v2style.csv"
        phase_diag.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    x = _get_x(trace)

    if action_col in trace.columns:
        trace[action_col].value_counts().plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Action distribution")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    for col, label in [("active_hosts", "Active"), ("sleep_hosts", "Sleep"), ("off_hosts", "Off")]:
        if col in trace.columns:
            axes[0, 1].plot(x, trace[col], label=label)
    axes[0, 1].set_title("Host states over time")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Number of hosts")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if "demand" in trace.columns:
        axes[1, 0].plot(x, trace["demand"], label="demand")
    if "active_hosts" in trace.columns:
        axes[1, 0].plot(x, trace["active_hosts"] / max(1, max_hosts), label="active_ratio")
    if "dvfs" in trace.columns:
        axes[1, 0].plot(x, trace["dvfs"], label="dvfs")
    axes[1, 0].set_title("Demand vs active ratio vs DVFS")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if "sla_violation" in trace.columns:
        axes[1, 1].plot(x, trace["sla_violation"], label="SLA")
    if "avg_temp" in trace.columns:
        axes[1, 1].plot(x, trace["avg_temp"], label="avg_temp")
    if "pue" in trace.columns:
        axes[1, 1].plot(x, trace["pue"], label="PUE")
    axes[1, 1].set_title("SLA / temperature / PUE")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Chẩn đoán hành vi {policy_name}", y=1.02)
    fig.tight_layout()

    fig_path = out_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)

    return {
        "action_counts": action_counts,
        "phase_diag": phase_diag,
        "fig_path": fig_path,
        "keep_pct": keep_pct,
        "distinct_actions": distinct_actions,
    }


def plot_phase_analysis(
    phase_summary: pd.DataFrame,
    fig_dir: Path | str,
    policy_name: str = "PPO",
    filename: str = "04_phan_tich_theo_phase_ppo.png",
) -> Path:
    """Vẽ 2x2 phase analysis: active, off, DVFS, temperature."""
    fig_dir = _ensure_dir(fig_dir)
    phase_plot = _sort_phase_frame(phase_summary.dropna(how="all")).copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    phase_metrics = [
        ("active_mean", "Số host active trung bình"),
        ("off_mean", "Số host off trung bình"),
        ("dvfs_mean", "DVFS trung bình"),
        ("temp_mean", "Nhiệt độ trung bình"),
    ]

    for ax, (col, title) in zip(axes.ravel(), phase_metrics):
        if col in phase_plot.columns:
            phase_plot[col].plot(kind="bar", ax=ax)
        else:
            ax.text(0.5, 0.5, f"Thiếu cột\n{col}", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Phân tích hành vi {policy_name} theo phase low / medium / high", y=1.02)
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path


def plot_reward_groups(
    trace: pd.DataFrame,
    fig_dir: Path | str,
    filename: str = "05_nhom_reward_theo_thoi_gian.png",
) -> Optional[Path]:
    """Vẽ reward_group_* nếu trace có các cột này."""
    fig_dir = _ensure_dir(fig_dir)
    reward_group_cols = [c for c in trace.columns if c.startswith("reward_group_")]

    if not reward_group_cols:
        print("Không có reward_group_* trong trace.")
        return None

    x = _get_x(trace)
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for col in reward_group_cols:
        ax.plot(x, trace[col], label=col.replace("reward_group_", ""))

    ax.set_title("Các nhóm reward theo thời gian")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig_path = fig_dir / filename
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)
    return fig_path
