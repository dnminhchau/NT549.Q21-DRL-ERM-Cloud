from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def list_experiments() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    experiments = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    experiments.sort(key=lambda p: p.name.lower())
    return experiments


def list_runs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def format_float(value: float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def row_at_step(df: pd.DataFrame, step: int) -> pd.Series:
    if "step" in df.columns:
        selected = df.loc[df["step"] == step]
        if not selected.empty:
            return selected.iloc[0]
    return df.iloc[min(step, len(df) - 1)]


# ─── Page config & global CSS ────────────────────────────────────────────────

st.set_page_config(page_title="PPO Trace Dashboard", layout="wide", page_icon="⚡")

st.markdown("""
<style>
/* ── Google Font ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark dashboard background ───────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Headings ─────────────────────────────────────────── */
h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; color: #f0f6fc; }
h1 { font-size: 1.7rem; letter-spacing: -0.5px; }
.subtitle { color: #8b949e; font-size: 0.82rem; margin-top: -0.5rem; margin-bottom: 1.2rem; }

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #388bfd; }
[data-testid="stMetricLabel"] { 
    font-size: 0.7rem !important; 
    color: #8b949e !important; 
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] { 
    font-family: 'JetBrains Mono', monospace !important; 
    font-size: 1.1rem !important; 
    color: #58a6ff !important;
}

/* ── Player controls live in sidebar ──────────────────── */

/* ── Dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }

/* ── Divider ──────────────────────────────────────────── */
hr { border-color: #21262d !important; }

/* ── Tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-size: 0.8rem;
    font-family: 'DM Sans', sans-serif;
    color: #8b949e;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* ── Line chart ───────────────────────────────────────── */
[data-testid="stArrowVegaLiteChart"] canvas { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ───────────────────────────────────────────────────

for key, default in [("_timestep", 0), ("playing", False), ("speed", 2)]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.speed not in (2, 5, 10):
    st.session_state.speed = 2

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Nguồn dữ liệu")
    experiments = list_experiments()
    if not experiments:
        st.error("Không tìm thấy outputs/. Hãy chạy notebook để tạo trace.")
        st.stop()

    default_idx = 0
    for i, exp in enumerate(experiments):
        if exp.name.lower() == "multiphase":
            default_idx = i
            break
    experiment_dir = st.selectbox(
        "Chọn nhóm kết quả",
        experiments,
        format_func=lambda p: p.name,
        index=default_idx,
    )

    runs = list_runs(experiment_dir)
    if not runs:
        st.error(f"Không tìm thấy run trong {experiment_dir.name}.")
        st.stop()

    run_dir = st.selectbox("Chọn run", runs, format_func=lambda p: p.name, index=0)
    trace_dir = run_dir / "traces"
    default_ppo_trace = trace_dir / "ppo_trace.csv"

    st.markdown("---")
    uploaded_trace = st.file_uploader("Upload trace CSV (tuỳ chọn)", type=["csv"])
    if uploaded_trace is not None:
        ppo_df = pd.read_csv(uploaded_trace)
        ppo_name = "📤 Uploaded trace"
    else:
        if not default_ppo_trace.exists():
            st.error(f"Không thấy: {default_ppo_trace}")
            st.stop()
        ppo_df = load_trace(default_ppo_trace)
        ppo_name = "PPO"

    st.markdown("---")
    baseline_options = {
        "RoundRobin": trace_dir / "roundrobin_trace.csv",
        "Threshold":  trace_dir / "threshold_trace.csv",
        "BestFit":    trace_dir / "bestfit_trace.csv",
        "Fixed-Keep": trace_dir / "fixed_keep_trace.csv",
    }
    available_baselines = {n: p for n, p in baseline_options.items() if p.exists()}
    selected_baselines = st.multiselect(
        "Baselines so sánh",
        list(available_baselines.keys()),
        default=list(available_baselines.keys()),
    )
    baseline_dfs = {
        n: load_trace(p)
        for n, p in available_baselines.items()
        if n in selected_baselines
    }

    st.markdown("---")
    st.markdown("### 📊 Thông tin run")
    total_steps = len(ppo_df)
    max_step = int(ppo_df["step"].max()) if "step" in ppo_df.columns else total_steps - 1
    st.caption(f"Steps: `{total_steps}` · Max: `{max_step}`")
    st.caption(f"Baselines: `{len(baseline_dfs)}`")

# ─── Column validation ─────────────────────────────────────────────────────────

required_cols = {
    "step", "demand", "power_total", "pue", "sla_violation",
    "active_hosts", "sleep_hosts", "off_hosts", "dvfs",
    "avg_temp", "max_temp", "migrations", "migration_cost",
}
missing = required_cols - set(ppo_df.columns)
if missing:
    st.error(f"Trace thiếu cột: {', '.join(sorted(missing))}")
    st.stop()

# ─── Header ───────────────────────────────────────────────────────────────────

header_col, controls_col = st.columns([2.6, 1.4], gap="large")
with header_col:
    st.markdown("## ⚡ PPO Trace Dashboard")
    st.markdown(
        '<p class="subtitle">Phát lại trace đánh giá sau training · '
        f'Run: <code>{run_dir.name}</code></p>',
        unsafe_allow_html=True,
    )

with controls_col:
    st.markdown("### ▶ Playback")
    speed_cols = st.columns(3)
    speeds = [2, 5, 10]
    for i, spd in enumerate(speeds):
        label = f"×{spd}"
        is_active = st.session_state.speed == spd
        if speed_cols[i].button(
            label,
            key=f"spd_{spd}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.speed = spd

    play_col, replay_col, step_col = st.columns(3)
    play_label = "⏸" if st.session_state.playing else "▶"
    if play_col.button(play_label, key="play_btn", use_container_width=True, help="Play/Pause"):
        st.session_state.playing = not st.session_state.playing
    if replay_col.button("↺", key="replay_btn", use_container_width=True, help="Replay từ đầu"):
        st.session_state._timestep = 0
        st.session_state.playing = False
    if step_col.button("⏭", key="step_btn", use_container_width=True, help="Tăng 1 timestep"):
        st.session_state._timestep = min(st.session_state._timestep + 1, max_step)

    _ts_before = int(st.session_state._timestep)
    _slider_val = st.slider(
        f"Timestep {_ts_before} / {max_step}",
        min_value=0,
        max_value=max_step,
        value=_ts_before,
    )
    if _slider_val != _ts_before:
        # User manually dragged → jump to that position + pause playback
        st.session_state._timestep = _slider_val
        st.session_state.playing = False

# ─── Current-step data ────────────────────────────────────────────────────────

step_value = int(st.session_state._timestep)
row = (
    ppo_df.loc[ppo_df["step"] == step_value].iloc[0]
    if "step" in ppo_df.columns
    else ppo_df.iloc[step_value]
)

# ─── KPI metrics row ──────────────────────────────────────────────────────────

# Progress bar (semantic colour based on violation level)
sla_val = float(row["sla_violation"])
pue_val = float(row["pue"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⚡ Power Total",  format_float(row["power_total"]) + " W")
col2.metric("📊 PUE",          format_float(row["pue"], 3))
col3.metric("🔥 SLA Violation",format_float(row["sla_violation"], 4))
col4.metric("🖥️ Active Hosts",  int(row["active_hosts"]))
col5.metric("📉 DVFS",          format_float(row["dvfs"], 2))

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("💤 Sleep Hosts",   int(row["sleep_hosts"]))
col7.metric("🔴 Off Hosts",     int(row["off_hosts"]))
col8.metric("🔄 Migrations",    int(row["migrations"]))
col9.metric("🌡️ Avg Temp",      format_float(row["avg_temp"], 1) + " °C")
col10.metric("🔝 Max Temp",     format_float(row["max_temp"], 1) + " °C")

st.markdown("---")

# ─── Tabs: Comparison + Charts ────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📋 So sánh tại timestep", "📈 Time Series", "💾 Tiết kiệm năng lượng"])

with tab1:
    ppo_power = float(row["power_total"])
    comparison_rows = [{
        "Policy": "🤖 PPO",
        "Demand": round(float(row["demand"]), 3),
        "Active": int(row["active_hosts"]),
        "Sleep": int(row["sleep_hosts"]),
        "Off": int(row["off_hosts"]),
        "DVFS": round(float(row["dvfs"]), 3),
        "Power (W)": round(ppo_power, 2),
        "PUE": round(float(row["pue"]), 4),
        "SLA": round(float(row["sla_violation"]), 5),
        "Migrations": int(row["migrations"]),
        "AvgTemp °C": round(float(row["avg_temp"]), 1),
        "MaxTemp °C": round(float(row["max_temp"]), 1),
        "Save vs PPO (W)": 0.0,
        "Save % vs PPO": 0.0,
    }]

    for name, df in baseline_dfs.items():
        if "power_total" not in df.columns:
            continue
        b_row = row_at_step(df, step_value)
        bp = float(b_row["power_total"])
        saving_abs = bp - ppo_power
        saving_pct = (saving_abs / bp * 100.0) if bp > 0 else 0.0
        comparison_rows.append({
            "Policy": name,
            "Demand": round(float(b_row.get("demand", row["demand"])), 3),
            "Active": int(b_row.get("active_hosts", 0)),
            "Sleep": int(b_row.get("sleep_hosts", 0)),
            "Off": int(b_row.get("off_hosts", 0)),
            "DVFS": round(float(b_row.get("dvfs", 0)), 3),
            "Power (W)": round(bp, 2),
            "PUE": round(float(b_row.get("pue", 0)), 4),
            "SLA": round(float(b_row.get("sla_violation", 0)), 5),
            "Migrations": int(b_row.get("migrations", 0)),
            "AvgTemp °C": round(float(b_row.get("avg_temp", 0)), 1),
            "MaxTemp °C": round(float(b_row.get("max_temp", 0)), 1),
            "Save vs PPO (W)": round(saving_abs, 1),
            "Save % vs PPO": round(saving_pct, 2),
        })

    compare_df = pd.DataFrame(comparison_rows).set_index("Policy")
    chart_a, chart_b = st.columns(2)
    chart_a.markdown("**Power (W)**")
    chart_a.bar_chart(compare_df["Power (W)"])
    chart_b.markdown("**SLA Violation**")
    chart_b.bar_chart(compare_df["SLA"])

    chart_c, chart_d = st.columns(2)
    chart_c.markdown("**PUE**")
    chart_c.bar_chart(compare_df["PUE"])
    chart_d.markdown("**Active Hosts**")
    chart_d.bar_chart(compare_df["Active"])

    chart_e, chart_f = st.columns(2)
    chart_e.markdown("**Migrations**")
    chart_e.bar_chart(compare_df["Migrations"])
    chart_f.markdown("**Save vs PPO (W)**")
    chart_f.bar_chart(compare_df["Save vs PPO (W)"])

with tab2:
    metric_map = {
        "Power Total (W)": "power_total",
        "PUE": "pue",
        "SLA Violation": "sla_violation",
        "DVFS": "dvfs",
        "Active Hosts": "active_hosts",
        "Avg Temp °C": "avg_temp",
    }
    col_m, col_smooth = st.columns([3, 1])
    metric_label = col_m.selectbox("Metric", list(metric_map.keys()), index=0)
    smooth_win = col_smooth.number_input("Smooth window", min_value=1, max_value=50, value=1)
    metric_col = metric_map[metric_label]

    chart_df = pd.DataFrame({"PPO": ppo_df.set_index("step")[metric_col]})
    for name, df in baseline_dfs.items():
        if metric_col in df.columns:
            chart_df[name] = df.set_index("step")[metric_col]

    if smooth_win > 1:
        chart_df = chart_df.rolling(smooth_win, min_periods=1).mean()

    # Highlight current step
    st.line_chart(chart_df, height=320)
    st.caption(f"Hiển thị {len(chart_df)} bước · Đường thẳng đứng tại step {step_value} (scroll slider để di chuyển)")

with tab3:
    total_ppo   = ppo_df["power_total"].sum()
    energy_rows = [{"Policy": "🤖 PPO", "Total Power (ΣW)": round(total_ppo, 1), "Saved vs PPO (ΣW)": 0.0, "Saved %": "—"}]
    for name, df in baseline_dfs.items():
        if "power_total" not in df.columns:
            continue
        total_b = df["power_total"].sum()
        saved   = total_b - total_ppo
        pct     = saved / total_b * 100 if total_b > 0 else 0
        energy_rows.append({
            "Policy":             name,
            "Total Power (ΣW)":  round(total_b, 1),
            "Saved vs PPO (ΣW)": round(saved, 1),
            "Saved %":           f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%",
        })
    st.dataframe(pd.DataFrame(energy_rows), hide_index=True)
    st.caption("Tổng toàn bộ trace (không phụ thuộc timestep hiện tại).")

# ─── Auto-advance playback ────────────────────────────────────────────────────

if st.session_state.playing:
    next_step = int(st.session_state._timestep) + int(st.session_state.speed)
    st.session_state._timestep = min(next_step, max_step)
    if st.session_state._timestep >= max_step:
        st.session_state.playing = False
    time.sleep(0.05)
    st.rerun()