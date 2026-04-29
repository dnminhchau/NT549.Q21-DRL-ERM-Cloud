from __future__ import annotations

from html import escape
from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"

RAINBOW_COLORS = [
    "#f4b6c2",  # pastel pink
    "#f7c9a9",  # pastel peach
    "#f6e3a1",  # pastel yellow
    "#bfe3c0",  # pastel green
    "#b9e3e6",  # pastel aqua
    "#bfd3f6",  # pastel blue
    "#d8c6f2",  # pastel lavender
]

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass


def list_experiments() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []

    experiments = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    experiments.sort(key=lambda p: (p.name.lower() != "multiphase", p.name.lower()))
    return experiments


def list_runs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def normalize_trace(df: pd.DataFrame) -> pd.DataFrame:
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
        df["step"] = df["step"].astype(int)

    return df


def load_trace(path: Path) -> pd.DataFrame:
    return normalize_trace(pd.read_csv(path))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None

    return None


def format_float(value: float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def format_int(value: float) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "N/A"


def format_compact(value: float, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:
        return "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f}M"

    if abs_value >= 1_000:
        return f"{value / 1_000:.{digits}f}K"

    return f"{value:.{digits}f}"


def row_at_step(df: pd.DataFrame, step: int) -> pd.Series:
    if df.empty:
        raise ValueError("Trace dataframe is empty.")

    if "step" in df.columns:
        selected = df.loc[df["step"] == step]

        if not selected.empty:
            return selected.iloc[0]

        nearest_idx = (df["step"] - step).abs().idxmin()
        return df.loc[nearest_idx]

    return df.iloc[min(step, len(df) - 1)]


def pct(part: float, total: float) -> float:
    total = float(total)

    if total <= 0:
        return 0.0

    return float(part) / total


def mini_badge(text: str, tone: str = "blue") -> str:
    return f'<span class="badge badge-{tone}">{escape(text)}</span>'


def kpi_card(label: str, value: str, sub: str = "", tone: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="kpi-card tone-{tone}">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            <div class="kpi-sub">{escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h3>{escape(title)}</h3>
            <p>{escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_comparison_df(
    ppo_df: pd.DataFrame,
    baseline_dfs: dict[str, pd.DataFrame],
    step_value: int,
) -> pd.DataFrame:
    row = row_at_step(ppo_df, step_value)
    ppo_power = float(row["power_total"])

    rows = [
        {
            "Policy": "PPO",
            "Demand": round(float(row["demand"]), 3),
            "Active": int(row["active_hosts"]),
            "Sleep": int(row["sleep_hosts"]),
            "Off": int(row["off_hosts"]),
            "DVFS": round(float(row["dvfs"]), 3),
            "Power (W)": round(ppo_power, 2),
            "PUE": round(float(row["pue"]), 4),
            "SLA": round(float(row["sla_violation"]), 5),
            "Migrations": int(row["migrations"]),
            "Avg Temp (°C)": round(float(row["avg_temp"]), 1),
            "Max Temp (°C)": round(float(row["max_temp"]), 1),
            "PPO Saving (W)": 0.0,
            "PPO Saving (%)": 0.0,
        }
    ]

    for name, df in baseline_dfs.items():
        if df.empty or "power_total" not in df.columns:
            continue

        b_row = row_at_step(df, step_value)
        baseline_power = float(b_row["power_total"])

        saving_abs = baseline_power - ppo_power
        saving_pct = saving_abs / baseline_power * 100 if baseline_power > 0 else 0

        rows.append(
            {
                "Policy": name,
                "Demand": round(float(b_row.get("demand", row["demand"])), 3),
                "Active": int(b_row.get("active_hosts", 0)),
                "Sleep": int(b_row.get("sleep_hosts", 0)),
                "Off": int(b_row.get("off_hosts", 0)),
                "DVFS": round(float(b_row.get("dvfs", 0)), 3),
                "Power (W)": round(baseline_power, 2),
                "PUE": round(float(b_row.get("pue", 0)), 4),
                "SLA": round(float(b_row.get("sla_violation", 0)), 5),
                "Migrations": int(b_row.get("migrations", 0)),
                "Avg Temp (°C)": round(float(b_row.get("avg_temp", 0)), 1),
                "Max Temp (°C)": round(float(b_row.get("max_temp", 0)), 1),
                "PPO Saving (W)": round(saving_abs, 1),
                "PPO Saving (%)": round(saving_pct, 2),
            }
        )

    return pd.DataFrame(rows)


def make_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, height: int = 260):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
        .encode(
            x=alt.X(
                f"{x_col}:N",
                sort="-y",
                title=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelColor="#334155",
                    labelFontWeight="bold",
                ),
            ),
            y=alt.Y(
                f"{y_col}:Q",
                title=None,
                axis=alt.Axis(labelColor="#64748b"),
            ),
            color=alt.Color(
                f"{x_col}:N",
                legend=None,
                scale=alt.Scale(range=RAINBOW_COLORS),
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:N"),
                alt.Tooltip(f"{y_col}:Q", format=",.4f"),
            ],
        )
        .properties(title=title, height=height)
        .configure_view(strokeWidth=0)
        .configure_title(
            anchor="start",
            fontSize=15,
            fontWeight="bold",
            color="#1f2c40",
        )
    )


def make_line_chart(chart_df: pd.DataFrame, metric_label: str, step_value: int, height: int = 380):
    long_df = (
        chart_df.reset_index()
        .rename(columns={"index": "step"})
        .melt(id_vars="step", var_name="Policy", value_name="Value")
        .dropna()
    )

    line = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.8)
        .encode(
            x=alt.X(
                "step:Q",
                title="Timestep",
                axis=alt.Axis(labelColor="#64748b", titleColor="#334155"),
            ),
            y=alt.Y(
                "Value:Q",
                title=metric_label,
                axis=alt.Axis(labelColor="#64748b", titleColor="#334155"),
            ),
            color=alt.Color(
                "Policy:N",
                title="Policy",
                scale=alt.Scale(range=RAINBOW_COLORS),
                legend=alt.Legend(labelColor="#334155", titleColor="#1f2c40"),
            ),
            tooltip=[
                alt.Tooltip("step:Q", title="Step"),
                alt.Tooltip("Policy:N"),
                alt.Tooltip("Value:Q", format=",.4f"),
            ],
        )
        .properties(height=height)
    )

    rule = (
        alt.Chart(pd.DataFrame({"step": [step_value]}))
        .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="#cc8fa0")
        .encode(x="step:Q")
    )

    return (line + rule).interactive().configure_view(strokeWidth=0)

st.set_page_config(
    page_title="Cloud Energy Optimization Dashboard",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #fffdfd;
    --panel: rgba(255, 255, 255, 0.86);
    --panel-strong: rgba(255, 255, 255, 0.96);
    --border: rgba(148, 163, 184, 0.20);
    --text: #223047;
    --muted: #6b7a90;

    --red: #f4b6c2;
    --orange: #f7c9a9;
    --yellow: #f6e3a1;
    --green: #bfe3c0;
    --cyan: #b9e3e6;
    --blue: #bfd3f6;
    --violet: #d8c6f2;

    --shadow: 0 16px 38px rgba(15, 23, 42, 0.07);
}
/* =========================
   Hide Streamlit Deploy + Menu
   ========================= */

/* Ẩn nút Deploy */
[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}

/* Ẩn dấu ba chấm menu góc phải */
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}

/* Ẩn decoration line phía trên nếu có */
[data-testid="stDecoration"] {
    display: none !important;
}
/* App background */
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 5% 5%, rgba(244, 182, 194, 0.18), transparent 24rem),
        radial-gradient(circle at 25% 0%, rgba(247, 201, 169, 0.18), transparent 24rem),
        radial-gradient(circle at 45% 4%, rgba(246, 227, 161, 0.16), transparent 22rem),
        radial-gradient(circle at 65% 0%, rgba(191, 227, 192, 0.16), transparent 22rem),
        radial-gradient(circle at 82% 8%, rgba(185, 227, 230, 0.16), transparent 22rem),
        radial-gradient(circle at 100% 0%, rgba(191, 211, 246, 0.16), transparent 22rem),
        linear-gradient(135deg, #fffaf9 0%, #fffdf7 35%, #f9fcff 68%, #fbfffd 100%);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="block-container"] {
    padding-top: 1.35rem;
    padding-bottom: 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(252, 250, 248, 0.95)),
        linear-gradient(135deg, rgba(244, 182, 194, 0.08), rgba(191, 211, 246, 0.08));
    border-right: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 10px 0 24px rgba(15, 23, 42, 0.04);
}

[data-testid="stSidebar"] * {
    color: #2b3a4f !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(148, 163, 184, 0.20) !important;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid transparent;
    background:
        linear-gradient(white, white) padding-box,
        linear-gradient(90deg,
            #f4b6c2,
            #f7c9a9,
            #f6e3a1,
            #bfe3c0,
            #b9e3e6,
            #bfd3f6,
            #d8c6f2
        ) border-box;
    border-radius: 28px;
    padding: 1.45rem 1.7rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.05rem;
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 8% 20%, rgba(244, 182, 194, 0.14), transparent 12rem),
        radial-gradient(circle at 35% 0%, rgba(246, 227, 161, 0.12), transparent 12rem),
        radial-gradient(circle at 68% 20%, rgba(185, 227, 230, 0.12), transparent 12rem),
        radial-gradient(circle at 98% 0%, rgba(216, 198, 242, 0.12), transparent 12rem);
    pointer-events: none;
}

.hero > * {
    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: 2.1rem;
    line-height: 1.08;
    font-weight: 900;
    margin: 0;
    letter-spacing: -0.045em;
    background: linear-gradient(90deg,
        #cc8fa0,
        #d8a688,
        #c8b26b,
        #8db894,
        #86b9bf,
        #90abd9,
        #b39ed8
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #64748b;
    margin-top: 0.48rem;
    font-size: 0.98rem;
    max-width: 62rem;
}

/* Badge */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 999px;
    padding: .36rem .76rem;
    font-size: .78rem;
    font-weight: 800;
    background: rgba(255, 255, 255, 0.78);
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
}

.badge-blue { color: #5779b8; background: rgba(234, 241, 255, 0.95); }
.badge-green { color: #5d9270; background: rgba(234, 248, 238, 0.95); }
.badge-amber { color: #a78d4b; background: rgba(252, 247, 223, 0.95); }
.badge-red { color: #b9818d; background: rgba(252, 238, 242, 0.95); }
.badge-violet { color: #8f7cb2; background: rgba(245, 240, 252, 0.95); }
.badge-gray { color: #607086; background: rgba(245, 248, 252, 0.95); }

/* KPI cards */
.kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 126px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 22px;
    padding: 1.02rem 1.08rem;
    background: rgba(255, 255, 255, 0.84);
    box-shadow: var(--shadow);
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}

.kpi-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 5px;
    background: linear-gradient(90deg,
        #f4b6c2,
        #f7c9a9,
        #f6e3a1,
        #bfe3c0,
        #b9e3e6,
        #bfd3f6,
        #d8c6f2
    );
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
    border-color: rgba(191, 211, 246, 0.45);
}

.kpi-label {
    color: #7a889b;
    font-size: .72rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-bottom: .52rem;
}

.kpi-value {
    color: #1f2c40;
    font-size: 1.75rem;
    font-weight: 900;
    letter-spacing: -0.045em;
}

.kpi-sub {
    color: #7a889b;
    font-size: .82rem;
    margin-top: .22rem;
    min-height: 1.1rem;
}

.tone-blue {
    background: linear-gradient(180deg, rgba(241, 246, 255, 0.98), rgba(255,255,255,0.90));
}

.tone-green {
    background: linear-gradient(180deg, rgba(241, 250, 243, 0.98), rgba(255,255,255,0.90));
}

.tone-amber {
    background: linear-gradient(180deg, rgba(255, 251, 238, 0.98), rgba(255,255,255,0.90));
}

.tone-red {
    background: linear-gradient(180deg, rgba(253, 243, 245, 0.98), rgba(255,255,255,0.90));
}

.tone-violet {
    background: linear-gradient(180deg, rgba(247, 243, 252, 0.98), rgba(255,255,255,0.90));
}

/* Section title */
.section-title {
    margin-top: .65rem;
    margin-bottom: .82rem;
}

.section-title h3 {
    margin: 0;
    font-size: 1.12rem;
    color: #1f2c40;
    letter-spacing: -0.03em;
    font-weight: 900;
}

.section-title p {
    margin: .18rem 0 0 0;
    color: #7a889b;
    font-size: .87rem;
}

/* Streamlit widgets */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: .82rem .95rem !important;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #7a889b !important;
    font-weight: 800 !important;
}

[data-testid="stMetricValue"] {
    color: #1f2c40 !important;
    font-weight: 900 !important;
}

[data-testid="stTabs"] button {
    font-weight: 850;
    color: #7a889b;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7b98cd !important;
    border-bottom-color: #7b98cd !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

[data-testid="stVegaLiteChart"] {
    background: rgba(255, 255, 255, 0.72);
    border-radius: 18px;
    padding: 0.35rem;
}

hr {
    border-color: rgba(148, 163, 184, 0.18) !important;
}

.small-note {
    color: #7a889b;
    font-size: .84rem;
}

.stSelectbox,
.stMultiSelect,
.stSlider,
.stRadio,
.stNumberInput,
.stFileUploader {
    color: #1f2c40 !important;
}

.stButton button {
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.20);
    background: linear-gradient(90deg, rgba(248,250,255,0.96), rgba(250,247,255,0.96));
    color: #2f4058;
    font-weight: 850;
    transition: all .15s ease;
}

.stButton button:hover {
    transform: translateY(-1px);
    border-color: rgba(191, 211, 246, 0.70);
    box-shadow: 0 10px 18px rgba(144, 171, 217, 0.12);
}

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
}
/* Baseline multiselect đẹp, dịu mắt */
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.82) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04) !important;
}

.stMultiSelect [data-baseweb="tag"] {
    background: #f3f6fa !important;
    border: 1px solid #dde5ef !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
    margin: 3px 4px 3px 0 !important;
}

.stMultiSelect [data-baseweb="tag"] span {
    color: #5f6f82 !important;
    font-weight: 600 !important;
}

.stMultiSelect [data-baseweb="tag"] svg {
    fill: #7f8ea2 !important;
}

.stMultiSelect [data-baseweb="tag"]:hover {
    background: #edf2f8 !important;
    border-color: #d3ddea !important;
}
.streamlit-expanderHeader {
    font-weight: 850;
    color: #2b3a4f !important;
}
/* Ẩn nút Deploy */
[data-testid="stDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}
/* Ẩn cụm nút góc phải phía trên */
[data-testid="stHeaderActionElements"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
p,
li,
span,
div {
    color: inherit;
}
</style>
""",
    unsafe_allow_html=True,
)

for key, default in [("_timestep", 0), ("playing", False), ("speed", 2)]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.speed not in (1, 2, 5, 10):
    st.session_state.speed = 2


with st.sidebar:
    st.markdown("## 🌈 Dashboard")
    st.caption("Cloud Energy Optimization · PPO Trace Replay")

    st.markdown("---")
    st.markdown("### 1) Nguồn dữ liệu")

    experiments = list_experiments()

    if not experiments:
        st.error("Không tìm thấy thư mục `outputs/`. Hãy chạy notebook 03 trước để tạo trace.")
        st.stop()

    experiment_dir = st.selectbox(
        "Experiment",
        experiments,
        format_func=lambda p: p.name,
        index=0,
    )

    runs = list_runs(experiment_dir)

    if not runs:
        st.error(f"Không tìm thấy run trong `{experiment_dir.name}`.")
        st.stop()

    run_dir = st.selectbox(
        "Run",
        runs,
        format_func=lambda p: p.name,
        index=0,
    )

    trace_dir = run_dir / "traces"

    uploaded_trace = st.file_uploader(
        "Upload PPO trace CSV khác nếu cần",
        type=["csv"],
    )

    if uploaded_trace is not None:
        ppo_df = normalize_trace(pd.read_csv(uploaded_trace))
        ppo_name = "Uploaded PPO Trace"
    else:
        default_ppo_trace = trace_dir / "ppo_trace.csv"

        if not default_ppo_trace.exists():
            st.error(f"Không thấy file `{default_ppo_trace.name}` trong thư mục traces.")
            st.stop()

        ppo_df = load_trace(default_ppo_trace)
        ppo_name = "PPO"

    st.markdown("### 2) Baseline so sánh")

    baseline_options = {
        "RoundRobin": trace_dir / "roundrobin_trace.csv",
        "Threshold": trace_dir / "threshold_trace.csv",
        "BestFit": trace_dir / "bestfit_trace.csv",
        "Fixed-Keep": trace_dir / "fixed_keep_trace.csv",
    }

    available_baselines = {
        name: path
        for name, path in baseline_options.items()
        if path.exists()
    }

    selected_baselines = st.multiselect(
        "Chọn baseline",
        list(available_baselines.keys()),
        default=list(available_baselines.keys()),
    )

    baseline_dfs = {
        name: load_trace(path)
        for name, path in available_baselines.items()
        if name in selected_baselines
    }

    st.markdown("---")
    st.markdown("### 3) Playback")

    total_steps = len(ppo_df)
    max_step = int(ppo_df["step"].max()) if "step" in ppo_df.columns else total_steps - 1

    if st.session_state._timestep > max_step:
        st.session_state._timestep = max_step

    st.progress(pct(st.session_state._timestep, max_step))

    before = int(st.session_state._timestep)

    slider_val = st.slider(
        "Timestep",
        min_value=0,
        max_value=max_step,
        value=before,
        help="Kéo thanh này để tua dashboard đến timestep mong muốn.",
    )

    if slider_val != before:
        st.session_state._timestep = slider_val
        st.session_state.playing = False

    speed = st.radio(
        "Tốc độ",
        [1, 2, 5, 10],
        index=[1, 2, 5, 10].index(st.session_state.speed),
        horizontal=True,
        format_func=lambda x: f"×{x}",
    )

    st.session_state.speed = int(speed)

    c1, c2, c3 = st.columns(3)

    if c1.button("▶ / ⏸", use_container_width=True):
        st.session_state.playing = not st.session_state.playing

    if c2.button("↺", use_container_width=True):
        st.session_state._timestep = 0
        st.session_state.playing = False

    if c3.button("+1", use_container_width=True):
        st.session_state._timestep = min(st.session_state._timestep + 1, max_step)

    st.caption(
        f"Step `{int(st.session_state._timestep)}` / `{max_step}` · "
        f"Baselines `{len(baseline_dfs)}`"
    )


required_cols = {
    "step",
    "demand",
    "power_total",
    "pue",
    "sla_violation",
    "active_hosts",
    "sleep_hosts",
    "off_hosts",
    "dvfs",
    "avg_temp",
    "max_temp",
    "migrations",
    "migration_cost",
}

missing = required_cols - set(ppo_df.columns)

if missing:
    st.error(f"Trace thiếu cột: {', '.join(sorted(missing))}")
    st.stop()


step_value = int(st.session_state._timestep)
row = row_at_step(ppo_df, step_value)

sla_val = float(row["sla_violation"])
pue_val = float(row["pue"])
power_val = float(row["power_total"])
demand_val = float(row["demand"])
active_val = int(row["active_hosts"])
sleep_val = int(row["sleep_hosts"])
off_val = int(row["off_hosts"])
dvfs_val = float(row["dvfs"])

sla_tone = "green" if sla_val <= 0.005 else "amber" if sla_val <= 0.02 else "red"
pue_tone = "green" if pue_val <= 1.35 else "amber" if pue_val <= 1.50 else "red"


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Cloud Energy Optimization Dashboard</div>
        <div class="hero-subtitle">
            Theo dõi chính sách PPO trong bài toán VM Consolidation, DVFS và Power Management.
            Dashboard này dùng để trình bày quan hệ giữa workload, host state, energy, SLA và PUE theo thời gian.
        </div>
        <div class="badge-row">
            {mini_badge("Experiment: " + experiment_dir.name, "blue")}
            {mini_badge("Run: " + run_dir.name, "violet")}
            {mini_badge("Trace: " + ppo_name, "green")}
            {mini_badge("Step " + str(step_value) + " / " + str(max_step), "gray")}
            {mini_badge("SLA " + format_float(sla_val, 4), sla_tone)}
            {mini_badge("PUE " + format_float(pue_val, 3), pue_tone)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card(
        "Power Total",
        f"{format_compact(power_val)} W",
        f"Demand = {format_float(demand_val, 3)}",
        "blue",
    )

with k2:
    kpi_card(
        "Host State",
        f"{active_val}/{sleep_val}/{off_val}",
        "Active / Sleep / Off",
        "green",
    )

with k3:
    kpi_card(
        "SLA Violation",
        format_float(sla_val, 5),
        "Thấp hơn là tốt hơn",
        sla_tone,
    )

with k4:
    kpi_card(
        "PUE",
        format_float(pue_val, 3),
        "Total facility / IT energy",
        pue_tone,
    )

k5, k6, k7, k8 = st.columns(4)

with k5:
    kpi_card(
        "DVFS",
        format_float(dvfs_val, 2),
        "CPU frequency scaling",
        "violet",
    )

with k6:
    kpi_card(
        "Temperature",
        f"{format_float(row['avg_temp'], 1)} °C",
        f"Max = {format_float(row['max_temp'], 1)} °C",
        "amber",
    )

with k7:
    kpi_card(
        "Migrations",
        format_int(row["migrations"]),
        f"Cost = {format_float(row['migration_cost'], 3)}",
        "blue",
    )

with k8:
    energy_now = float(ppo_df.loc[ppo_df["step"] <= step_value, "power_total"].sum())

    kpi_card(
        "Cumulative Energy",
        format_compact(energy_now),
        "Σ power up to current step",
        "green",
    )


if sla_val > 0.02:
    st.error(
        "SLA violation đang cao tại timestep này. "
        "Đây là điểm tốt để giải thích trade-off giữa energy saving và SLA."
    )
elif sla_val > 0.005:
    st.warning(
        "SLA violation có xuất hiện nhưng vẫn ở mức tương đối thấp. "
        "Cần biện luận rõ trong báo cáo."
    )
else:
    st.success(
        "Timestep hiện tại ổn định: SLA violation thấp, "
        "phù hợp để minh họa chính sách PPO hoạt động bình thường."
    )


summary_eval = safe_read_csv(run_dir / "summary" / "evaluation_results.csv")
summary_saving = safe_read_csv(run_dir / "summary" / "energy_saving_percentages.csv")

section_title(
    "Tổng quan run",
    "Các chỉ số tổng hợp của PPO trên toàn bộ trace. Phần này không phụ thuộc timestep hiện tại.",
)

s1, s2, s3, s4, s5 = st.columns(5)

with s1:
    st.metric("Total Energy PPO", format_compact(ppo_df["power_total"].sum()))

with s2:
    st.metric("Avg Power", f"{format_compact(ppo_df['power_total'].mean())} W")

with s3:
    st.metric("Avg SLA", format_float(ppo_df["sla_violation"].mean(), 5))

with s4:
    st.metric("Avg PUE", format_float(ppo_df["pue"].mean(), 3))

with s5:
    st.metric("Total Migrations", format_int(ppo_df["migrations"].sum()))

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📌 Timestep Snapshot",
        "📈 Time Series",
        "💾 Energy Saving",
        "🧾 Raw Data",
    ]
)


with tab1:
    section_title(
        "So sánh tại timestep hiện tại",
        "So sánh PPO với các baseline tại đúng timestep đang chọn trên thanh playback.",
    )

    compare_df = build_comparison_df(ppo_df, baseline_dfs, step_value)

    c1, c2 = st.columns(2)

    with c1:
        st.altair_chart(
            make_bar_chart(compare_df, "Policy", "Power (W)", "Power tại timestep hiện tại"),
            use_container_width=True,
        )

    with c2:
        st.altair_chart(
            make_bar_chart(compare_df, "Policy", "SLA", "SLA violation tại timestep hiện tại"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)

    with c3:
        st.altair_chart(
            make_bar_chart(compare_df, "Policy", "PUE", "PUE tại timestep hiện tại"),
            use_container_width=True,
        )

    with c4:
        st.altair_chart(
            make_bar_chart(compare_df, "Policy", "Active", "Số host Active"),
            use_container_width=True,
        )

    st.dataframe(compare_df, hide_index=True, use_container_width=True)


with tab2:
    section_title(
        "Diễn biến theo thời gian",
        "Dùng chart này để trình bày PPO phản ứng thế nào khi workload thay đổi.",
    )

    metric_map = {
        "Power Total (W)": "power_total",
        "Demand": "demand",
        "PUE": "pue",
        "SLA Violation": "sla_violation",
        "DVFS": "dvfs",
        "Active Hosts": "active_hosts",
        "Sleep Hosts": "sleep_hosts",
        "Off Hosts": "off_hosts",
        "Avg Temp °C": "avg_temp",
        "Migrations": "migrations",
    }

    col_m, col_smooth, col_window = st.columns([2.5, 1, 1])

    metric_label = col_m.selectbox(
        "Metric",
        list(metric_map.keys()),
        index=0,
    )

    smooth_win = int(
        col_smooth.number_input(
            "Smooth",
            min_value=1,
            max_value=100,
            value=3,
        )
    )

    max_points = int(
        col_window.number_input(
            "Max points",
            min_value=200,
            max_value=10000,
            value=4000,
            step=200,
        )
    )

    metric_col = metric_map[metric_label]

    chart_df = pd.DataFrame(
        {
            "PPO": ppo_df.set_index("step")[metric_col]
        }
    )

    for name, df in baseline_dfs.items():
        if metric_col in df.columns:
            chart_df[name] = df.set_index("step")[metric_col]

    if smooth_win > 1:
        chart_df = chart_df.rolling(smooth_win, min_periods=1).mean()

    if len(chart_df) > max_points:
        stride = max(1, len(chart_df) // max_points)
        chart_df = chart_df.iloc[::stride]

    st.altair_chart(
        make_line_chart(chart_df, metric_label, step_value),
        use_container_width=True,
    )

    st.markdown(
        '<p class="small-note">Đường dọc màu pastel hồng là timestep hiện tại. '
        'Kéo slider ở sidebar để tua dashboard.</p>',
        unsafe_allow_html=True,
    )


with tab3:
    section_title(
        "Tổng năng lượng và phần trăm tiết kiệm",
        "Bảng này dùng tốt cho slide/báo cáo vì thể hiện trực tiếp PPO tiết kiệm bao nhiêu so với baseline.",
    )

    total_ppo = float(ppo_df["power_total"].sum())

    energy_rows = [
        {
            "Policy": "PPO",
            "Total Energy (ΣW)": round(total_ppo, 1),
            "PPO Saving (ΣW)": 0.0,
            "PPO Saving (%)": 0.0,
        }
    ]

    for name, df in baseline_dfs.items():
        if "power_total" not in df.columns:
            continue

        total_b = float(df["power_total"].sum())
        saved = total_b - total_ppo
        saved_pct = saved / total_b * 100 if total_b > 0 else 0.0

        energy_rows.append(
            {
                "Policy": name,
                "Total Energy (ΣW)": round(total_b, 1),
                "PPO Saving (ΣW)": round(saved, 1),
                "PPO Saving (%)": round(saved_pct, 2),
            }
        )

    energy_df = pd.DataFrame(energy_rows)

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.dataframe(energy_df, hide_index=True, use_container_width=True)

    with c2:
        chart_df = energy_df.loc[energy_df["Policy"] != "PPO"].copy()

        if not chart_df.empty:
            st.altair_chart(
                make_bar_chart(
                    chart_df,
                    "Policy",
                    "PPO Saving (%)",
                    "PPO energy saving so với baseline",
                ),
                use_container_width=True,
            )

    if summary_saving is not None:
        with st.expander("Xem file summary/energy_saving_percentages.csv"):
            st.dataframe(summary_saving, hide_index=True, use_container_width=True)

    if summary_eval is not None:
        with st.expander("Xem file summary/evaluation_results.csv"):
            st.dataframe(summary_eval, hide_index=True, use_container_width=True)

    st.info(
        "Khi viết báo cáo, nên claim PPO tốt nhất về total energy. "
        "Không nên claim PPO tốt nhất về PUE/SLA nếu bảng kết quả không chứng minh điều đó."
    )


with tab4:
    section_title(
        "Dữ liệu trace",
        "Dùng để kiểm tra nhanh các cột và export khi cần debug.",
    )

    preview_cols = [
        "step",
        "demand",
        "power_total",
        "pue",
        "sla_violation",
        "active_hosts",
        "sleep_hosts",
        "off_hosts",
        "dvfs",
        "avg_temp",
        "max_temp",
        "migrations",
        "migration_cost",
    ]

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**PPO trace preview**")
        st.dataframe(
            ppo_df[preview_cols].head(300),
            hide_index=True,
            use_container_width=True,
        )

    with c2:
        st.markdown("**PPO trace statistics**")
        st.dataframe(
            ppo_df[preview_cols].describe().T,
            use_container_width=True,
        )

    st.download_button(
        "Download PPO trace CSV",
        data=ppo_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{experiment_dir.name}_{run_dir.name}_ppo_trace.csv",
        mime="text/csv",
        use_container_width=True,
    )

if st.session_state.playing:
    next_step = int(st.session_state._timestep) + int(st.session_state.speed)
    st.session_state._timestep = min(next_step, max_step)

    if st.session_state._timestep >= max_step:
        st.session_state.playing = False

    time.sleep(0.08)
    st.rerun()