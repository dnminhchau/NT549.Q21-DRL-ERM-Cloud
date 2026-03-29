from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def extract_workload_from_azure_packing(
    db_path: str | Path,
    output_csv: str | Path,
    *,
    window_start_days: float = 0.0,
    window_end_days: float = 14.0,
    bin_hours: float = 1.0,
    vmtype_agg: str = "mean",
    demand_min: float = 0.05,
    demand_max: float = 1.20,
    chunk_size: int = 300_000,
) -> pd.DataFrame:
    """
    Biến AzurePackingTraceV1 (.sqlite) thành workload.csv.

    Ý tưởng:
    - vm: cho biết VM nào chạy từ lúc nào tới lúc nào
    - vmType: cho biết mỗi loại VM dùng bao nhiêu core
    - Ta cộng tổng core của tất cả VM còn sống tại từng timestep

    Tham số quan trọng:
    - window_start_days, window_end_days: cửa sổ thời gian muốn lấy
    - bin_hours: độ rộng mỗi timestep
    - vmtype_agg: 'mean' hoặc 'max' khi gộp vmType theo vmTypeId
    """
    db_path = Path(db_path)
    output_csv = Path(output_csv)

    if not db_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file SQLite: {db_path}")

    if vmtype_agg not in {"mean", "max"}:
        raise ValueError("vmtype_agg chỉ nhận 'mean' hoặc 'max'.")

    agg_expr = "AVG(core)" if vmtype_agg == "mean" else "MAX(core)"
    bin_days = bin_hours / 24.0
    n_bins = int(math.ceil((window_end_days - window_start_days) / bin_days))

    conn = sqlite3.connect(db_path)
    diff_core = np.zeros(n_bins + 1, dtype=np.float64)
    diff_count = np.zeros(n_bins + 1, dtype=np.float64)

    sql = f"""
    WITH profile AS (
        SELECT vmTypeId, {agg_expr} AS core
        FROM vmType
        GROUP BY vmTypeId
    )
    SELECT
        CASE WHEN v.starttime < ? THEN ? ELSE v.starttime END AS starttime,
        CASE
            WHEN COALESCE(v.endtime, ?) > ? THEN ?
            ELSE COALESCE(v.endtime, ?)
        END AS endtime,
        p.core AS core
    FROM vm v
    JOIN profile p
      ON v.vmTypeId = p.vmTypeId
    WHERE COALESCE(v.endtime, ?) > ?
      AND v.starttime < ?
    """

    params = (
        window_start_days,
        window_start_days,
        window_end_days,
        window_end_days,
        window_end_days,
        window_end_days,
        window_end_days,
        window_start_days,
        window_end_days,
    )

    total_rows = 0
    for chunk in pd.read_sql_query(sql, conn, params=params, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["starttime", "endtime", "core"])
        if chunk.empty:
            continue

        start_bin = np.floor((chunk["starttime"].to_numpy() - window_start_days) / bin_days).astype(int)
        end_bin = np.ceil((chunk["endtime"].to_numpy() - window_start_days) / bin_days).astype(int)
        end_bin = np.maximum(end_bin, start_bin + 1)

        start_bin = np.clip(start_bin, 0, n_bins - 1)
        end_bin = np.clip(end_bin, 1, n_bins)

        core = chunk["core"].to_numpy(dtype=np.float64)
        count = np.ones(len(chunk), dtype=np.float64)

        np.add.at(diff_core, start_bin, core)
        np.add.at(diff_core, end_bin, -core)
        np.add.at(diff_count, start_bin, count)
        np.add.at(diff_count, end_bin, -count)

        total_rows += len(chunk)

    conn.close()

    raw_core_demand = diff_core[:-1].cumsum()
    active_vm_count = diff_count[:-1].cumsum()

    raw_core_demand = np.maximum(raw_core_demand, 0)
    active_vm_count = np.maximum(active_vm_count, 0)

    scale = np.percentile(raw_core_demand, 95)
    if scale <= 0:
        raise RuntimeError("Không thể chuẩn hóa demand vì scale <= 0.")

    demand = raw_core_demand / scale
    demand = np.clip(demand, demand_min, demand_max)

    df = pd.DataFrame(
        {
            "timestep": np.arange(n_bins),
            "demand": demand,
            "raw_core_demand": raw_core_demand,
            "active_vm_count": active_vm_count.astype(int),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
