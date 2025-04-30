#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 MVN1_excel.csv / MVN1.csv 拆成 2 份：
    • network_vertices.csv
    • network_edges.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# ---------- 列重命名映射 ----------
_COL_MAP = {
    "index":    "node_id",
    "coords:0": "x",
    "coords:1": "y",
    "coords:2": "z",
}

# 必须存在的顶点列
_REQ_V_COLS = {"node_id", "x", "y", "z", "pressure"}


# --------------------------------------------------------------------------- #
def split_table(src: Path, out_dir: Path) -> None:
    """读取 *src*，拆分后写入 *out_dir*。"""
    df: pd.DataFrame = (
        pd.read_csv(src)
          .rename(columns=_COL_MAP, errors="ignore")
    )

    # ---------- 顶点表 ----------
    v_cols = list(_REQ_V_COLS)
    missing = set(v_cols) - set(df.columns)
    if missing:
        raise ValueError(f"缺失顶点列: {missing}")

    # node_id 安全转 int
    df["node_id"] = df["node_id"].round().astype("Int64")
    df[v_cols].to_csv(out_dir / "network_vertices.csv", index=False)

    # ---------- 边表 ----------
    # 基于“连续行构成一条边”的假设
    e_src = df[["node_id", "diameters"]].copy()

    # vertex_1
    e_src["vertex_1"] = e_src["node_id"].round().astype("Int64")

    # vertex_2：shift(-1) 后四舍五入取整，最后一行 NaN 会被丢弃
    v2 = e_src["node_id"].shift(-1)                       # float64(含 NaN)
    mask = v2.notna()
    e_src.loc[mask, "vertex_2"] = (
        v2[mask].round().astype("Int64")
    )

    e_src = (
        e_src
        .dropna(subset=["vertex_2"])                     # 去掉最后一行
        .rename(columns={"diameters": "diameter"})
        [["vertex_1", "vertex_2", "diameter"]]           # 调整列顺序
    )

    e_src.to_csv(out_dir / "network_edges.csv", index=False)
    print(f"[split_edge_table] 已写入: {out_dir/'network_vertices.csv'}")
    print(f"[split_edge_table] 已写入: {out_dir/'network_edges.csv'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: split_edge_table.py <src_csv> <out_dir>\n"
            "  例如：python scripts/split_edge_table.py "
            "data/network/MVN1_excel.csv data/network/MVN1_splitted"
        )
        sys.exit(1)

    src_csv  = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.mkdir(parents=True, exist_ok=True)

    split_table(src_csv, out_path)
