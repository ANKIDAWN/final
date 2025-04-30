#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split MVN1_excel.csv / MVN1.csv into 2 files:
    • network_vertices.csv
    • network_edges.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# ---------- Column renaming mapping ----------
_COL_MAP = {
    "index":    "node_id",
    "coords:0": "x",
    "coords:1": "y",
    "coords:2": "z",
}

# Required vertex columns
_REQ_V_COLS = {"node_id", "x", "y", "z", "pressure"}


# --------------------------------------------------------------------------- #
def split_table(src: Path, out_dir: Path) -> None:
    """Read *src*, split and write to *out_dir*."""
    df: pd.DataFrame = (
        pd.read_csv(src)
          .rename(columns=_COL_MAP, errors="ignore")
    )

    # ---------- Vertex table ----------
    v_cols = list(_REQ_V_COLS)
    missing = set(v_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing vertex columns: {missing}")

    # Safely convert node_id to int
    df["node_id"] = df["node_id"].round().astype("Int64")
    df[v_cols].to_csv(out_dir / "network_vertices.csv", index=False)

    # ---------- Edge table ----------
    # Based on the assumption that consecutive rows form an edge
    e_src = df[["node_id", "diameters"]].copy()

    # vertex_1
    e_src["vertex_1"] = e_src["node_id"].round().astype("Int64")

    # vertex_2: shift(-1) then round to integer, last row with NaN will be dropped
    v2 = e_src["node_id"].shift(-1)                       # float64(with NaN)
    mask = v2.notna()
    e_src.loc[mask, "vertex_2"] = (
        v2[mask].round().astype("Int64")
    )

    e_src = (
        e_src
        .dropna(subset=["vertex_2"])                     # Remove the last row
        .rename(columns={"diameters": "diameter"})
        [["vertex_1", "vertex_2", "diameter"]]           # Adjust column order
    )

    e_src.to_csv(out_dir / "network_edges.csv", index=False)
    print(f"[split_edge_table] Written: {out_dir/'network_vertices.csv'}")
    print(f"[split_edge_table] Written: {out_dir/'network_edges.csv'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: split_edge_table.py <src_csv> <out_dir>\n"
            "  Example: python scripts/split_edge_table.py "
            "data/network/MVN1_excel.csv data/network/MVN1_splitted"
        )
        sys.exit(1)

    src_csv  = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.mkdir(parents=True, exist_ok=True)

    split_table(src_csv, out_path)
