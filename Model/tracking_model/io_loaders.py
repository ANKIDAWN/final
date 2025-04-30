"""
tracking_model/io_loaders.py
负责读 MVN 文件，以及把网络顶点/边表写 CSV
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import pandas as pd
import igraph as ig


# ---------- load_mvn ---------------------------------------------------- #
_NAME_MAP = {
    "index":    "node_id",
    "coords:0": "x",
    "coords:1": "y",
    "coords:2": "z",
}
_REQ_COLS = {"node_id", "x", "y", "z"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_NAME_MAP, errors="ignore")
    missing = _REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[load_mvn] 缺失必需列: {missing}")
    df["node_id"] = df["node_id"].round().astype("Int64")
    return df


def load_mvn(csv_file: Path | str) -> pd.DataFrame:
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)
    return _normalize_columns(pd.read_csv(csv_file))


# ---------- save_edges / vertices -------------------------------------- #
def _edges_to_df(g: ig.Graph) -> pd.DataFrame:
    data = {
        "edge_id": range(g.ecount()),
        "vertex_1": [e.tuple[0] for e in g.es],
        "vertex_2": [e.tuple[1] for e in g.es],
    }
    for key in g.es.attributes():          # diameter, flow_rate, …
        data[key] = g.es[key]
    return pd.DataFrame(data)


def save_edges_csv(g: ig.Graph, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    _edges_to_df(g).to_csv(file, index=False)


def _vertices_to_df(g: ig.Graph) -> pd.DataFrame:
    data = {"vertex_id": range(g.vcount())}
    for key in g.vs.attributes():          # x,y,z, pressure…
        data[key] = g.vs[key]
    return pd.DataFrame(data)


def save_vertices_csv(g: ig.Graph, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    _vertices_to_df(g).to_csv(file, index=False)


# 供外部 `from tracking_model.io_loaders import *`
__all__: List[str] = [
    "load_mvn",
    "save_edges_csv",
    "save_vertices_csv",
]

# io_graph.py
import pandas as pd, igraph as ig, numpy as np

def load_network(vert_csv:str, edge_csv:str) -> ig.Graph:
    vdf = pd.read_csv(vert_csv)
    edf = pd.read_csv(edge_csv)

    # ▸ 字段统一
    vdf = vdf.rename(columns=str.strip)
    edf = edf.rename(columns=str.strip).rename(columns={
        'vertex_1':'v1', 'vertex_2':'v2',
        'diameter':'D',  'length':'L'
    })

    # ▸ 缺字段报警
    need_v = {'vertex_id','x','y','z','pressure'}
    need_e = {'v1','v2','D','L'}
    miss_v = need_v - set(vdf.columns)
    miss_e = need_e - set(edf.columns)
    if miss_v or miss_e:
        raise ValueError(f"缺列: vert={miss_v}, edge={miss_e}")

    # ▸ 建图
    idx_map = {vid:i for i,vid in enumerate(vdf['vertex_id'])}
    edges = [(idx_map[a],idx_map[b]) for a,b in zip(edf.v1, edf.v2)]
    g = ig.Graph(edges=edges, directed=True)
    g.vs['p']   = vdf['pressure'].to_numpy()
    g.vs['x']   = vdf['x'].to_numpy()
    g.vs['y']   = vdf['y'].to_numpy()
    g.vs['z']   = vdf['z'].to_numpy()
    g.es['D']   = edf['D'].to_numpy() * 1e-6      # μm→m
    g.es['L']   = edf['L'].to_numpy() * 1e-6
    return g
