from __future__ import annotations
from pathlib import Path
from typing import List, Union

import igraph as ig
import pandas as pd
import pickle
import pyvista as pv
import math

from .phys_utils import (
    mu_rel_fahraeus_lindqvist,
    flow_rate_poisseuille,
    rbc_centerline_velocity,
)
from .io_loaders import load_mvn

__all__: List[str] = [
    "build_graph_from_edges",
    "build_graph_from_mvn",
]


def _ensure_pressure_diff(g: ig.Graph) -> None:
    """
    若 edge 属性里没有 'pressure_diff'，则根据 vertex['pressure']
    自动计算并写入。
    """
    if 'pressure_diff' in g.es.attributes():
        return
    if 'pressure' not in g.vs.attributes():
        raise ValueError(
            "[build_graph] vertex 表缺少 'pressure' 列，无法计算 pressure_diff"
        )
    pres = g.vs['pressure']
    g.es['pressure_diff'] = [
        abs(pres[e.tuple[0]] - pres[e.tuple[1]]) for e in g.es
    ]


def _ensure_length(g: ig.Graph) -> None:
    """如果 edge 没有 length，就用两端坐标算一列再写进去。"""
    if 'length' in g.es.attributes():
        return
    need = {'x', 'y', 'z'}
    if not need.issubset(g.vs.attributes()):
        missing = need - set(g.vs.attributes())
        raise ValueError(f"vertex 缺列 {missing}，算不了 length")
    xs, ys, zs = g.vs['x'], g.vs['y'], g.vs['z']
    g.es['length'] = [
        math.sqrt(
            (xs[e.tuple[0]] - xs[e.tuple[1]])**2 +
            (ys[e.tuple[0]] - ys[e.tuple[1]])**2 +
            (zs[e.tuple[0]] - zs[e.tuple[1]])**2
        )
        for e in g.es
    ]


def _add_flow_attributes(
    g: ig.Graph,
    *,
    plasma_visc: float = 3.5e-3,
) -> None:
    """
    利用 Poiseuille 定律计算每条边的流量和 RBC 速度。
    """
    _ensure_pressure_diff(g)
    _ensure_length(g)

    diam   = g.es['diameter']
    dp     = g.es['pressure_diff']
    length = g.es['length']

    mu_rels = [mu_rel_fahraeus_lindqvist(d) for d in diam]
    flow_qs = [
        flow_rate_poisseuille(d, d_p, L, mu_rel * plasma_visc)
        for d, d_p, L, mu_rel in zip(diam, dp, length, mu_rels)
    ]
    g.es['flow_rate'] = flow_qs

    g.es['rbc_velocity'] = [
        rbc_centerline_velocity(q, d) for q, d in zip(flow_qs, diam)
    ]


def build_graph_from_edges(
    edges_csv: Union[Path, str],
    *,
    vertices_csv: Union[Path, str],
    plasma_viscosity_pa_s: float = 3.5e-3,
) -> ig.Graph:
    """
    从已拆分的 network_edges.csv 和 network_vertices.csv 构建 igraph.Graph。
    自动计算流量和 RBC 速度。
    """
    edges_csv = Path(edges_csv)
    vertices_csv = Path(vertices_csv)
    if not edges_csv.exists():
        raise FileNotFoundError(f"缺少边文件: {edges_csv}")
    if not vertices_csv.exists():
        raise FileNotFoundError(f"缺少顶点文件: {vertices_csv}")

    e_df = pd.read_csv(edges_csv)
    v_df = pd.read_csv(vertices_csv)

    g = ig.Graph(directed=True)
    g.add_vertices(v_df.shape[0])

    # 顶点属性
    for col in ("x", "y", "z", "pressure"):
        if col in v_df.columns:
            g.vs[col] = v_df[col].tolist()

    # 边 & 属性
    g.add_edges(list(zip(e_df['vertex_1'], e_df['vertex_2'])))
    for col in e_df.columns:
        if col not in ('vertex_1', 'vertex_2'):
            g.es[col] = e_df[col].tolist()

    # 计算流体学量
    _add_flow_attributes(g, plasma_visc=plasma_viscosity_pa_s)
    return g


def build_graph_from_mvn(
    mvn_csv: Union[Path, str],
    *,
    out_dir: Union[Path, str] = Path('output'),
    plasma_viscosity_pa_s: float = 3.5e-3,
) -> ig.Graph:
    """
    从 MVN Excel (CSV) 构建血管网络：
      1. 读取 mvn_csv, 需包含: 'index', 'diameters', 'pressure', 'coords:0','coords:1','coords:2'
      2. 每行作为一个节点, 自动分配 node_id
      3. 顶点属性: x, y, z, pressure
      4. 按 'index' 分组, 对每组排序后连接相邻节点作为边
      5. 边属性: diameter(两端平均), pressure_diff, length
      6. 调用 Poiseuille 公式计算 flow_rate 和 rbc_velocity
      7. 序列化 igraph.Graph 到 PKL: out_dir/complete_network.pkl
      8. 用 PyVista 导出完整 vtk: out_dir/complete_network.vtk
    """
    mvn_path = Path(mvn_csv)
    if not mvn_path.exists():
        raise FileNotFoundError(f"缺少 MVN 文件: {mvn_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mvn_path)
    required_cols = {'index', 'diameters', 'pressure', 'coords:0', 'coords:1', 'coords:2'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[build_graph_from_mvn] 缺失列: {missing}")

    # 分配 node_id
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'node_id'})

    # 顶点属性
    xs = df['coords:0'].tolist()
    ys = df['coords:1'].tolist()
    zs = df['coords:2'].tolist()
    pressures = df['pressure'].tolist()

    g = ig.Graph(directed=True)
    g.add_vertices(len(df))
    g.vs['x'] = xs
    g.vs['y'] = ys
    g.vs['z'] = zs
    g.vs['pressure'] = pressures

    # 构建边 & 收集管径
    edge_list = []
    diam_list = []
    for _, group in df.groupby('index'):
        grp = group.sort_values('node_id')
        nodes = grp['node_id'].tolist()
        diams = grp['diameters'].tolist()
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            edge_list.append((u, v))
            diam_list.append((diams[i] + diams[i+1]) / 2)

    g.add_edges(edge_list)
    g.es['diameter'] = diam_list

    # 计算压力差、长度、流量等
    _ensure_pressure_diff(g)
    _ensure_length(g)
    _add_flow_attributes(g, plasma_visc=plasma_viscosity_pa_s)

    # 保存 Graph 对象
    pkl_file = out_dir / 'complete_network.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(g, f)
    print(f"[graph_builder] Pickled graph → {pkl_file}")

    # 导出 VTK
    points = list(zip(xs, ys, zs))
    mesh = pv.PolyData(points)
    lines = []
    for u, v in edge_list:
        lines.extend([2, u, v])
    mesh.lines = lines
    vtk_file = out_dir / 'complete_network.vtk'
    mesh.save(str(vtk_file))
    print(f"[graph_builder] VTK exported → {vtk_file}")

    return g