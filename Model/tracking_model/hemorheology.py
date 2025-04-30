"""
hemorheology.py
===============

包含血液相对粘度、Fahraeus 效应，以及全局压力求解和边属性计算。
"""
from __future__ import annotations
import numpy as np
import igraph as ig
from typing import List, Union

__all__: List[str] = [
    "mu_rel_FahraeusLindqvist",
    "fahraeus_effect",
    "solve_pressures",
    "compute_edge_hemo",
    "solve_pressures_igraph",
    "compute_edge_hemo_igraph",
]

# ------------------------------------------------------------ #
#  viscosity helpers
# ------------------------------------------------------------ #
def mu_rel_FahraeusLindqvist(d_um: float, H_D: float) -> float:
    """
    relative viscosity  µ_rel  (Pries+Gaehtgens 1992)
    d_um : vessel inner diameter [µm]
    H_D  : discharge hematocrit (0-1)
    """
    mu_45 = 220 * np.exp(-1.3 * d_um) + 3.2 - 2.44 * np.exp(-0.06 * d_um**0.645)
    C = (0.8 + np.exp(-0.075 * d_um)) * (1 / (1 + 1e-11 * d_um**12))
    return 1 + (mu_45 - 1) * ((1 - H_D)**C - 1) / ((1 - 0.45)**C - 1)


def fahraeus_effect(d_um: float, H_D: float) -> float:
    """
    Fahraeus 效应：速度比 HT/HD (Pries 1990)
    """
    return 1 + (1 - H_D) * (1 - 1 / (1 + 0.1 * d_um)**2)

# ------------------------------------------------------------ #
#  Edge-wise computation for NetworkX
# ------------------------------------------------------------ #
import networkx as nx

def compute_edge_hemo(
        g: nx.DiGraph,
        plasma_visc: float = 1.2e-3,      # Pa·s
    ) -> None:
    """
    对 nx.DiGraph 里的每条 edge 写入：
       mu_eff, pressure_diff, flow_rate, rbc_velocity
    要求节点已有 'pressure' 属性，edge 已有 'length','diameter'.
    """
    for u, v, ed in g.edges(data=True):
        L = ed["length"]
        d = ed["diameter"]
        H_D = ed.get("ht", 0.40)
        d_um = d * 1e6
        mu_rel = mu_rel_FahraeusLindqvist(d_um, H_D)
        mu_eff = plasma_visc * mu_rel
        ed["mu_eff"] = mu_eff
        # pressure_diff
        dp = g.nodes[u]["pressure"] - g.nodes[v]["pressure"]
        ed["pressure_diff"] = dp
        # flow_rate
        Q = (np.pi * d**4 * dp) / (128 * mu_eff * L)
        ed["flow_rate"] = Q
        # rbc_velocity
        v_bulk = Q / (np.pi * (d / 2)**2)
        ed["rbc_velocity"] = v_bulk / fahraeus_effect(d_um, H_D)

# ------------------------------------------------------------ #
#  Global pressure solver for NetworkX
# ------------------------------------------------------------ #

def solve_pressures(
        g: nx.DiGraph,
        inlet: List[int],
        outlet: List[int],
        p_in: float = 40e3,
        p_out: float = 10e3,
    ) -> None:
    """
    线性水管网络求解：ΣQ=0 方程组，已知 inlet/outlet 压力。
    将解赋给节点属性 'pressure'.
    """
    # 边界条件
    for n in inlet:
        g.nodes[n]["pressure"] = p_in
    for n in outlet:
        g.nodes[n]["pressure"] = p_out
    # 未知节点
    unknowns = [n for n in g.nodes if n not in inlet + outlet]
    nU = len(unknowns)
    if nU == 0:
        return
    A = np.zeros((nU, nU))
    b = np.zeros(nU)
    idx = {n: i for i, n in enumerate(unknowns)}
    # 构造线性方程 A p = b
    for n in unknowns:
        row = idx[n]
        # 出边
        for nbr, ed in g[n].items():
            R = 128 * ed["length"] * ed.get("mu_eff", plasma_visc) / (np.pi * ed["diameter"]**4)
            if nbr in idx:
                A[row, idx[nbr]] -= 1 / R
            else:
                b[row] += g.nodes[nbr]["pressure"] / R
            A[row, row] += 1 / R
    # 求解
    p = np.linalg.solve(A, b)
    for n, val in zip(unknowns, p):
        g.nodes[n]["pressure"] = float(val)

# ------------------------------------------------------------ #
#  Edge-wise computation for igraph
# ------------------------------------------------------------ #

def compute_edge_hemo_igraph(
        g: ig.Graph,
        plasma_visc: float = 1.2e-3,
    ) -> None:
    """
    与 compute_edge_hemo 类似，但作用于 igraph.Graph.
    要求 g.vs['pressure'], g.es['length'], g.es['diameter'] 已存在。
    """
    # 确保压力差与长度
    if 'pressure_diff' not in g.es.attributes():
        # compute from node pressures
        pres = g.vs['pressure']
        g.es['pressure_diff'] = [
            pres[e.tuple[0]] - pres[e.tuple[1]] for e in g.es
        ]
    if 'length' not in g.es.attributes():
        xs, ys, zs = g.vs['x'], g.vs['y'], g.vs['z']
        g.es['length'] = [
            math.sqrt(
                (xs[e.tuple[0]]-xs[e.tuple[1]])**2 +
                (ys[e.tuple[0]]-ys[e.tuple[1]])**2 +
                (zs[e.tuple[0]]-zs[e.tuple[1]])**2
            ) for e in g.es
        ]
    # 计算每条边属性
    mu_effs = []
    flow_rates = []
    rbc_vs = []
    for e in g.es:
        d = e['diameter']
        L = e['length']
        dp = e['pressure_diff']
        H_D = e.attributes().get('ht', 0.40)
        d_um = d * 1e6
        mu_rel = mu_rel_FahraeusLindqvist(d_um, H_D)
        mu_eff = plasma_visc * mu_rel
        mu_effs.append(mu_eff)
        Q = (np.pi * d**4 * dp) / (128 * mu_eff * L)
        flow_rates.append(Q)
        # 使用 Fahraeus 效应调整
        v_bulk = Q / (np.pi * (d / 2)**2)
        v_rbc = v_bulk / fahraeus_effect(d_um, H_D)
        rbc_vs.append(v_rbc)
    g.es['mu_eff'] = mu_effs
    g.es['flow_rate'] = flow_rates
    g.es['rbc_velocity'] = rbc_vs

# ------------------------------------------------------------ #
#  Global pressure solver for igraph
# ------------------------------------------------------------ #

def solve_pressures_igraph(
        g: ig.Graph,
        inlet: List[int],
        outlet: List[int],
        p_in: float = 40e3,
        p_out: float = 10e3,
        plasma_visc: float = 1.2e-3,
    ) -> None:
    """
    对 igraph.Graph 求解节点压力，ΣQ=0.
    """
    # 边界条件
    for v in inlet:
        g.vs[v]['pressure'] = p_in
    for v in outlet:
        g.vs[v]['pressure'] = p_out
    # 未知节点
    unknowns = [v.index for v in g.vs if v.index not in inlet + outlet]
    nU = len(unknowns)
    if nU == 0:
        return
    A = np.zeros((nU, nU))
    b = np.zeros(nU)
    idx = {n: i for i, n in enumerate(unknowns)}
    # 构造方程
    for n in unknowns:
        row = idx[n]
        # 所有出边
        for e in g.es.select(_source=n):
            nbr = e.target
            mu_eff = e['mu_eff'] if 'mu_eff' in e.attribute_names() else plasma_visc
            R = 128 * e['length'] * mu_eff / (np.pi * e['diameter']**4)
            if nbr in idx:
                A[row, idx[nbr]] -= 1 / R
            else:
                b[row] += g.vs[nbr]['pressure'] / R
            A[row, row] += 1 / R
        # 所有入边
        for e in g.es.select(_target=n):
            nbr = e.source
            mu_eff = e['mu_eff'] if 'mu_eff' in e.attribute_names() else plasma_visc
            R = 128 * e['length'] * mu_eff / (np.pi * e['diameter']**4)
            if nbr in idx:
                A[row, idx[nbr]] -= 1 / R
            else:
                b[row] += g.vs[nbr]['pressure'] / R
            A[row, row] += 1 / R
    # 求解并写回
    p = np.linalg.solve(A, b)
    for n, val in zip(unknowns, p):
        g.vs[n]['pressure'] = float(val)