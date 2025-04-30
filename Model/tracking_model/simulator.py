"""
一个函数：run_simulation(g)  
  1. 若节点无 pressure → solve_pressures  
  2. compute_edge_hemo  
  3. phase split 概率写到 edge['rbc_prob']  
  4. 调 queue_sim.tracks_from_graph → 产出轨迹
"""

from __future__ import annotations
import networkx as nx
from collections import defaultdict

from .hemorheology import solve_pressures, compute_edge_hemo
from .phase_separation import rbc_split_fraction
from .queue_sim import tracks_from_graph      # 你已有的函数

def add_phase_separation(g: nx.DiGraph) -> None:
    g.es["rbc_prob"] = [1.0]*g.ecount()   # default
    # 先按流量比初始化
    for v in g.nodes:
        out_e = g.out_edges(v, data=True)
        if len(out_e)==2:
            Qa = abs(out_e[0][2]["flow_rate"]); Qb = abs(out_e[1][2]["flow_rate"])
            Da = out_e[0][2]["diameter"];       Db = out_e[1][2]["diameter"]
            Df = Da   # 近似
            Pa = rbc_split_fraction(Qa,Qb,Da,Db,Df)
            g.edges[out_e[0][0], out_e[0][1]]["rbc_prob"] = Pa
            g.edges[out_e[1][0], out_e[1][1]]["rbc_prob"] = 1-Pa


def run_simulation(g: nx.DiGraph,
                   inlet: list[int], outlet: list[int],
                   **queue_kwargs):
    # 1 pressure & hemo
    solve_pressures(g, inlet, outlet)
    compute_edge_hemo(g)

    # 2 phase-separation probability
    add_phase_separation(g)

    # 3 粒子模拟
    tracks_from_graph(g, **queue_kwargs)
