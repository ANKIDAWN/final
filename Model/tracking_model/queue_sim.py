# tracking_model/queue_sim.py

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import deque

import numpy as np
import igraph as ig
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# 内部：三维线性插值
# ----------------------------------------------------------------------
def _interp(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    alpha: float
) -> Tuple[float, float, float]:
    return (
        p1[0] + alpha * (p2[0] - p1[0]),
        p1[1] + alpha * (p2[1] - p1[1]),
        p1[2] + alpha * (p2[2] - p1[2]),
    )

# ----------------------------------------------------------------------
# 公共 API 1：静态 BFS → DataFrame
# ----------------------------------------------------------------------
def tracks_from_graph(
    g: ig.Graph,
    *,
    start_vertex: int | None = None,
    speed: float = 0.5,
    timestep: float = 0.5,
    progress: bool = True
) -> pd.DataFrame:
    if start_vertex is None:
        start_vertex = 0

    coords = list(zip(g.vs["x"], g.vs["y"], g.vs["z"]))
    lengths = g.es["length"]

    # 构建邻接表：u -> [(v, length, eid), ...]
    adj: Dict[int, List[Tuple[int, float, int]]] = {}
    for eid, e in enumerate(g.es):
        u, v = e.tuple
        adj.setdefault(u, []).append((v, float(lengths[eid]), eid))

    q = deque([(start_vertex, 0.0)])
    records: List[Dict[str, Any]] = []
    frame_no = 0

    pbar = tqdm(total=g.ecount(), desc="Static BFS", ncols=80, disable=not progress)

    while q:
        u, t0 = q.popleft()
        for v, L, eid in adj.get(u, []):
            dt = L / speed
            t1 = t0 + dt
            q.append((v, t1))

            steps = max(1, math.ceil(dt / timestep))
            for k in range(steps + 1):
                alpha = k / steps
                x, y, z = _interp(coords[u], coords[v], alpha)
                records.append({
                    "frame": frame_no,
                    "time":  t0 + alpha * dt,
                    "from":  u,
                    "to":    v,
                    "x":     x,
                    "y":     y,
                    "z":     z,
                })
                frame_no += 1

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records, dtype="float64")


# ----------------------------------------------------------------------
# 公共 API 2：静态 BFS 写 CSV（兼容旧接口）
# ----------------------------------------------------------------------
def run_queue_sim(
    g: ig.Graph,
    *,
    tracks_csv: Path,
    steady_csv: Path | None = None,
    speed: float = 0.5,
    timestep: float = 0.5,
    start_vertex: int | None = None,
    steady_window: Tuple[float, float] = (1.0, 2.0),
    progress: bool = True
) -> None:
    df = tracks_from_graph(
        g,
        start_vertex=start_vertex,
        speed=speed,
        timestep=timestep,
        progress=progress,
    )

    tracks_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tracks_csv, index=False)
    print(f"[queue] wrote → {tracks_csv}")

    if steady_csv:
        t0, t1 = steady_window
        steady_df = df[(df["time"] >= t0) & (df["time"] < t1)].reset_index(drop=True)
        steady_csv.parent.mkdir(parents=True, exist_ok=True)
        steady_df.to_csv(steady_csv, index=False)
        print(f"[queue] wrote steady window → {steady_csv}")


# ----------------------------------------------------------------------
# 公共 API 3：动态注入 & RBC 追踪（按“边”注入）
# ----------------------------------------------------------------------
def run_dynamic_sim(
    g: ig.Graph,
    *,
    time_step: float,
    total_time: float,
    out_vtp_dir: Path,
    C_RBC: float = 5e14  # cells/m³
) -> pd.DataFrame:
    """
    1) 按每条边的流量方向计算每个节点 Q_in/Q_out
    2) 将有 Q_in>Q_out 的节点标为入口，收集其所有出边作为入口边
    3) 入口边的注入速率 R_e = |flow_rate|*C_RBC（cells/s）
    4) 每个 time_step 累计“欠注数”，>=1 时在该边源端生成 RBC，随机选一条出边进入网络
    5) RBC 在边上以 rbc_velocity 推进，到端点再按 rbc_probability 分支或出网
    6) 记录每个 RBC 每步的 (rbc_id, time, x, y, z)
    """

    # —— 1) Q_in/Q_out & 入口/出口识别 —— #
    n = g.vcount()
    Q_in  = np.zeros(n)
    Q_out = np.zeros(n)
    for e in g.es:
        u, v = e.tuple
        q = e["flow_rate"]
        if q >= 0:
            Q_out[u] += q
            Q_in[v]  += q
        else:
            Q_out[v] += -q
            Q_in[u]  += -q

    exit_nodes  = {v.index for v in g.vs if v.outdegree(mode="out") == 0}
    entry_nodes = [i for i in range(n) if Q_in[i] > Q_out[i] and i not in exit_nodes]

    # —— 2) 由入口节点→入口边 & 注入速率 —— #
    entry_edges = [e.index for e in g.es if e.tuple[0] in entry_nodes]
    entry_rates = {eid: abs(g.es[eid]["flow_rate"]) * C_RBC for eid in entry_edges}

    print(">>>>> using EDGE‐BASED run_dynamic_sim <<<<<")
    print("  entry_nodes:", entry_nodes[:10], "…")
    print("  entry_edges:", entry_edges[:10], "…")
    print("  entry_rates:", [entry_rates[eid] for eid in entry_edges[:10]], "…")

    # —— 3) 准备 & 循环 —— #
    steps   = int(total_time / time_step)
    coords  = np.vstack([g.vs["x"], g.vs["y"], g.vs["z"]]).T
    accum   = {eid: 0.0 for eid in entry_edges}
    active  : Dict[int, Dict[str, Any]] = {}
    records : List[Dict[str, Any]] = []
    next_id = 0

    out_vtp_dir.mkdir(parents=True, exist_ok=True)
    start_clock = time.time()

    pbar = tqdm(range(steps), desc="Dynamic Sim", ncols=80, unit="step")
    for step in pbar:
        t = step * time_step

        # — 3.1) 注入新 RBC — #
        for eid in entry_edges:
            accum[eid] += entry_rates[eid] * time_step
            n_new = int(accum[eid]); accum[eid] -= n_new
            if n_new <= 0:
                continue

            # 边源端坐标
            u, _ = g.es[eid].tuple
            x0, y0, z0 = coords[u]
            for _ in range(n_new):
                records.append({
                    "rbc_id": next_id,
                    "time":   t,
                    "x":      x0,
                    "y":      y0,
                    "z":      z0,
                })
                active[next_id] = {"edge_index": eid, "edge_pos": 0.0}
                next_id += 1

        # — 3.2) 推进 & 分叉 — #
        to_remove: List[int] = []
        for rid, info in list(active.items()):
            eid  = info["edge_index"]
            pos0 = info["edge_pos"]
            length = g.es[eid]["length"]
            vel    = abs(g.es[eid]["rbc_velocity"])
            pos1   = pos0 + vel * time_step

            # 到达或越过该边末端
            if pos1 >= length:
                u, w = g.es[eid].tuple
                # 到出口则移除
                if w in exit_nodes:
                    to_remove.append(rid)
                    continue
                # 在 w 分叉
                outs = g.es.select(_source=w)
                if not outs:
                    to_remove.append(rid)
                    continue
                probs = np.array([e["rbc_probability"] for e in outs], dtype=float)
                if probs.sum() > 0:
                    probs /= probs.sum()
                else:
                    probs[:] = 1.0 / len(outs)
                choice = np.random.choice(len(outs), p=probs)
                eid, pos1 = outs[choice].index, 0.0

            # 插值计算当前位置
            u2, w2 = g.es[eid].tuple
            frac    = pos1 / g.es[eid]["length"]
            x = coords[u2,0] + frac * (coords[w2,0] - coords[u2,0])
            y = coords[u2,1] + frac * (coords[w2,1] - coords[u2,1])
            z = coords[u2,2] + frac * (coords[w2,2] - coords[u2,2])

            records.append({
                "rbc_id": rid,
                "time":   t,
                "x":      x,
                "y":      y,
                "z":      z,
            })
            active[rid]["edge_index"] = eid
            active[rid]["edge_pos"]   = pos1

        # 删除出网的 RBC
        for rid in to_remove:
            del active[rid]

    elapsed = time.time() - start_clock
    print(f"\n[Sim] injected={next_id}, records={len(records)}, remaining={len(active)}")
    print(f"[Sim] duration={elapsed:.1f}s ({elapsed/steps:.3f}s/step)\n")

    return pd.DataFrame(records, columns=["rbc_id","time","x","y","z"])


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
__all__ = [
    "tracks_from_graph",
    "run_queue_sim",
    "run_dynamic_sim",
]
