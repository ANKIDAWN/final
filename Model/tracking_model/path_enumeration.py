
"""
BFS 全路径枚举 & per-path CSV 导出
从指定起点枚举所有简单路径，并按固定 time_step 输出每条路径的分段 CSV
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import igraph as ig


def enumerate_all_paths(
    g: ig.Graph,
    start: int,
    speed: float,
    timestep: float,
    out_dir: Path,
) -> None:
    """
    枚举从 start 出发的所有简单路径，并对每条路径按 timestep 生成位置记录 CSV。

    Parameters
    ----------
    g : ig.Graph
        有向图，需包含 edge 属性 'length'
    start : int
        起点顶点索引
    speed : float
        运动速度 (m/s)
    timestep : float
        采样间隔 (s)
    out_dir : Path
        输出目录，会生成 path_1.csv, path_2.csv, ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建邻接表: u -> list of (v, length)
    adj: dict[int, List[Tuple[int, float]]] = {}
    for e in g.es:
        u, v = e.tuple
        adj.setdefault(u, []).append((v, float(e['length'])))

    # BFS 枚举简单路径，记录累计耗时
    paths: List[Tuple[List[int], float]] = []
    queue = deque([(start, [start], 0.0)])
    while queue:
        node, path, cum_time = queue.popleft()
        for v, L in adj.get(node, []):
            if v in path:
                continue  # 避免环
            new_path = path + [v]
            dt = L / speed
            new_time = cum_time + dt
            paths.append((new_path, new_time))
            queue.append((v, new_path, new_time))

    # 对每条路径输出 CSV
    for idx, (path, total_time) in enumerate(paths, start=1):
        records = []
        t = 0.0
        rec_id = 1
        while t <= total_time:
            # 找到当前时刻 t 对应在哪条边上
            elapsed = 0.0
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # 获取边长度
                eid = g.get_eid(u, v)
                L = float(g.es[eid]['length'])
                dt = L / speed
                if elapsed <= t <= elapsed + dt:
                    pos_on_edge = (t - elapsed) * speed
                    records.append([
                        rec_id,
                        t,
                        u,
                        v,
                        pos_on_edge,
                        path.copy(),  # 完整路径
                    ])
                    rec_id += 1
                    break
                elapsed += dt
            t += timestep
        # 保存 CSV
        df = pd.DataFrame(
            records,
            columns=['id', 'timestep', 'start_node', 'end_node', 'position_on_edge', 'path']
        )
        csv_path = out_dir / f"path_{idx}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[path_enum] wrote → {csv_path}")