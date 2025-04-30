"""
tracking_model/path_enumeration.py

BFS path enumeration & per-path CSV export
Enumerate all simple paths from a specified starting point and output segmented CSV for each path at fixed time_step
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Callable
import csv
import igraph as ig
import numpy as np
from tqdm import tqdm


__all__ = ["enumerate_all_paths"]


class Path(NamedTuple):
    """Simplified path representation"""
    nodes: list[int]
    times: list[float]


def enumerate_all_paths(
    g: ig.Graph,
    *,
    start: int,
    speed: float,
    timestep: float,
    out_dir: Optional[os.PathLike] = None,
) -> Dict[int, Path]:
    """
    Enumerate all simple paths starting from 'start' and generate position records CSV for each path at fixed timestep.

    Args:
        g: Directed graph, must contain edge attribute 'length'
        start: Starting vertex index
        speed: Movement speed (m/s)
        timestep: Sampling interval (s)
        out_dir: Output directory, will generate path_1.csv, path_2.csv, ...

    Returns:
        Dictionary of path_id -> Path
    """
    # Build adjacency list: u -> list of (v, length)
    adj: Dict[int, list[tuple[int, float]]] = {}
    for e in g.es:
        u, v = e.source, e.target
        if u not in adj: adj[u] = []
        adj[u].append((v, e["length"]))

    # BFS to enumerate simple paths, record cumulative time
    path_id = 0
    paths: Dict[int, Path] = {}
    q = [(start, [start], [0.0])]  # (current, path, times)
    seen = set([start])

    with tqdm(desc="BFS paths", unit=" paths") as pbar:
        while q:
            node, path, times = q.pop(0)
            if node not in adj:  # Leaf node
                path_id += 1
                paths[path_id] = Path(nodes=path, times=times)
                pbar.update(1)
                seen.remove(node)
                continue

            any_pushed = False
            for nbr, length in adj[node]:
                if nbr in seen:
                    continue
                # Extend path
                if length <= 0:
                    time_delta = timestep  # Safety default
                else:
                    time_delta = length / (speed or 1.0)  # Time to traverse edge
                new_time = times[-1] + time_delta
                new_path = path + [nbr]
                new_times = times + [new_time]
                q.append((nbr, new_path, new_times))
                seen.add(nbr)
                any_pushed = True

            if not any_pushed:
                seen.remove(node)  # Backtrack from this branch

    # Write paths to CSV files
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        points = g.vs["x", "y", "z"]

        for path_id, (nodes, times) in tqdm(
            paths.items(), desc="Saving paths", unit=" files"
        ):
            # Create a CSV file with time-based positions on this path
            interp_times = np.arange(times[0], times[-1], timestep)
            with open(out_path / f"path_{path_id}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "node_id", "x", "y", "z"])
                # Simple linear interpolation for intermediate points
                for t in interp_times:
                    # Find section containing time t
                    for i in range(len(times) - 1):
                        if times[i] <= t < times[i + 1]:
                            n1, n2 = nodes[i], nodes[i + 1]
                            t1, t2 = times[i], times[i + 1]
                            p1, p2 = points[n1], points[n2]
                            # Linear interpolation
                            alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                            px = p1[0] + alpha * (p2[0] - p1[0])
                            py = p1[1] + alpha * (p2[1] - p1[1])
                            pz = p1[2] + alpha * (p2[2] - p1[2])
                            writer.writerow([t, -1, px, py, pz])
                            break
                    else:
                        # Handle the case of t == times[-1]
                        if abs(t - times[-1]) < 1e-6:
                            n = nodes[-1]
                            p = points[n]
                            writer.writerow([t, n, p[0], p[1], p[2]])

    return paths