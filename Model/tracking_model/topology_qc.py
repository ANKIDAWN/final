
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import pandas as pd
import igraph as ig

__all__ = [
    "generate_connectivity_reports",
    "run_topology_qc",
]

def generate_connectivity_reports(
    edge_csv: Path | str,
    out_vertex1: Path | str,
    out_vertex2: Path | str,
) -> None:
   
    edge_csv   = Path(edge_csv)
    out_vertex1 = Path(out_vertex1)
    out_vertex2 = Path(out_vertex2)

    df = pd.read_csv(edge_csv)

    # -------- vertex_1 → vertex_2 列表 --------
    conn1: dict[int, set[int]] = defaultdict(set)
    conn2: dict[int, set[int]] = defaultdict(set)

    for _, r in df.iterrows():
        conn1[int(r.vertex_1)].add(int(r.vertex_2))
        conn2[int(r.vertex_2)].add(int(r.vertex_1))

    # flag helper
    _labels = ["", "WRONG", "WRONG_WRONG", "WRONG_WRONG_WRONG",
               "WRONG_WRONG_WRONG_WRONG", "WRONG_WRONG_WRONG_WRONG_WRONG"]

    rows1 = [
        (v, ", ".join(map(str, sorted(neigh))), _labels[min(len(neigh), 5)-1])
        for v, neigh in conn1.items()
    ]
    out_vertex1.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows1, columns=["vertex_1", "connected_vertex_2", "QA_flag"]) \
      .to_csv(out_vertex1, index=False)

    # -------- vertex_2 ← vertex_1 计数 --------
    rows2 = [
        (v, len(neigh), _labels[min(len(neigh), 5)-1])
        for v, neigh in conn2.items()
    ]
    pd.DataFrame(rows2,
                 columns=["vertex_2", "connected_vertex_1_count", "QA_flag"]) \
      .to_csv(out_vertex2, index=False)

    print(f"[QC] vertex_1 / vertex_2 reports written → {out_vertex1.parent}")



def _topology_qc_dataframe(g: ig.Graph) -> pd.DataFrame:
    

    deg_total = g.degree(mode="ALL")
    neighbours = [g.neighbors(v.index, mode="ALL") for v in g.vs]

    df = pd.DataFrame({
        "vertex_id":        range(g.vcount()),
        "degree":           deg_total,
        "is_isolated":      [d == 0 for d in deg_total],
        "is_dead_end":      [d == 1 for d in deg_total],
        "is_branch":        [d >= 3 for d in deg_total],
        "neigh_ids":        [", ".join(map(str, nbs)) for nbs in neighbours],
    })
    return df


def run_topology_qc(
    g: ig.Graph,
    out_file: Path | str = Path("output/qc/topology_qc.csv"),
) -> None:
   
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = _topology_qc_dataframe(g)
    df.to_csv(out_file, index=False)

    print(f"[QC] topology_qc.csv → {out_file}")
