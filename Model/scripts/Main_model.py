#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end RBC-tracking pipeline (full feature set) with progress bars.

Steps:
  1) Build igraph (MVN or edge/vertex CSV)
  2) (Optional) Inverse-model BC tuning
  3) Pressure solve & hemo compute
  4) Phase-separation (flow | logit)
  5) Topology QA
  6) (Optional) Enumerate paths
  7) RBC simulation (dynamic injection / static BFS)
  8) Export vessel network to VTK
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# graph & IO
from tracking_model.graph_builder    import build_graph_from_edges, build_graph_from_mvn
from tracking_model.io_loaders       import save_edges_csv, save_vertices_csv
# inverse-model
from tracking_model.inverse_model    import tune_boundary_conditions
# hemodynamics
from tracking_model.hemorheology     import solve_pressures_igraph, compute_edge_hemo_igraph
# phase separation
from tracking_model.phase_separation import assign_rbc_probabilities
# topology QC
from tracking_model.topology_qc      import run_topology_qc
# path enumeration
from tracking_model.path_enumeration import enumerate_all_paths
# simulation
from tracking_model.queue_sim        import run_queue_sim, run_dynamic_sim
# VTK export
from tracking_model.vtk_writer       import export_vessel_network


def run_full_pipeline(
    *,
    use_mvn: bool,
    mvn_csv: Path | None,
    edges_csv: Path | None,
    vertices_csv: Path | None,
    out_dir: Path,
    tune_bc: bool,
    inverse_model_dir: Path,
    split_mode: str,
    enumerate_paths: bool,
    start_node: int,
    path_speed: float,
    path_timestep: float,
    dynamic: bool,
    inlet_edges: list[int],
    rbc_rates: list[float],
    sim_timestep: float,
    sim_total_time: float,
    steady_window: tuple[float, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Total of 8 stages
    stage_bar = tqdm(total=8,
                     desc="Pipeline",
                     ncols=80,
                     bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    overall_start = time.time()

    # 1) Build graph
    if use_mvn:
        g = build_graph_from_mvn(mvn_csv, out_dir=out_dir)
    else:
        g = build_graph_from_edges(edges_csv, vertices_csv=vertices_csv)
        save_edges_csv(g, out_dir / "network_edges.csv")
        save_vertices_csv(g, out_dir / "network_vertices.csv")
    stage_bar.update(1)

    # 2) Inverse-model BC tuning (optional)
    if tune_bc:
        g = tune_boundary_conditions(g, inverse_model_dir, out_dir / "tuned_edges.csv")
    stage_bar.update(1)

    # 3) Pressure solve & hemo compute
    entry_nodes = [v.index for v in g.vs if v.indegree(mode='in') == 0]
    exit_nodes  = [v.index for v in g.vs if v.outdegree(mode='out') == 0]
    solve_pressures_igraph(g, entry_nodes, exit_nodes)
    compute_edge_hemo_igraph(g)
    stage_bar.update(1)

    # 4) Phase-separation
    assign_rbc_probabilities(g, mode=split_mode)
    stage_bar.update(1)

    # 5) Topology QA
    run_topology_qc(g, out_dir / "topology_QA.csv")
    stage_bar.update(1)

    # 6) Enumerate paths (optional)
    if enumerate_paths:
        enumerate_all_paths(
            g,
            start=start_node,
            speed=path_speed,
            timestep=path_timestep,
            out_dir=out_dir / "paths"
        )
    stage_bar.update(1)

    # 7) RBC simulation
    if dynamic:
        # We let run_dynamic_sim auto-detect entry/exit and compute rates
        df_tracks = run_dynamic_sim(
            g,
            time_step   = sim_timestep,
            total_time  = sim_total_time,
            out_vtp_dir = out_dir / "rbc_vtp"
        )

        # --- Always write complete trajectory table ---
        out_tracks = out_dir / "rbc_tracks.csv"
        df_tracks.to_csv(out_tracks, index=False)
        print(f"[Main] wrote full tracks → {out_tracks}")

        # --- Optional: Steady-state extraction ---
        try:
            t0, t1 = steady_window
            steady_df = df_tracks[
                (df_tracks['time'] >= t0) & (df_tracks['time'] < t1)
            ].reset_index(drop=True)
            out_steady = out_dir / "steady_tracks.csv"
            steady_df.to_csv(out_steady, index=False)
            print(f"[Main] wrote steady-state tracks → {out_steady}")
        except KeyError:
            print("[Warning] no 'time' column; skipping steady-state export")
    else:
        # Static BFS branch (compatible with old interface)
        run_queue_sim(
            g,
            tracks_csv    = out_dir / "rbc_tracks.csv",
            steady_csv    = out_dir / "steady_tracks.csv",
            speed         = 1.0,
            timestep      = sim_timestep,
            start_vertex  = start_node,
            steady_window = steady_window,
        )
    stage_bar.update(1)

    # 8) Export vessel network
    export_vessel_network(g, out_dir / "vessel_network.vtp")
    stage_bar.update(1)

    stage_bar.close()
    total = time.time() - overall_start
    print(f"[Main] ✅ Pipeline finished in {total:.1f}s. See {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RBC-tracking full pipeline")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--mvn",    type=Path, help="Use MVN CSV to build graph")
    grp.add_argument("--edges",  type=Path, help="Edge CSV (requires --vertices)")
    p.add_argument("--vertices", type=Path, help="Vertex CSV (with --edges)")
    p.add_argument("--out",      dest="out_dir",      type=Path, default=Path("output"), help="Output directory")
    p.add_argument("--tune-bc",  action="store_true", help="Run inverse-model BC tuning first")
    p.add_argument("--inverse-model-dir", type=Path, default=Path("data/inverse_model"), help="Directory for inverse_model CSVs")
    p.add_argument("--split-mode", choices=("flow","logit"), default="flow", help="Branch split mode")
    p.add_argument("--enumerate-paths",  action="store_true", help="Enumerate all simple paths")
    p.add_argument("--start-node",    type=int,   default=0,   help="Start node for BFS / path enumeration")
    p.add_argument("--path-speed",    type=float, default=0.5, help="Speed for path enumeration (m/s)")
    p.add_argument("--path-timestep", type=float, default=0.1, help="Timestep for path enumeration (s)")
    p.add_argument("--dynamic",       action="store_true",    help="Run dynamic injection sim")
    p.add_argument("--sim-timestep",  type=float, default=0.1,  help="Simulation timestep (s)")
    p.add_argument("--sim-total-time",type=float, default=30.0, help="Total sim time (s)")
    p.add_argument("--steady", nargs=2, type=float, default=(1.0,2.0), help="Steady-state window (s)")
    args = p.parse_args()

    if args.edges and not args.vertices:
        sys.exit("❌ --edges requires --vertices")

    run_full_pipeline(
        use_mvn           = bool(args.mvn),
        mvn_csv           = args.mvn,
        edges_csv         = args.edges,
        vertices_csv      = args.vertices,
        out_dir           = args.out_dir,
        tune_bc           = args.tune_bc,
        inverse_model_dir = args.inverse_model_dir,
        split_mode        = args.split_mode,
        enumerate_paths   = args.enumerate_paths,
        start_node        = args.start_node,
        path_speed        = args.path_speed,
        path_timestep     = args.path_timestep,
        dynamic           = args.dynamic,
        inlet_edges       = [],   # not used
        rbc_rates         = [],   # not used
        sim_timestep      = args.sim_timestep,
        sim_total_time    = args.sim_total_time,
        steady_window     = tuple(args.steady),
    )
