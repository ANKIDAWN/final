
import argparse
from pathlib import Path

from tracking_model.io_loaders import load_edge_vertex, save_edge
from tracking_model.graph_builder import build_graph
from tracking_model.bifurcation_model import assign_rbc_probabilities


def main() -> None:
    p = argparse.ArgumentParser(description="RBC-probability assignment only.")
    p.add_argument("--edge",   type=Path, required=True)
    p.add_argument("--vertex", type=Path, required=True)
    p.add_argument("--out",    type=Path, required=True, help="output edge CSV")
    args = p.parse_args()

    edge_df, vertex_df = load_edge_vertex(args.edge, args.vertex)
    g                  = build_graph(None, edge_df, vertex_df)          # MVN not needed
    assign_rbc_probabilities(g)

    save_edge(g, args.out)
    print(f"Wrote updated edge file → {args.out}")


if __name__ == "__main__":
    main()
