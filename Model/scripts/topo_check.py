
import argparse
from pathlib import Path
from tracking_model.topology_qc import run_topology_qc


def main() -> None:
    p = argparse.ArgumentParser(description="Vertex connectivity QC.")
    p.add_argument("--edge", type=Path, required=True)
    p.add_argument("--out",  type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_topology_qc(args.edge, args.out)
    print("Topology QC finished.")


if __name__ == "__main__":
    main()
