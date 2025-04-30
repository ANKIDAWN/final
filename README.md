
## Module Responsibilities

- **`src/io_loaders.py`**  
  - Centralized I/O: path checks, CSV loading, header/column alignment, missing‐value handling.

- **`src/graph_builder.py`**  
  - Read `edges.csv` + `vertices.csv` (or MVN spreadsheet), build an `igraph.Graph`.  
  - Attach vertex attributes (`x, y, z, pressure`) and edge attributes (`diameter, length, initial_flow, initial_velocity`).

- **`src/bifurcation_model.py`**  
  - For each branching vertex (out‐degree ≥ 2), compute RBC split probabilities using the A/B/X₀ formulas + logit model.  
  - Store result in `edge['rbc_probability']`.

- **`src/topology_qc.py`**  
  - Compute in‐/out‐degree for every vertex, identify anomalies (degree ≠ 1–3).  
  - Write out a CSV report (`topology_QA.csv`).

- **`src/phys_utils.py`**  
  - Poiseuille’s law, relative viscosity (Fåhræus‐Lindqvist), Fåhræus e↵ect, etc. as standalone functions.

- **`src/queue_simulator.py`**  
  - Breadth‐first “queue” simulator that propagates RBC particles through the directed graph.  
  - Samples positions at fixed `timestep` → outputs full trajectory CSV + optional steady‐state slice.

- **`src/vtk_writer.py`**  
  - Export vessel geometry as `.vtp` polydata with cell/point arrays.  
  - Export RBC point clouds per frame (`rbc_positions_000000.vtp`, …) and generate a `.pvd` collection file for ParaView animation.

- **`scripts/Main_model.py`**  
  - Parses command‐line args (`--edges`, `--vertices`, `--out`, `--dt`, `--steady`).  
  - Calls in sequence:  
    1. `build_graph_from_edges`  
    2. `run_topology_qc`  
    3. `assign_rbc_probabilities` (from `bifurcation_model`)  
    4. `run_queue_simulator`  
    5. `export_vessel_network`  

- **Other scripts** (`calc_prob_flowrate.py`, `topo_check.py`, `queue_simulate.py`)  
  - Provide convenient entry points to just one of the modules for testing or batch runs.

## Installation & Dependencies

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   # .venv\Scripts\activate.bat   # Windows
