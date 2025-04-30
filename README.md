# RBC Tracking Model

A toolkit for simulating and analyzing red blood cell movement in microvascular networks.

## Installation

Ensure you have Python 3.8 or later installed, then run:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install the project and its dependencies
pip install -e .
```

## Running the Model

The model accepts two types of input data:
- MVN format CSV file
- Or a pair of edge and vertex CSV files

Examples:

```bash
# Using edge and vertex data
python scripts/Main_model.py --edges data/network/network_simulated_edge_data.csv --vertices data/network/network_simulated_vertex_data.csv --out output

# Or using MVN data (if data contains required columns)
# python scripts/Main_model.py --mvn data/network/sample_mvn.csv --out output
```

## Output

Output will be generated in the specified directory, including:
- Red blood cell trajectory data (rbc_tracks.csv)
- Steady-state trajectory data (steady_tracks.csv)
- Network topology quality check results (topology_QA.csv)
- Edge and vertex data