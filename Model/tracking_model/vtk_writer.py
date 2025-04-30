

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np          # noqa: F401  
import igraph as ig
import vtk
from tqdm import tqdm


def _create_time_array(time_value: float) -> vtk.vtkDoubleArray:
    arr = vtk.vtkDoubleArray()
    arr.SetName("TimeValue")
    arr.SetNumberOfTuples(1)
    arr.SetValue(0, time_value)
    return arr


def export_rbc_positions_to_vtk(
    rbc_positions: List[Dict[str, Any]],
    output_file: Path,
    *,
    time_value: float = 0.0,
) -> None:
    """
    Save one simulation frame of RBCs as a *.vtp* point-cloud.

    Parameters
    ----------
    rbc_positions : list[dict]
        Dicts must contain ``id`` and ``position`` (x,y,z).
    output_file : Path
        Destination *.vtp* filename.
    time_value : float, optional (default=0)
        Stored to field-data so ParaView can treat files as a time-series.
    """
    try:
        # --- points + scalar ID array ---------------------------------
        points = vtk.vtkPoints()
        rbc_id = vtk.vtkIntArray()
        rbc_id.SetName("RBC_ID")

        for rbc in rbc_positions:
            x, y, z = rbc["position"]
            points.InsertNextPoint(x, y, z)
            rbc_id.InsertNextValue(int(rbc["id"]))

        # --- polydata + field-data ------------------------------------
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.GetPointData().AddArray(rbc_id)
        poly.GetFieldData().AddArray(_create_time_array(time_value))

        # --- write ----------------------------------------------------
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_file))
        writer.SetInputData(poly)
        writer.Write()

    except Exception as exc:     # pragma: no-cover
        print(f"[VTK-writer] Failed exporting {output_file}: {exc}")


def export_vessel_network_to_vtk(
    graph: ig.Graph,
    output_file: Path,
) -> None:
   
    try:
        print(f"[VTK-writer] Exporting network → {output_file}")
        points = vtk.vtkPoints()
        point_lookup: dict[tuple[float, float, float], int] = {}
        lines = vtk.vtkCellArray()
        diam_arr = vtk.vtkFloatArray()
        diam_arr.SetName("Diameter")

        # ---- iterate over edges exactly like original ----------------
        for e in tqdm(graph.es, desc="building vtk", ascii=True, ncols=80):
            coords = e["coords"]              # list[(x,y,z)] – already np.float
            id_list = vtk.vtkIdList()

            for xyz in coords:
                key = tuple(xyz)
                if key not in point_lookup:
                    pid = points.InsertNextPoint(*xyz)
                    point_lookup[key] = pid
                id_list.InsertNextId(point_lookup[key])

            lines.InsertNextCell(id_list)
            diam_arr.InsertNextValue(e["diameter"])

        # ---- assemble polydata --------------------------------------
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.GetCellData().AddArray(diam_arr)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_file))
        writer.SetInputData(poly)
        writer.Write()

    except Exception as exc:     # pragma: no-cover
        print(f"[VTK-writer] Failed exporting {output_file}: {exc}")


export_vessel_network = export_vessel_network_to_vtk


__all__ = [
    "export_rbc_positions_to_vtk",
    "export_vessel_network",          # primary
    "export_vessel_network_to_vtk",   
]
