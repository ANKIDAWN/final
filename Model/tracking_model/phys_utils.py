"""
* Fahraeus–Lindqvist apparent-viscosity model
* Hagen–Poiseuille volumetric flow-rate
* Bulk-to-RBC centre-line velocity conversion
"""

import numpy as np


def mu_rel_fahraeus_lindqvist(diam_m: float, h_d: float = 0.45) -> float:
    d_um = diam_m * 1e6                                # convert to µm
    mu_045 = 220 * np.exp(-1.3 * d_um) + 3.2 - 2.44 * np.exp(-0.06 * d_um ** 0.645)
    c_term = ((0.8 + np.exp(-0.075 * d_um))
              * (-1 + 1 / (1 + 1e-11 * d_um ** 12))
              + 1 / (1 + 1e-11 * d_um ** 12))
    return 1 + (mu_045 - 1) * ((1 - h_d) ** c_term - 1) / ((1 - 0.45) ** c_term - 1)


def flow_rate_poisseuille(diam_m: float, dp_pa: float, length_m: float,
                          mu_pa_s: float) -> float:
    r = diam_m / 2.0
    length_m = max(length_m, 1e-9)                     # avoid division by zero
    return np.pi * r ** 4 * abs(dp_pa) / (8 * mu_pa_s * length_m)


def rbc_centerline_velocity(flow_rate_m3s: float, diam_m: float) -> float:
    area = np.pi * (diam_m / 2.0) ** 2
    return flow_rate_m3s / area
