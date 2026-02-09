"""Pydantic data models for aerodynamic coefficient calculations."""

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AircraftConfig(BaseModel):
    """User-provided aircraft parameters."""

    mass: Annotated[float, Field(gt=0, description="Aircraft mass (kg)")]
    wing_area: Annotated[float, Field(gt=0, description="Wing area S (m²)")]
    wing_span: Annotated[float, Field(gt=0, description="Wing span b (m)")]
    mean_aero_chord: Annotated[
        float, Field(gt=0, description="Mean aerodynamic chord c̄ (m)")
    ]
    i_xx: Annotated[
        float, Field(gt=0, description="Moment of inertia about x-axis (kg·m²)")
    ]
    i_yy: Annotated[
        float, Field(gt=0, description="Moment of inertia about y-axis (kg·m²)")
    ]
    i_zz: Annotated[
        float, Field(gt=0, description="Moment of inertia about z-axis (kg·m²)")
    ]
    i_xz: Annotated[
        float, Field(default=0.0, description="Product of inertia I_xz (kg·m²)")
    ]


class AtmosphereConfig(BaseModel):
    """Atmospheric conditions configuration."""

    rho: Annotated[
        float | None,
        Field(
            default=None,
            description="Fixed air density (kg/m³). None to use ISA model.",
        ),
    ]
    temperature_offset: Annotated[
        float,
        Field(default=0.0, description="ISA temperature offset delta-T (K)"),
    ]


class FlightState(BaseModel):
    """Intermediate unified time-series data on a common interpolated time grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time: np.ndarray
    ax_body: np.ndarray
    ay_body: np.ndarray
    az_body: np.ndarray
    p: np.ndarray
    q: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    v_north: np.ndarray
    v_east: np.ndarray
    v_down: np.ndarray
    altitude: np.ndarray


class AeroCoefficients(BaseModel):
    """Output aerodynamic coefficients and supplementary data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cy: np.ndarray
    c_roll: np.ndarray
    cm: np.ndarray
    cn: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    dynamic_pressure: np.ndarray
    airspeed: np.ndarray
