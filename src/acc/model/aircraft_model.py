from typing import Annotated

from pydantic import BaseModel, Field


class AircraftModel(BaseModel):
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
    max_thrust: Annotated[
        float,
        Field(default=0.0, gt=0, description="Maximum static thrust (N)"),
    ]
