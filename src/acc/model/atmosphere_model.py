from typing import Annotated

from pydantic import BaseModel, Field


class AtmosphereModel(BaseModel):
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
