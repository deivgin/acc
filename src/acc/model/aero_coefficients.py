from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from acc.model.wind_estimate import WindEstimate


class AeroCoefficients(BaseModel):
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
    wind_estimate: Optional[WindEstimate] = None
