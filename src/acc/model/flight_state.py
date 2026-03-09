import numpy as np
from pydantic import BaseModel, ConfigDict


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
    throttle: np.ndarray | None = None  # normalized 0–1
