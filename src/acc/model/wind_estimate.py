import numpy as np
from pydantic import BaseModel, ConfigDict


class WindEstimate(BaseModel):
    """Output of the Johansen (2015) wind velocity observer.

    All arrays share the same time grid as the input FlightState.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    v_wind_north: np.ndarray  # estimated wind, NED north component (m/s)
    v_wind_east: np.ndarray   # estimated wind, NED east component (m/s)
    v_wind_down: np.ndarray   # estimated wind, NED down component (m/s)
    gamma: np.ndarray         # pitot calibration factor (dimensionless; 1.0 = no error)
    v_rel_north: np.ndarray   # wind-relative NED velocity, north (m/s)
    v_rel_east: np.ndarray    # wind-relative NED velocity, east (m/s)
    v_rel_down: np.ndarray    # wind-relative NED velocity, down (m/s)
