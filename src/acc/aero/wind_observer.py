"""Johansen (2015) model-free wind velocity observer.

Estimates 3-D wind velocity and pitot calibration factor using a linear
time-varying Kalman filter driven only by GNSS velocity, AHRS attitude, and
a pitot-static tube measurement — no aerodynamic model required.

State:
    x = [v_wind_N, v_wind_E, v_wind_D, gamma]^T  (R^4)

System model (A = 0):
    x_dot = 0  (slowly time-varying wind + slowly drifting calibration factor)

Measurement equation (fixed pitot, body-x axis):
    y_k = u_gps_k = d1^T · R_nb_k · v_g^NED_k          (from GPS + AHRS)
    H_k = [d1^T · R_nb_k,  u_r^m_k]                     (1×4, time-varying)
    y_k = H_k · x + noise

where:
    d1       = (1, 0, 0)^T (body x-axis unit vector)
    R_nb     = NED-to-body rotation (3-2-1 Euler)
    u_r^m_k  = pitot measurement at time k (INPUT, not state)
    gamma    = unknown pitot scaling factor (STATE)

Note: d1^T · R_nb = [cos θ cos ψ, cos θ sin ψ, -sin θ] for any roll φ,
so only pitch θ and yaw ψ affect the wind observability.

KF discrete predict-update (per step k):
    P_pred   = P + Q · dt
    e        = y - H · x_hat
    S        = max(H · P_pred · H^T + R_meas, ε)   (scalar, floored)
    K        = P_pred · H^T / S
    x_hat   += K · e
    M        = I - K · H
    P        = M · P_pred · M^T + K · R_meas · K^T  (Joseph form)

Persistence of excitation condition (Johansen 2015, Sec. V):
    The aircraft must vary both pitch (θ) and yaw (ψ) during the estimation
    window.  Straight-and-level flight limits rank(W_o) = 2, yielding only
    horizontal wind components.  The vertical component (v_wind_D) requires
    pitch variations.

Reference:
    Johansen et al., "On estimation of wind velocity, angle-of-attack and
    sideslip angle of small UAVs using standard sensors," ICUAS 2015.
"""

import numpy as np

from acc.model.flight_state import FlightState
from acc.model.wind_estimate import WindEstimate

# Default continuous-time process noise covariance Q_c.
# The predict step applies Q_c * dt, so these are per-second change variances.
# Derived from Johansen (2015) Table I, reinterpreted as continuous-time.
#   [v_wind_N, v_wind_E, v_wind_D, gamma]
_DEFAULT_Q = np.diag([1e-3, 1e-3, 1e-6, 1e-8])

# Initial state estimation covariance P(0).
# Reflects uncertainty in wind components (loose) and gamma (tighter, near 1).
_DEFAULT_P0 = np.diag([1e-2, 1e-2, 1e-6, 1e-4])

# Measurement noise variance on the body-x GPS velocity (m/s)^2.
# R = 1 as used in Johansen (2015) experimental tests.
_DEFAULT_R = 1.0


def estimate_wind(
    state: FlightState,
    Q: np.ndarray | None = None,
    R_meas: float = _DEFAULT_R,
    P0: np.ndarray | None = None,
) -> WindEstimate:
    """Run the Johansen (2015) linear time-varying Kalman filter wind observer.

    Reads pitot airspeed from ``state.pitot_airspeed``.  The measurement is
    assumed to be the longitudinal airspeed u_r (body x-axis component).  For
    a standard fixed-mount pitot tube at typical small-UAV angles of attack
    (< 15°), the logged total airspeed V_a is a good approximation:
    u_r ≈ V_a · cos α · cos β ≈ V_a.

    Args:
        state:  FlightState with GNSS velocity, attitude, and pitot_airspeed
                arrays on a common time grid.
        Q:      4×4 continuous-time process noise covariance.
                Defaults to Johansen (2015) values.
        R_meas: Measurement noise variance on the body-x GPS velocity
                component (m²/s²).  Default 1.0.
        P0:     4×4 initial state covariance.  Defaults to loose priors
                on wind components and tight prior on gamma.

    Returns:
        WindEstimate with estimated 3-D wind, pitot calibration factor, and
        wind-relative NED velocity ready for alpha/beta computation.
    """
    assert state.pitot_airspeed is not None, "pitot_airspeed required"
    pitot_airspeed = state.pitot_airspeed

    Q = _DEFAULT_Q if Q is None else Q
    P0 = _DEFAULT_P0 if P0 is None else P0

    num_samples = len(state.time)

    # --- Pre-compute measurement observations ---
    # gps_body_x = d1^T · R_nb · v_gps^NED  (body-x component of GPS velocity)
    # d1^T · R_nb = [ct*cs, ct*ss, -st] (independent of roll φ)
    ct = np.cos(state.theta)
    st = np.sin(state.theta)
    cs = np.cos(state.psi)
    ss = np.sin(state.psi)
    gps_body_x = ct * cs * state.v_north + ct * ss * state.v_east - st * state.v_down

    # --- KF initialisation ---
    # state_estimate = [v_wind_N, v_wind_E, v_wind_D, gamma]
    state_estimate = np.array([0.0, 0.0, 0.0, 1.0])
    covariance = P0.copy()
    identity = np.eye(4)

    state_history = np.empty((num_samples, 4))

    # --- Forward KF pass ---
    for k in range(num_samples):
        dt = state.time[k] - state.time[k - 1] if k > 0 else 0.0

        # Predict (A = 0 → state unchanged, covariance grows)
        covariance = covariance + Q * dt

        # Time-varying measurement matrix  H_k ∈ R^{1×4}
        #   H[0:3] = d1^T · R_nb_k  (wind velocity gains)
        #   H[3]   = pitot_airspeed_k (calibration factor gain)
        H = np.array([ct[k] * cs[k], ct[k] * ss[k], -st[k], pitot_airspeed[k]])

        # Update
        innovation = gps_body_x[k] - H @ state_estimate
        innovation_var = max(H @ covariance @ H + R_meas, 1e-12)
        kalman_gain = (covariance @ H) / innovation_var
        state_estimate = state_estimate + kalman_gain * innovation
        # Joseph form: preserves symmetry and positive-definiteness of P
        IKH = identity - np.outer(kalman_gain, H)
        covariance = IKH @ covariance @ IKH.T + R_meas * np.outer(kalman_gain, kalman_gain)

        state_history[k] = state_estimate

    v_wind_north = state_history[:, 0]
    v_wind_east = state_history[:, 1]
    v_wind_down = state_history[:, 2]
    gamma = state_history[:, 3]

    return WindEstimate(
        v_wind_north=v_wind_north,
        v_wind_east=v_wind_east,
        v_wind_down=v_wind_down,
        gamma=gamma,
        v_rel_north=state.v_north - v_wind_north,
        v_rel_east=state.v_east - v_wind_east,
        v_rel_down=state.v_down - v_wind_down,
    )
