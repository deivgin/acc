import numpy as np

from model.aircraft_model import AircraftModel

_ISA_T0 = 288.15  # K
_ISA_P0 = 101325.0  # Pa
_ISA_RHO0 = 1.225  # kg/m³
_ISA_LAPSE = 0.0065  # K/m
_ISA_G = 9.80665  # m/s²
_ISA_R = 287.05  # J/(kg·K)

# Below this dynamic pressure (Pa), mask coefficients to NaN
_Q_MIN = 10.0


def compute_airspeed(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute true airspeed.

    AV_TAS = sqrt(u² + v² + w²).
    """
    return np.sqrt(u**2 + v**2 + w**2)


def compute_alpha_beta(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, v_tas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angle of attack and sideslip angle.

    alpha = atan2(w, u)
    beta = asin(v / V_TAS)  (clamped to avoid NaN from numerical noise)
    """
    alpha = np.arctan2(w, u)
    ratio = np.clip(v / np.where(v_tas > 0.1, v_tas, 0.1), -1.0, 1.0)
    beta = np.arcsin(ratio)
    return alpha, beta


def compute_thrust(throttle: np.ndarray, max_thrust: float) -> np.ndarray:
    """Compute thrust force from throttle using quadratic model.

    T = max_thrust * throttle².
    """
    return max_thrust * throttle**2


def compute_isa_density(
    altitude: np.ndarray, temperature_offset: float = 0.0
) -> np.ndarray:
    """ISA troposphere model: compute air density from altitude.

    Valid for altitudes below the tropopause (~11 km).
    """
    t = _ISA_T0 + temperature_offset - _ISA_LAPSE * altitude
    t0 = _ISA_T0 + temperature_offset
    p = _ISA_P0 * (t / t0) ** (_ISA_G / (_ISA_LAPSE * _ISA_R))
    rho = p / (_ISA_R * t)
    return rho


def compute_dynamic_pressure(rho: np.ndarray, v_tas: np.ndarray) -> np.ndarray:
    """Compute dynamic pressure.

    q = 0.5 * rho * V².
    """
    return 0.5 * rho * v_tas**2


def compute_angular_acceleration(
    p: np.ndarray, q: np.ndarray, r: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerical differentiation of angular rates: p_dot, q_dot, r_dot."""
    p_dot = np.gradient(p, time)
    q_dot = np.gradient(q, time)
    r_dot = np.gradient(r, time)
    return p_dot, q_dot, r_dot


def compute_body_moments(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p_dot: np.ndarray,
    q_dot: np.ndarray,
    r_dot: np.ndarray,
    aircraft: AircraftModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute body-axis moments using Euler's rotational equations.

    M = I·ω̇ + ω × (I·ω)

    With full I_xz cross-coupling terms:
      l = I_xx·p_dot - I_xz·(r_dot + p·q) + (I_zz - I_yy)·q·r
      m = I_yy·q_dot + (I_xx - I_zz)·p·r + I_xz·(p² - r²)
      n = I_zz·r_dot - I_xz·(p_dot - q·r) + (I_yy - I_xx)·p·q
    """
    ixx = aircraft.i_xx
    iyy = aircraft.i_yy
    izz = aircraft.i_zz
    ixz = aircraft.i_xz

    l_moment = ixx * p_dot - ixz * (r_dot + p * q) + (izz - iyy) * q * r
    m_moment = iyy * q_dot + (ixx - izz) * p * r + ixz * (p**2 - r**2)
    n_moment = izz * r_dot - ixz * (p_dot - q * r) + (iyy - ixx) * p * q

    return l_moment, m_moment, n_moment


def normalize_coefficients(
    lift: np.ndarray,
    drag: np.ndarray,
    side: np.ndarray,
    l_moment: np.ndarray,
    m_moment: np.ndarray,
    n_moment: np.ndarray,
    q_dyn: np.ndarray,
    aircraft: AircraftModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize forces/moments to non-dimensional coefficients.

    CL = L / (q·S)
    CD = D / (q·S)
    CY = Y / (q·S)
    Cl = l / (q·S·b)
    Cm = m / (q·S·c̄)
    Cn = n / (q·S·b)

    Low dynamic pressure regions (q < threshold) are masked to NaN.
    """
    s = aircraft.wing_area
    b = aircraft.wing_span
    c = aircraft.mean_aero_chord

    # Mask low-q regions
    mask = q_dyn < _Q_MIN
    q_safe = np.where(mask, np.nan, q_dyn)

    qs = q_safe * s
    qsb = qs * b
    qsc = qs * c

    cl = lift / qs
    cd = drag / qs
    cy = side / qs
    c_roll = l_moment / qsb
    cm = m_moment / qsc
    cn = n_moment / qsb

    return cl, cd, cy, c_roll, cm, cn
