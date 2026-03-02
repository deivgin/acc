"""Core aerodynamic coefficient calculation pipeline.

Computes 6-DOF aerodynamic coefficients (CL, CD, CY, Cl, Cm, Cn) from
ArduPilot dataflash log data using IMU, attitude, and GPS measurements.
"""

from typing import Any

import numpy as np

from acc.models import AeroCoefficients, AircraftConfig, AtmosphereConfig, FlightState

# ISA sea-level constants
_ISA_T0 = 288.15  # K
_ISA_P0 = 101325.0  # Pa
_ISA_RHO0 = 1.225  # kg/m³
_ISA_LAPSE = 0.0065  # K/m
_ISA_G = 9.80665  # m/s²
_ISA_R = 287.05  # J/(kg·K)

# Below this dynamic pressure (Pa), mask coefficients to NaN
_Q_MIN = 10.0


def _extract_time_and_field(
    rows: list[dict[str, Any]], field: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time (seconds) and a named field from message rows."""
    time = np.array([float(r["TimeMS"]) / 1e3 for r in rows])
    values = np.array([float(r[field]) for r in rows])
    return time, values


def _interpolate_to_common_time(
    log_data: dict[str, list[dict[str, Any]]],
) -> dict[str, np.ndarray]:
    """Align ATT, IMU, and GPS data onto a common time grid using np.interp.

    Uses the ATT message timestamps as the reference grid since ATT provides
    the attitude data needed throughout the pipeline.
    """
    att_rows = log_data["ATT"]
    imu_rows = log_data["IMU"]
    gps_rows = log_data["GPS"]

    # Reference time grid from ATT
    t_att = np.array([float(r["TimeMS"]) / 1e3 for r in att_rows])

    # ATT fields (already on reference grid)
    roll = np.array([float(r["Roll"]) for r in att_rows])
    pitch = np.array([float(r["Pitch"]) for r in att_rows])
    yaw = np.array([float(r["Yaw"]) for r in att_rows])

    # IMU fields — interpolate to ATT time grid
    t_imu, acc_x = _extract_time_and_field(imu_rows, "AccX")
    _, acc_y = _extract_time_and_field(imu_rows, "AccY")
    _, acc_z = _extract_time_and_field(imu_rows, "AccZ")
    _, gyr_x = _extract_time_and_field(imu_rows, "GyrX")
    _, gyr_y = _extract_time_and_field(imu_rows, "GyrY")
    _, gyr_z = _extract_time_and_field(imu_rows, "GyrZ")

    acc_x_i = np.interp(t_att, t_imu, acc_x)
    acc_y_i = np.interp(t_att, t_imu, acc_y)
    acc_z_i = np.interp(t_att, t_imu, acc_z)
    gyr_x_i = np.interp(t_att, t_imu, gyr_x)
    gyr_y_i = np.interp(t_att, t_imu, gyr_y)
    gyr_z_i = np.interp(t_att, t_imu, gyr_z)

    # GPS fields — interpolate to ATT time grid
    t_gps = np.array([float(r["TimeMS"]) / 1e3 for r in gps_rows])
    spd = np.array([float(r["Spd"]) for r in gps_rows])
    gcrs = np.array([float(r["GCrs"]) for r in gps_rows])
    vz = np.array([float(r["VZ"]) for r in gps_rows])
    alt = np.array([float(r["Alt"]) for r in gps_rows])

    # GPS velocity: NED components
    gcrs_rad = np.radians(gcrs)
    v_north_gps = spd * np.cos(gcrs_rad)
    v_east_gps = spd * np.sin(gcrs_rad)
    v_down_gps = -vz  # VZ is positive up in ArduPilot GPS

    v_north_i = np.interp(t_att, t_gps, v_north_gps)
    v_east_i = np.interp(t_att, t_gps, v_east_gps)
    v_down_i = np.interp(t_att, t_gps, v_down_gps)
    alt_i = np.interp(t_att, t_gps, alt)

    return {
        "time": t_att,
        "ax_body": acc_x_i,
        "ay_body": acc_y_i,
        "az_body": acc_z_i,
        "p": gyr_x_i,
        "q": gyr_y_i,
        "r": gyr_z_i,
        "phi": np.radians(roll),
        "theta": np.radians(pitch),
        "psi": np.radians(yaw),
        "v_north": v_north_i,
        "v_east": v_east_i,
        "v_down": v_down_i,
        "altitude": alt_i,
    }


def _ned_to_body(
    v_north: np.ndarray,
    v_east: np.ndarray,
    v_down: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate NED velocity to body frame using 3-2-1 Euler rotation.

    Returns (u, v, w) body-frame velocity components.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    cs = np.cos(psi)
    ss = np.sin(psi)

    # DCM rows (NED -> body)
    u = ct * cs * v_north + ct * ss * v_east - st * v_down
    v = (
        (sp * st * cs - cp * ss) * v_north
        + (sp * st * ss + cp * cs) * v_east
        + sp * ct * v_down
    )
    w = (
        (cp * st * cs + sp * ss) * v_north
        + (cp * st * ss - sp * cs) * v_east
        + cp * ct * v_down
    )
    return u, v, w


def _compute_airspeed(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute true airspeed: V_TAS = sqrt(u² + v² + w²)."""
    return np.sqrt(u**2 + v**2 + w**2)


def _compute_alpha_beta(
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


def _isa_density(altitude: np.ndarray, temperature_offset: float = 0.0) -> np.ndarray:
    """ISA troposphere model: compute air density from altitude.

    Valid for altitudes below the tropopause (~11 km).
    """
    t = _ISA_T0 + temperature_offset - _ISA_LAPSE * altitude
    t0 = _ISA_T0 + temperature_offset
    p = _ISA_P0 * (t / t0) ** (_ISA_G / (_ISA_LAPSE * _ISA_R))
    rho = p / (_ISA_R * t)
    return rho


def _compute_dynamic_pressure(rho: np.ndarray, v_tas: np.ndarray) -> np.ndarray:
    """Compute dynamic pressure: q = 0.5 * rho * V²."""
    return 0.5 * rho * v_tas**2


def _body_to_wind_forces(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform body-frame forces to wind-frame (lift, drag, sideforce).

    Body to wind rotation via alpha (pitch) and beta (yaw):
      D = -fx·cos(α)·cos(β) - fy·sin(β) - fz·sin(α)·cos(β)
      Y = -fx·cos(α)·sin(β) + fy·cos(β) - fz·sin(α)·sin(β)
      L = fx·sin(α) - fz·cos(α)
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    drag = -fx * ca * cb - fy * sb - fz * sa * cb
    side = -fx * ca * sb + fy * cb - fz * sa * sb
    lift = fx * sa - fz * ca

    return lift, drag, side


def _compute_angular_acceleration(
    p: np.ndarray, q: np.ndarray, r: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerical differentiation of angular rates: p_dot, q_dot, r_dot."""
    p_dot = np.gradient(p, time)
    q_dot = np.gradient(q, time)
    r_dot = np.gradient(r, time)
    return p_dot, q_dot, r_dot


def _compute_body_moments(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p_dot: np.ndarray,
    q_dot: np.ndarray,
    r_dot: np.ndarray,
    aircraft: AircraftConfig,
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


def _normalize_coefficients(
    lift: np.ndarray,
    drag: np.ndarray,
    side: np.ndarray,
    l_moment: np.ndarray,
    m_moment: np.ndarray,
    n_moment: np.ndarray,
    q_dyn: np.ndarray,
    aircraft: AircraftConfig,
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


def extract_flight_state(
    log_data: dict[str, list[dict[str, Any]]],
) -> FlightState:
    """Convert parse_log() output to a unified FlightState.

    Requires ATT, IMU, and GPS message types in the log data.
    """
    for msg_type in ("ATT", "IMU", "GPS"):
        if msg_type not in log_data or not log_data[msg_type]:
            raise ValueError(f"Log data missing required '{msg_type}' messages.")

    interpolated = _interpolate_to_common_time(log_data)
    return FlightState(**interpolated)


def compute_coefficients(
    state: FlightState,
    aircraft: AircraftConfig,
    atmosphere: AtmosphereConfig,
) -> AeroCoefficients:
    """Run the full physics pipeline to compute aerodynamic coefficients.

    Args:
        state: Unified flight state time-series data.
        aircraft: Aircraft configuration parameters.
        atmosphere: Optional atmosphere config. Defaults to ISA standard.

    Returns:
        AeroCoefficients with 6-DOF coefficients and supplementary data.
    """
    if atmosphere is None:
        atmosphere = AtmosphereConfig()

    # Step 1: NED to body velocity
    u, v, w = _ned_to_body(
        state.v_north,
        state.v_east,
        state.v_down,
        state.phi,
        state.theta,
        state.psi,
    )

    # Step 2: Airspeed, alpha, beta
    v_tas = _compute_airspeed(u, v, w)
    alpha, beta = _compute_alpha_beta(u, v, w, v_tas)

    # Step 3: Density and dynamic pressure
    if atmosphere.rho is not None:
        rho = np.full_like(state.altitude, atmosphere.rho)
    else:
        rho = _isa_density(state.altitude, atmosphere.temperature_offset)
    q_dyn = _compute_dynamic_pressure(rho, v_tas)

    # Step 4: Body forces from IMU (F = m * a_specific)
    mass = aircraft.mass
    fx = mass * state.ax_body
    fy = mass * state.ay_body
    fz = mass * state.az_body

    # Step 5: Wind-frame forces
    lift, drag, side = _body_to_wind_forces(fx, fy, fz, alpha, beta)

    # Step 6: Angular accelerations and body moments
    p_dot, q_dot, r_dot = _compute_angular_acceleration(
        state.p,
        state.q,
        state.r,
        state.time,
    )
    l_moment, m_moment, n_moment = _compute_body_moments(
        state.p,
        state.q,
        state.r,
        p_dot,
        q_dot,
        r_dot,
        aircraft,
    )

    # Step 7: Normalize to coefficients
    cl, cd, cy, c_roll, cm, cn = _normalize_coefficients(
        lift,
        drag,
        side,
        l_moment,
        m_moment,
        n_moment,
        q_dyn,
        aircraft,
    )

    return AeroCoefficients(
        time=state.time,
        cl=cl,
        cd=cd,
        cy=cy,
        c_roll=c_roll,
        cm=cm,
        cn=cn,
        alpha=alpha,
        beta=beta,
        dynamic_pressure=q_dyn,
        airspeed=v_tas,
    )


def compute_from_log(
    log_data: dict[str, list[dict[str, Any]]],
    aircraft: AircraftConfig,
    atmosphere: AtmosphereConfig,
) -> AeroCoefficients:
    """Convenience end-to-end wrapper: log data -> aerodynamic coefficients."""
    state = extract_flight_state(log_data)
    return compute_coefficients(state, aircraft, atmosphere)
