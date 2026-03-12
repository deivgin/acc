import numpy as np

from acc.aero.frames import body_to_wind_forces, ned_to_body
from acc.aero.physics import (
    compute_airspeed,
    compute_alpha_beta,
    compute_angular_acceleration,
    compute_body_moments,
    compute_dynamic_pressure,
    compute_isa_density,
    compute_thrust,
    normalize_coefficients,
)
from acc.aero.wind_observer import estimate_wind
from acc.model.aero_coefficients import AeroCoefficients
from acc.model.aircraft_model import AircraftModel
from acc.model.atmosphere_model import AtmosphereModel
from acc.model.flight_state import FlightState


def compute_coefficients(
    state: FlightState,
    aircraft: AircraftModel,
    atmosphere: AtmosphereModel,
) -> AeroCoefficients:
    # Wind correction (Johansen 2015).
    # When pitot airspeed is available, run the linear time-varying KF to
    # estimate 3-D wind velocity and subtract it from the GPS ground velocity
    # before rotating to body frame.  Without pitot data, GPS ground velocity
    # is used directly as a fallback (equivalent to assuming zero wind).
    wind = None
    if state.pitot_airspeed is not None:
        wind = estimate_wind(state)
        v_north_air = wind.v_rel_north
        v_east_air = wind.v_rel_east
        v_down_air = wind.v_rel_down
    else:
        v_north_air = state.v_north
        v_east_air = state.v_east
        v_down_air = state.v_down

    # Step 1: NED to body velocity (wind-relative)
    u, v, w = ned_to_body(
        v_north_air,
        v_east_air,
        v_down_air,
        state.phi,
        state.theta,
        state.psi,
    )

    # Step 2: Airspeed, alpha, beta
    v_tas = compute_airspeed(u, v, w)
    alpha, beta = compute_alpha_beta(u, v, w, v_tas)

    # Step 3: Density and dynamic pressure
    if atmosphere.rho is not None:
        rho = np.full_like(state.altitude, atmosphere.rho)
    else:
        rho = compute_isa_density(state.altitude, atmosphere.temperature_offset)
    q_dyn = compute_dynamic_pressure(rho, v_tas)

    # Step 4: Body forces from IMU (F = m * a_specific)
    mass = aircraft.mass
    fx = mass * state.ax_body
    fy = mass * state.ay_body
    fz = mass * state.az_body

    # Step 4.5: Subtract thrust (body x-axis, single motor)
    if aircraft.max_thrust is not None and state.throttle is not None:
        thrust = compute_thrust(state.throttle, aircraft.max_thrust)
        fx = fx - thrust

    # Step 5: Wind-frame forces
    lift, drag, side = body_to_wind_forces(fx, fy, fz, alpha, beta)

    # Step 6: Angular accelerations and body moments
    p_dot, q_dot, r_dot = compute_angular_acceleration(
        state.p,
        state.q,
        state.r,
        state.time,
    )
    l_moment, m_moment, n_moment = compute_body_moments(
        state.p,
        state.q,
        state.r,
        p_dot,
        q_dot,
        r_dot,
        aircraft,
    )

    # Step 7: Normalize to coefficients
    cl, cd, cy, c_roll, cm, cn = normalize_coefficients(
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
        wind_estimate=wind,
    )
