import numpy as np

from aero.frames import body_to_wind_forces, ned_to_body
from aero.physics import (
    compute_airspeed,
    compute_alpha_beta,
    compute_angular_acceleration,
    compute_body_moments,
    compute_dynamic_pressure,
    compute_isa_density,
    compute_thrust,
    normalize_coefficients,
)
from model.aero_coefficients import AeroCoefficients
from model.aircraft_model import AircraftModel
from model.atmosphere_model import AtmosphereModel
from model.flight_state import FlightState


def compute_coefficients(
    state: FlightState,
    aircraft: AircraftModel,
    atmosphere: AtmosphereModel,
) -> AeroCoefficients:
    # Step 1: NED to body velocity
    u, v, w = ned_to_body(
        state.v_north,
        state.v_east,
        state.v_down,
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
    )
