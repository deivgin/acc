"""Tests for aerodynamic coefficient calculation pipeline."""

import numpy as np
import pytest

from acc.aero.frames import body_to_wind_forces, ned_to_body
from acc.aero.physics import (
    compute_airspeed,
    compute_alpha_beta,
    compute_angular_acceleration,
    compute_dynamic_pressure,
    compute_thrust,
    compute_isa_density,
    normalize_coefficients,
)
from acc.aero.compute_coefficients import compute_coefficients
from acc.log_parser import extract_flight_state
from acc.model.aircraft_model import AircraftModel
from acc.model.atmosphere_model import AtmosphereModel
from acc.model.flight_state import FlightState


@pytest.fixture
def aircraft():
    return AircraftModel(
        mass=1.5,
        wing_area=0.35,
        wing_span=1.4,
        mean_aero_chord=0.25,
        i_xx=0.029,
        i_yy=0.031,
        i_zz=0.055,
        i_xz=0.002,
    )


class TestISADensity:
    def test_sea_level(self):
        rho = compute_isa_density(np.array([0.0]))
        np.testing.assert_allclose(rho, [1.225], atol=0.001)

    def test_decreases_with_altitude(self):
        alt = np.array([0.0, 1000.0, 5000.0])
        rho = compute_isa_density(alt)
        assert rho[0] > rho[1] > rho[2]

    def test_temperature_offset(self):
        rho_standard = compute_isa_density(np.array([1000.0]))
        rho_hot = compute_isa_density(np.array([1000.0]), temperature_offset=20.0)
        # Hotter atmosphere -> lower density
        assert rho_hot[0] < rho_standard[0]


class TestNEDToBody:
    def test_identity_at_zero_angles(self):
        """With all Euler angles zero, NED = body frame."""
        vn = np.array([10.0])
        ve = np.array([0.0])
        vd = np.array([0.0])
        phi = theta = psi = np.array([0.0])
        u, v, w = ned_to_body(vn, ve, vd, phi, theta, psi)
        np.testing.assert_allclose(u, [10.0], atol=1e-10)
        np.testing.assert_allclose(v, [0.0], atol=1e-10)
        np.testing.assert_allclose(w, [0.0], atol=1e-10)

    def test_90_yaw(self):
        """90° yaw (heading east): north velocity maps to body -v (left)."""
        vn = np.array([10.0])
        ve = np.array([0.0])
        vd = np.array([0.0])
        phi = theta = np.array([0.0])
        psi = np.array([np.pi / 2])
        u, v, w = ned_to_body(vn, ve, vd, phi, theta, psi)
        np.testing.assert_allclose(u, [0.0], atol=1e-10)
        np.testing.assert_allclose(v, [-10.0], atol=1e-10)
        np.testing.assert_allclose(w, [0.0], atol=1e-10)

    def test_preserves_speed_magnitude(self):
        """Rotation should not change the magnitude of the velocity vector."""
        vn = np.array([3.0])
        ve = np.array([4.0])
        vd = np.array([5.0])
        phi = np.array([0.3])
        theta = np.array([0.2])
        psi = np.array([1.0])
        u, v, w = ned_to_body(vn, ve, vd, phi, theta, psi)
        speed_ned = np.sqrt(vn**2 + ve**2 + vd**2)
        speed_body = np.sqrt(u**2 + v**2 + w**2)
        np.testing.assert_allclose(speed_body, speed_ned, atol=1e-10)


class TestAirspeedAlphaBeta:
    def test_straight_flight(self):
        """Pure forward flight: alpha=0, beta=0."""
        u = np.array([20.0])
        v = np.array([0.0])
        w = np.array([0.0])
        v_tas = compute_airspeed(u, v, w)
        np.testing.assert_allclose(v_tas, [20.0])
        alpha, beta = compute_alpha_beta(u, v, w, v_tas)
        np.testing.assert_allclose(alpha, [0.0], atol=1e-10)
        np.testing.assert_allclose(beta, [0.0], atol=1e-10)

    def test_positive_alpha(self):
        """Positive w (downward in body) gives positive alpha."""
        u = np.array([20.0])
        v = np.array([0.0])
        w = np.array([2.0])
        v_tas = compute_airspeed(u, v, w)
        alpha, beta = compute_alpha_beta(u, v, w, v_tas)
        assert alpha[0] > 0
        np.testing.assert_allclose(beta, [0.0], atol=1e-10)

    def test_positive_beta(self):
        """Positive v (rightward in body) gives positive beta."""
        u = np.array([20.0])
        v = np.array([2.0])
        w = np.array([0.0])
        v_tas = compute_airspeed(u, v, w)
        alpha, beta = compute_alpha_beta(u, v, w, v_tas)
        assert beta[0] > 0


class TestDynamicPressure:
    def test_known_value(self):
        rho = np.array([1.225])
        v = np.array([20.0])
        q = compute_dynamic_pressure(rho, v)
        np.testing.assert_allclose(q, [0.5 * 1.225 * 400.0])

    def test_zero_speed(self):
        rho = np.array([1.225])
        v = np.array([0.0])
        q = compute_dynamic_pressure(rho, v)
        np.testing.assert_allclose(q, [0.0])


class TestBodyToWindForces:
    def test_no_alpha_no_beta(self):
        """At zero alpha/beta, body and wind frames align:
        Drag = -Fx, Lift = -Fz, Side = Fy."""
        fx = np.array([-10.0])  # thrust forward
        fy = np.array([0.0])
        fz = np.array([-5.0])  # upward in body (negative z)
        alpha = np.array([0.0])
        beta = np.array([0.0])
        lift, drag, side = body_to_wind_forces(fx, fy, fz, alpha, beta)
        np.testing.assert_allclose(lift, [5.0], atol=1e-10)
        np.testing.assert_allclose(drag, [10.0], atol=1e-10)
        np.testing.assert_allclose(side, [0.0], atol=1e-10)


class TestAngularAcceleration:
    def test_constant_rate(self):
        """Constant angular rate -> zero acceleration."""
        t = np.linspace(0, 1, 100)
        p = np.full_like(t, 0.5)
        q = np.full_like(t, 0.0)
        r = np.full_like(t, 0.0)
        p_dot, q_dot, r_dot = compute_angular_acceleration(p, q, r, t)
        np.testing.assert_allclose(p_dot, 0.0, atol=1e-10)
        np.testing.assert_allclose(q_dot, 0.0, atol=1e-10)
        np.testing.assert_allclose(r_dot, 0.0, atol=1e-10)

    def test_linear_rate(self):
        """Linearly increasing rate -> constant acceleration."""
        t = np.linspace(0, 1, 100)
        p = 2.0 * t  # p_dot should be ~2.0
        q = np.zeros_like(t)
        r = np.zeros_like(t)
        p_dot, _, _ = compute_angular_acceleration(p, q, r, t)
        # Interior points should be very close to 2.0
        np.testing.assert_allclose(p_dot[5:-5], 2.0, atol=0.01)


class TestNormalization:
    def test_known_coefficients(self, aircraft):
        q_dyn = np.array([100.0])
        s = aircraft.wing_area
        b = aircraft.wing_span
        c = aircraft.mean_aero_chord

        lift = np.array([q_dyn[0] * s * 0.5])  # CL should be 0.5
        drag = np.array([q_dyn[0] * s * 0.03])  # CD should be 0.03
        side = np.array([0.0])
        l_m = np.array([q_dyn[0] * s * b * 0.01])
        m_m = np.array([q_dyn[0] * s * c * 0.02])
        n_m = np.array([0.0])

        cl, cd, cy, c_roll, cm, cn = normalize_coefficients(
            lift, drag, side, l_m, m_m, n_m, q_dyn, aircraft,
        )
        np.testing.assert_allclose(cl, [0.5], atol=1e-10)
        np.testing.assert_allclose(cd, [0.03], atol=1e-10)
        np.testing.assert_allclose(cy, [0.0], atol=1e-10)
        np.testing.assert_allclose(c_roll, [0.01], atol=1e-10)
        np.testing.assert_allclose(cm, [0.02], atol=1e-10)
        np.testing.assert_allclose(cn, [0.0], atol=1e-10)

    def test_low_q_masked(self, aircraft):
        """Dynamic pressure below threshold should produce NaN."""
        q_dyn = np.array([5.0])  # below 10 Pa threshold
        cl, cd, cy, c_roll, cm, cn = normalize_coefficients(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            q_dyn,
            aircraft,
        )
        assert np.isnan(cl[0])
        assert np.isnan(cd[0])


class TestExtractFlightState:
    def test_missing_message_type(self):
        with pytest.raises(ValueError, match="ATT"):
            extract_flight_state({"IMU": [{}], "GPS": [{}]})

    def test_empty_message_type(self):
        with pytest.raises(ValueError, match="ATT"):
            extract_flight_state({"ATT": [], "IMU": [{}], "GPS": [{}]})


class TestIntegration:
    """Integration test with synthetic straight-and-level flight data."""

    def test_straight_level_flight(self, aircraft):
        """In straight-and-level flight, CL ≈ weight / (q·S)."""
        n = 200
        time = np.linspace(0, 10, n)

        # Straight-and-level: 20 m/s north, 100m altitude
        v_north = np.full(n, 20.0)
        v_east = np.zeros(n)
        v_down = np.zeros(n)
        altitude = np.full(n, 100.0)

        # Zero Euler angles (flying north, wings level)
        phi = np.zeros(n)
        theta = np.zeros(n)
        psi = np.zeros(n)

        # In straight-and-level, body axes = NED, u=20, v=0, w=0
        # IMU specific force: AccZ ≈ -g (supporting weight)
        # For level flight, total specific force in z_body ≈ -g
        g = 9.80665
        ax_body = np.zeros(n)  # no net forward acceleration
        ay_body = np.zeros(n)
        az_body = np.full(n, -g)  # specific force supporting weight

        # No rotation
        p = np.zeros(n)
        q = np.zeros(n)
        r = np.zeros(n)

        state = FlightState(
            time=time,
            ax_body=ax_body,
            ay_body=ay_body,
            az_body=az_body,
            p=p,
            q=q,
            r=r,
            phi=phi,
            theta=theta,
            psi=psi,
            v_north=v_north,
            v_east=v_east,
            v_down=v_down,
            altitude=altitude,
        )

        atm = AtmosphereModel(rho=1.225)
        result = compute_coefficients(state, aircraft, atm)

        # Expected CL = m*g / (q*S) = 1.5*9.80665 / (0.5*1.225*400*0.35)
        q_dyn = 0.5 * 1.225 * 20.0**2
        expected_cl = aircraft.mass * g / (q_dyn * aircraft.wing_area)

        # CL should be approximately the lift coefficient for level flight
        valid = np.isfinite(result.cl)
        mean_cl = np.nanmean(result.cl[valid])
        np.testing.assert_allclose(mean_cl, expected_cl, atol=0.01)

        # Alpha and beta should be ~0 for straight flight
        np.testing.assert_allclose(result.alpha, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.beta, 0.0, atol=1e-10)

        # Moment coefficients should be ~0 (no rotation)
        np.testing.assert_allclose(result.c_roll[valid], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.cn[valid], 0.0, atol=1e-10)


class TestComputeThrust:
    def test_known_values(self):
        throttle = np.array([0.0, 0.5, 1.0])
        max_thrust = 10.0
        result = compute_thrust(throttle, max_thrust)
        np.testing.assert_allclose(result, [0.0, 2.5, 10.0])

    def test_quadratic_relationship(self):
        throttle = np.array([0.25, 0.5, 0.75])
        max_thrust = 20.0
        expected = max_thrust * throttle**2
        result = compute_thrust(throttle, max_thrust)
        np.testing.assert_allclose(result, expected)


class TestThrustIntegration:
    """Integration tests for thrust subtraction in the coefficient pipeline."""

    def _make_level_state(self, n=200, throttle=None):
        """Create a straight-and-level flight state."""
        time = np.linspace(0, 10, n)
        g = 9.80665
        return FlightState(
            time=time,
            ax_body=np.zeros(n),
            ay_body=np.zeros(n),
            az_body=np.full(n, -g),
            p=np.zeros(n),
            q=np.zeros(n),
            r=np.zeros(n),
            phi=np.zeros(n),
            theta=np.zeros(n),
            psi=np.zeros(n),
            v_north=np.full(n, 20.0),
            v_east=np.zeros(n),
            v_down=np.zeros(n),
            altitude=np.full(n, 100.0),
            throttle=throttle,
        )

    def test_thrust_changes_cd(self, aircraft):
        """With thrust, CD should differ from the no-thrust case."""
        n = 200
        atm = AtmosphereModel(rho=1.225)

        # Without thrust
        state_no_thrust = self._make_level_state(n)
        result_no_thrust = compute_coefficients(state_no_thrust, aircraft, atm)

        # With thrust
        throttle = np.full(n, 0.5)
        state_with_thrust = self._make_level_state(n, throttle=throttle)
        aircraft_thrust = AircraftModel(
            mass=aircraft.mass,
            wing_area=aircraft.wing_area,
            wing_span=aircraft.wing_span,
            mean_aero_chord=aircraft.mean_aero_chord,
            i_xx=aircraft.i_xx,
            i_yy=aircraft.i_yy,
            i_zz=aircraft.i_zz,
            i_xz=aircraft.i_xz,
            max_thrust=15.0,
        )
        result_with_thrust = compute_coefficients(
            state_with_thrust, aircraft_thrust, atm
        )

        valid = np.isfinite(result_no_thrust.cd)
        mean_cd_no = np.nanmean(result_no_thrust.cd[valid])
        mean_cd_yes = np.nanmean(result_with_thrust.cd[valid])
        assert mean_cd_no != pytest.approx(mean_cd_yes, abs=1e-6)

    def test_thrust_does_not_change_cl(self, aircraft):
        """CL should remain the same since thrust acts along body x-axis."""
        n = 200
        atm = AtmosphereModel(rho=1.225)

        state_no_thrust = self._make_level_state(n)
        result_no_thrust = compute_coefficients(state_no_thrust, aircraft, atm)

        throttle = np.full(n, 0.5)
        state_with_thrust = self._make_level_state(n, throttle=throttle)
        aircraft_thrust = AircraftModel(
            mass=aircraft.mass,
            wing_area=aircraft.wing_area,
            wing_span=aircraft.wing_span,
            mean_aero_chord=aircraft.mean_aero_chord,
            i_xx=aircraft.i_xx,
            i_yy=aircraft.i_yy,
            i_zz=aircraft.i_zz,
            i_xz=aircraft.i_xz,
            max_thrust=15.0,
        )
        result_with_thrust = compute_coefficients(
            state_with_thrust, aircraft_thrust, atm
        )

        valid = np.isfinite(result_no_thrust.cl)
        mean_cl_no = np.nanmean(result_no_thrust.cl[valid])
        mean_cl_yes = np.nanmean(result_with_thrust.cl[valid])
        np.testing.assert_allclose(mean_cl_no, mean_cl_yes, atol=1e-10)

    def test_no_thrust_backward_compat(self, aircraft):
        """Without max_thrust or throttle, results are unchanged."""
        n = 200
        atm = AtmosphereModel(rho=1.225)

        state = self._make_level_state(n)
        result = compute_coefficients(state, aircraft, atm)

        g = 9.80665
        q_dyn = 0.5 * 1.225 * 20.0**2
        expected_cl = aircraft.mass * g / (q_dyn * aircraft.wing_area)
        valid = np.isfinite(result.cl)
        mean_cl = np.nanmean(result.cl[valid])
        np.testing.assert_allclose(mean_cl, expected_cl, atol=0.01)
