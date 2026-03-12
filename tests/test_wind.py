"""Tests for wind velocity observer and WindEstimate model."""

import numpy as np
import pytest

from acc.aero.wind_observer import estimate_wind
from acc.model.flight_state import FlightState
from acc.model.wind_estimate import WindEstimate


def _make_state(
    n: int = 200,
    airspeed: float = 15.0,
    theta_rad: float = 0.087,
    psi_rad: float = 0.0,
    wind_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FlightState:
    """Build a FlightState with pitot data and optional constant wind.

    The aircraft flies at constant attitude.  GPS ground velocity equals
    airspeed + wind, and the pitot measurement is the body-x component
    of the wind-relative (airspeed) NED vector.
    """
    time = np.linspace(0, 10, n)

    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    cs = np.cos(psi_rad)
    ss = np.sin(psi_rad)

    # Airspeed NED: aircraft flies along heading with given pitch
    airspeed_north = airspeed * ct * cs
    airspeed_east = airspeed * ct * ss
    airspeed_down = -airspeed * st

    # GPS = airspeed + wind
    wn, we, wd = wind_ned
    v_north = np.full(n, airspeed_north + wn)
    v_east = np.full(n, airspeed_east + we)
    v_down = np.full(n, airspeed_down + wd)

    # Pitot measures body-x component of airspeed:
    # u_r = d1^T · R_nb · airspeed_NED
    pitot = np.full(n, ct * cs * airspeed_north + ct * ss * airspeed_east - st * airspeed_down)

    return FlightState(
        time=time,
        ax_body=np.zeros(n),
        ay_body=np.zeros(n),
        az_body=np.full(n, -9.80665),
        p=np.zeros(n),
        q=np.zeros(n),
        r=np.zeros(n),
        phi=np.zeros(n),
        theta=np.full(n, theta_rad),
        psi=np.full(n, psi_rad),
        v_north=v_north,
        v_east=v_east,
        v_down=v_down,
        altitude=np.full(n, 100.0),
        pitot_airspeed=pitot,
    )


class TestWindEstimateModel:
    def test_creation(self):
        n = 5
        est = WindEstimate(
            v_wind_north=np.zeros(n),
            v_wind_east=np.zeros(n),
            v_wind_down=np.zeros(n),
            gamma=np.ones(n),
            v_rel_north=np.full(n, 15.0),
            v_rel_east=np.zeros(n),
            v_rel_down=np.zeros(n),
        )
        assert len(est.gamma) == n
        assert est.v_rel_north[0] == 15.0


class TestEstimateWind:
    def test_zero_wind_gamma_converges(self):
        """With zero wind, gamma should stay near 1 and wind near zero."""
        state = _make_state(wind_ned=(0.0, 0.0, 0.0))
        result = estimate_wind(state)

        # Check at the end of the time series (after convergence)
        assert abs(result.gamma[-1] - 1.0) < 0.05
        assert abs(result.v_wind_north[-1]) < 0.5
        assert abs(result.v_wind_down[-1]) < 0.1

    def test_wind_relative_velocity_matches_airspeed(self):
        """Wind-relative velocity should recover the original airspeed."""
        airspeed = 18.0
        theta = 0.05
        state = _make_state(airspeed=airspeed, theta_rad=theta, wind_ned=(0.0, 0.0, 0.0))
        result = estimate_wind(state)

        v_rel_mag = np.sqrt(
            result.v_rel_north**2 + result.v_rel_east**2 + result.v_rel_down**2
        )
        # After KF settles, wind-relative speed should be close to airspeed
        np.testing.assert_allclose(v_rel_mag[-1], airspeed, atol=0.5)

    def test_with_yaw_variation_recovers_wind(self):
        """With yaw variation, the KF can observe horizontal wind components.

        Uses a faster process noise Q to speed convergence in a short test,
        and a ±45° yaw sweep over 60 s to give sufficient excitation.
        """
        n = 3000
        time = np.linspace(0, 60, n)
        airspeed = 15.0
        wind_north = 3.0
        wind_east = 2.0

        # Sinusoidal yaw sweep: psi varies ±45° with 8 s period
        psi = np.radians(45.0) * np.sin(2 * np.pi * time / 8.0)
        theta = np.full(n, 0.05)

        ct = np.cos(theta)
        st = np.sin(theta)
        cs = np.cos(psi)
        ss = np.sin(psi)

        # Airspeed NED (follows heading)
        as_north = airspeed * ct * cs
        as_east = airspeed * ct * ss
        as_down = -airspeed * st

        # GPS = airspeed + wind
        v_north = as_north + wind_north
        v_east = as_east + wind_east
        v_down = as_down

        # Pitot body-x
        pitot = ct * cs * as_north + ct * ss * as_east - st * as_down

        state = FlightState(
            time=time,
            ax_body=np.zeros(n),
            ay_body=np.zeros(n),
            az_body=np.full(n, -9.80665),
            p=np.zeros(n),
            q=np.zeros(n),
            r=np.zeros(n),
            phi=np.zeros(n),
            theta=theta,
            psi=psi,
            v_north=v_north,
            v_east=v_east,
            v_down=v_down,
            altitude=np.full(n, 100.0),
            pitot_airspeed=pitot,
        )

        # Use faster process noise for quicker convergence in the test
        Q_fast = np.diag([1e-1, 1e-1, 1e-6, 1e-6])
        result = estimate_wind(state, Q=Q_fast)

        # After convergence, wind estimate should approach true values
        np.testing.assert_allclose(result.v_wind_north[-1], wind_north, atol=1.0)
        np.testing.assert_allclose(result.v_wind_east[-1], wind_east, atol=1.0)
        assert abs(result.gamma[-1] - 1.0) < 0.1

    def test_single_sample(self):
        """Single-sample input should not crash."""
        state = _make_state(n=1)
        result = estimate_wind(state)
        assert len(result.gamma) == 1

    def test_requires_pitot(self):
        """Should raise if pitot_airspeed is None."""
        state = FlightState(
            time=np.arange(10, dtype=float),
            ax_body=np.zeros(10),
            ay_body=np.zeros(10),
            az_body=np.zeros(10),
            p=np.zeros(10),
            q=np.zeros(10),
            r=np.zeros(10),
            phi=np.zeros(10),
            theta=np.zeros(10),
            psi=np.zeros(10),
            v_north=np.zeros(10),
            v_east=np.zeros(10),
            v_down=np.zeros(10),
            altitude=np.zeros(10),
        )
        with pytest.raises(AssertionError, match="pitot_airspeed required"):
            estimate_wind(state)
