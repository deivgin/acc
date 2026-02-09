"""Tests for Pydantic data models."""

import numpy as np
import pytest
from pydantic import ValidationError

from acc.models import AeroCoefficients, AircraftConfig, AtmosphereConfig, FlightState


class TestAircraftConfig:
    def test_valid_config(self):
        cfg = AircraftConfig(
            mass=1.5,
            wing_area=0.35,
            wing_span=1.4,
            mean_aero_chord=0.25,
            i_xx=0.029,
            i_yy=0.031,
            i_zz=0.055,
            i_xz=0.002,
        )
        assert cfg.mass == 1.5
        assert cfg.i_xz == 0.002

    def test_default_i_xz(self):
        cfg = AircraftConfig(
            mass=1.5,
            wing_area=0.35,
            wing_span=1.4,
            mean_aero_chord=0.25,
            i_xx=0.029,
            i_yy=0.031,
            i_zz=0.055,
        )
        assert cfg.i_xz == 0.0

    def test_rejects_zero_mass(self):
        with pytest.raises(ValidationError):
            AircraftConfig(
                mass=0,
                wing_area=0.35,
                wing_span=1.4,
                mean_aero_chord=0.25,
                i_xx=0.029,
                i_yy=0.031,
                i_zz=0.055,
            )

    def test_rejects_negative_wing_area(self):
        with pytest.raises(ValidationError):
            AircraftConfig(
                mass=1.5,
                wing_area=-0.35,
                wing_span=1.4,
                mean_aero_chord=0.25,
                i_xx=0.029,
                i_yy=0.031,
                i_zz=0.055,
            )

    def test_rejects_negative_inertia(self):
        with pytest.raises(ValidationError):
            AircraftConfig(
                mass=1.5,
                wing_area=0.35,
                wing_span=1.4,
                mean_aero_chord=0.25,
                i_xx=-0.029,
                i_yy=0.031,
                i_zz=0.055,
            )

    def test_rejects_missing_required(self):
        with pytest.raises(ValidationError):
            AircraftConfig(mass=1.5, wing_area=0.35)  # type: ignore[call-arg]


class TestAtmosphereConfig:
    def test_defaults(self):
        cfg = AtmosphereConfig()
        assert cfg.rho is None
        assert cfg.temperature_offset == 0.0

    def test_fixed_rho(self):
        cfg = AtmosphereConfig(rho=1.1)
        assert cfg.rho == 1.1

    def test_temperature_offset(self):
        cfg = AtmosphereConfig(temperature_offset=15.0)
        assert cfg.temperature_offset == 15.0


class TestFlightState:
    def test_creation_with_arrays(self):
        n = 10
        state = FlightState(
            time=np.arange(n, dtype=float),
            ax_body=np.zeros(n),
            ay_body=np.zeros(n),
            az_body=np.zeros(n),
            p=np.zeros(n),
            q=np.zeros(n),
            r=np.zeros(n),
            phi=np.zeros(n),
            theta=np.zeros(n),
            psi=np.zeros(n),
            v_north=np.zeros(n),
            v_east=np.zeros(n),
            v_down=np.zeros(n),
            altitude=np.zeros(n),
        )
        assert len(state.time) == n


class TestAeroCoefficients:
    def test_creation_with_arrays(self):
        n = 5
        coeffs = AeroCoefficients(
            time=np.arange(n, dtype=float),
            cl=np.zeros(n),
            cd=np.zeros(n),
            cy=np.zeros(n),
            c_roll=np.zeros(n),
            cm=np.zeros(n),
            cn=np.zeros(n),
            alpha=np.zeros(n),
            beta=np.zeros(n),
            dynamic_pressure=np.zeros(n),
            airspeed=np.zeros(n),
        )
        assert len(coeffs.cl) == n
