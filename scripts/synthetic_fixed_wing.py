"""Generate synthetic fixed-wing flight data and validate the aero pipeline.

Creates a 30-second multi-phase flight profile for a Bixler-class fixed-wing UAV
and runs it through the aerodynamic coefficient pipeline.

Aircraft: small fixed-wing UAV (Bixler-class), 1.8 kg, 0.32 m² wing area.

Flight profile (3 phases, 30 s total):
  1. Cruise    (0-10 s):  15 m/s north, alpha=5°, 50% throttle, level at 100 m
  2. Fast cruise (12-22 s): 20 m/s north, alpha=3°, 40% throttle, level at 100 m
  3. Climb    (24-30 s): 15 m/s north, alpha=8°, 75% throttle, climb 2 m/s

Smooth cosine transitions between phases (10-12 s and 22-24 s).

Usage:
    python scripts/synthetic_fixed_wing.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acc.model.aero_coefficients import AeroCoefficients

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acc.log_parser import compute_from_log
from acc.model.aircraft_model import AircraftModel
from acc.model.atmosphere_model import AtmosphereModel

G = 9.80665  # m/s²


def _cosine_blend(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Cosine interpolation factor: 0 before t0, smoothly rises to 1 at t1."""
    frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * frac))


def _piecewise_profile(t: np.ndarray, v1: float, v2: float, v3: float) -> np.ndarray:
    """Three-phase piecewise profile with cosine transitions.

    Phase 1: 0-10 s, Transition: 10-12 s, Phase 2: 12-22 s,
    Transition: 22-24 s, Phase 3: 24-30 s.
    """
    b1 = _cosine_blend(t, 10.0, 12.0)
    b2 = _cosine_blend(t, 22.0, 24.0)
    return v1 + b1 * (v2 - v1) + b2 * (v3 - v2)


def generate_flight_data(
    dt: float = 0.02,
    noise_scale: float = 0.0,
    wind_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, list[dict[str, float]]]:
    """Generate synthetic flight log data matching parse_log() output format.

    Args:
        dt: Sample period in seconds (default 50 Hz).
        noise_scale: Multiplier for sensor noise (0 = no noise).
        wind_ned: Constant wind vector (north, east, down) in m/s added to GPS
            ground velocity to simulate ambient wind.  The ARSP pitot reading
            still reflects only the airspeed (wind-relative) component.

    Returns:
        Dict with ATT, IMU, GPS, CTUN, ARSP message lists matching
        parse_log() format.
    """
    times = np.arange(0, 30.0 + dt, dt)
    n = len(times)

    # --- Build flight parameter profiles (airspeed / aircraft frame) ---
    airspeed_mag = _piecewise_profile(times, 15.0, 20.0, 15.0)
    alpha_deg = _piecewise_profile(times, 5.0, 3.0, 8.0)
    throttle_pct = _piecewise_profile(times, 50.0, 40.0, 75.0)
    climb_rate_air = _piecewise_profile(times, 0.0, 0.0, 2.0)

    alpha_rad = np.radians(alpha_deg)

    # Airspeed NED components (flying north through the air mass, wings level)
    airspeed_north = airspeed_mag
    airspeed_down = -climb_rate_air  # NED: down is positive

    # Pitch angle from desired alpha and airspeed components
    tan_a = np.tan(alpha_rad)
    theta = np.arctan2(
        tan_a * airspeed_north - airspeed_down,
        airspeed_north + tan_a * airspeed_down,
    )

    # Altitude: integrate climb rate
    altitude = 100.0 + np.cumsum(climb_rate_air * dt)
    altitude = np.concatenate(([100.0], altitude[:-1]))

    # --- IMU specific force (depends on airspeed derivatives) ---
    # For a constant wind, d(airspeed)/dt = d(ground_vel)/dt, so the
    # accelerometer readings are identical to the zero-wind case.
    d_airspeed_north = np.gradient(airspeed_north, times)
    d_airspeed_down = np.gradient(airspeed_down, times)
    ct = np.cos(theta)
    st = np.sin(theta)

    acc_x = ct * d_airspeed_north - st * (d_airspeed_down - G)
    acc_y = np.zeros(n)
    acc_z = st * d_airspeed_north + ct * (d_airspeed_down - G)

    # Gyroscope (wings level, heading constant)
    gyr_x = np.zeros(n)
    gyr_y = np.gradient(theta, times)
    gyr_z = np.zeros(n)

    # Pitot body-x airspeed: d1^T · R_nb · airspeed^NED  (phi=psi=0)
    pitot_airspeed = ct * airspeed_north - st * airspeed_down

    # GPS ground velocity = airspeed + wind
    wind_north, wind_east, wind_down = wind_ned
    gps_north = airspeed_north + wind_north
    gps_east = np.full(n, wind_east)
    gps_down = airspeed_down + wind_down

    ground_speed_2d = np.sqrt(gps_north**2 + gps_east**2)
    ground_course = np.degrees(np.arctan2(gps_east, gps_north)) % 360.0
    climb_rate_gps = -gps_down  # ArduPilot VZ positive-up

    # --- Add sensor noise ---
    rng = np.random.default_rng(42)
    if noise_scale > 0:
        acc_x = acc_x + rng.normal(0, 0.01 * noise_scale, n)
        acc_y = acc_y + rng.normal(0, 0.01 * noise_scale, n)
        acc_z = acc_z + rng.normal(0, 0.01 * noise_scale, n)
        gyr_x = gyr_x + rng.normal(0, 0.001 * noise_scale, n)
        gyr_y = gyr_y + rng.normal(0, 0.001 * noise_scale, n)
        gyr_z = gyr_z + rng.normal(0, 0.001 * noise_scale, n)
        ground_speed_2d = ground_speed_2d + rng.normal(0, 0.05 * noise_scale, n)
        altitude = altitude + rng.normal(0, 0.2 * noise_scale, n)
        pitot_airspeed = pitot_airspeed + rng.normal(0, 0.05 * noise_scale, n)

    # --- Package into parse_log() format ---
    time_ms = (times * 1e3).astype(float)

    att_rows = [
        {
            "TimeMS": time_ms[i],
            "Roll": 0.0,
            "Pitch": float(np.degrees(theta[i])),
            "Yaw": 0.0,
        }
        for i in range(n)
    ]
    imu_rows = [
        {
            "TimeMS": time_ms[i],
            "AccX": float(acc_x[i]),
            "AccY": float(acc_y[i]),
            "AccZ": float(acc_z[i]),
            "GyrX": float(gyr_x[i]),
            "GyrY": float(gyr_y[i]),
            "GyrZ": float(gyr_z[i]),
        }
        for i in range(n)
    ]
    gps_rows = [
        {
            "T": time_ms[i],
            "Spd": float(ground_speed_2d[i]),
            "GCrs": float(ground_course[i]),
            "VZ": float(climb_rate_gps[i]),
            "Alt": float(altitude[i]),
        }
        for i in range(n)
    ]
    ctun_rows = [
        {
            "TimeMS": time_ms[i],
            "ThrOut": float(throttle_pct[i]),
        }
        for i in range(n)
    ]

    # ARSP: pitot measures body-x airspeed component u_r (not ground velocity)
    arsp_rows = [
        {
            "TimeMS": time_ms[i],
            "Airspeed": float(pitot_airspeed[i]),
        }
        for i in range(n)
    ]

    return {
        "ATT": att_rows,
        "IMU": imu_rows,
        "GPS": gps_rows,
        "CTUN": ctun_rows,
        "ARSP": arsp_rows,
    }


def main() -> int:
    aircraft = AircraftModel(
        mass=1.8,
        wing_area=0.32,
        wing_span=1.4,
        mean_aero_chord=0.23,
        i_xx=0.025,
        i_yy=0.030,
        i_zz=0.050,
        i_xz=0.001,
        max_thrust=12.0,
    )
    atmosphere = AtmosphereModel(rho=None, temperature_offset=0.0)

    # Generate synthetic flight data.  The ARSP channel activates the Johansen
    # wind observer.  For this straight-level profile the wind observer
    # converges gamma → 1 and wind → 0 immediately (no yaw variation).
    # To validate the KF with non-zero wind, real flight logs with figure-8 or
    # circular patterns (as used in Johansen 2015) are required.
    WIND = (0.0, 0.0, 0.0)
    log_data = generate_flight_data(dt=0.02, noise_scale=0.0, wind_ned=WIND)

    # Run pipeline (wind observer activates automatically via ARSP data)
    result = compute_from_log(log_data, aircraft, atmosphere)

    # --- Print results table ---
    t = result.time
    phase_samples = [
        ("Cruise (5s)", np.argmin(np.abs(t - 5.0))),
        ("Fast (17s)", np.argmin(np.abs(t - 17.0))),
        ("Climb (27s)", np.argmin(np.abs(t - 27.0))),
    ]

    print("=" * 80)
    print("Synthetic Fixed-Wing Flight Data — Aero Pipeline Validation")
    print("=" * 80)
    print()

    hdr = (
        f"{'Phase':<16} {'alpha':>6} {'beta':>6} {'V_TAS':>6} {'q':>7}"
        f"  {'CL':>7} {'CD':>7} {'Cm':>7} {'Cn':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for label, idx in phase_samples:
        print(
            f"{label:<16} "
            f"{np.degrees(result.alpha[idx]):6.2f} "
            f"{np.degrees(result.beta[idx]):6.2f} "
            f"{result.airspeed[idx]:6.2f} "
            f"{result.dynamic_pressure[idx]:7.1f}"
            f"  {result.cl[idx]:7.4f} {result.cd[idx]:7.4f}"
            f"  {result.cm[idx]:7.4f} {result.cn[idx]:7.4f}"
        )

    print()

    # --- Verification checks ---
    print("Verification checks (cruise phase at t=5 s):")
    idx = np.argmin(np.abs(t - 5.0))
    cl = result.cl[idx]
    cd = result.cd[idx]
    alpha_val = np.degrees(result.alpha[idx])
    beta_val = np.degrees(result.beta[idx])
    cm_val = result.cm[idx]
    cn_val = result.cn[idx]

    # Wind observer sanity checks — verifies the KF ran and is well-behaved.
    # Full wind-convergence validation requires attitude-varying manoeuvres.
    wind_checks: list[tuple[str, bool, float]] = []
    if result.wind_estimate is not None:
        wind_est = result.wind_estimate
        gamma_mid = wind_est.gamma[idx]
        wind_checks = [
            ("gamma → 1.0", abs(gamma_mid - 1.0) < 0.05, gamma_mid),
        ]

    checks = [
        ("CL ~ 0.40", abs(cl - 0.40) < 0.02, cl),
        ("CD ~ 0.07", abs(cd - 0.07) < 0.01, cd),
        ("alpha = 5 deg", abs(alpha_val - 5.0) < 0.1, alpha_val),
        ("beta ~ 0 deg", abs(beta_val) < 0.1, beta_val),
        ("Cm ~ 0 (steady)", abs(cm_val) < 0.01, cm_val),
        ("Cn ~ 0 (steady)", abs(cn_val) < 0.01, cn_val),
        *wind_checks,
    ]

    all_pass = True
    for name, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:30s} = {val:+.5f}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")

    # --- Plots ---
    plot_results(result)

    return 0 if all_pass else 1


def plot_results(result: "AeroCoefficients") -> None:
    """Plot aero coefficients and flight state."""
    t = result.time

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    # CL
    ax = axes[0, 0]
    ax.plot(t, result.cl)
    ax.set_ylabel("CL")
    ax.set_title("Lift Coefficient")
    ax.grid(True, alpha=0.3)

    # CD
    ax = axes[0, 1]
    ax.plot(t, result.cd)
    ax.set_ylabel("CD")
    ax.set_title("Drag Coefficient")
    ax.grid(True, alpha=0.3)

    # Alpha & Beta
    ax = axes[1, 0]
    ax.plot(t, np.degrees(result.alpha), label="alpha")
    ax.plot(t, np.degrees(result.beta), label="beta")
    ax.set_ylabel("Angle (deg)")
    ax.legend()
    ax.set_title("Angle of Attack & Sideslip")
    ax.grid(True, alpha=0.3)

    # Airspeed & dynamic pressure
    ax = axes[1, 1]
    ax.plot(t, result.airspeed, label="TAS")
    ax.set_ylabel("Airspeed (m/s)")
    ax.legend(loc="upper left")
    ax.set_title("True Airspeed & Dynamic Pressure")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, result.dynamic_pressure, color="tab:orange", label="q")
    ax2.set_ylabel("q (Pa)")
    ax2.legend(loc="upper right")

    # Pitch moment
    ax = axes[2, 0]
    ax.plot(t, result.cm)
    ax.set_ylabel("Cm")
    ax.set_xlabel("Time (s)")
    ax.set_title("Pitching Moment Coefficient")
    ax.grid(True, alpha=0.3)

    # Yaw & roll moments
    ax = axes[2, 1]
    ax.plot(t, result.cn, label="Cn")
    ax.plot(t, result.c_roll, label="Cl (roll)")
    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_title("Yaw & Roll Moment Coefficients")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Synthetic Fixed-Wing — Aero Pipeline Results", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
