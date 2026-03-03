"""Generate synthetic fixed-wing flight data and validate the aero pipeline.

Creates a 30-second multi-phase flight profile for a Bixler-class fixed-wing UAV
and runs it through the aerodynamic coefficient pipeline with and without thrust
correction to demonstrate the effect of propulsive force subtraction.

Aircraft: small fixed-wing UAV (Bixler-class), 1.8 kg, 0.32 m² wing area.

Flight profile (3 phases, 30 s total):
  1. Cruise    (0-10 s):  15 m/s north, alpha=5°, 50% throttle, level at 100 m
  2. Fast cruise (12-22 s): 20 m/s north, alpha=3°, 40% throttle, level at 100 m
  3. Climb    (24-30 s): 15 m/s north, alpha=8°, 75% throttle, climb 2 m/s

Smooth cosine transitions between phases (10-12 s and 22-24 s).

Usage:
    python scripts/synthetic_fixed_wing.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acc.aero import compute_from_log
from acc.models import AircraftConfig, AtmosphereConfig

G = 9.80665  # m/s²


def _cosine_blend(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Cosine interpolation factor: 0 before t0, smoothly rises to 1 at t1."""
    frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * frac))


def _piecewise_profile(
    t: np.ndarray, v1: float, v2: float, v3: float
) -> np.ndarray:
    """Three-phase piecewise profile with cosine transitions.

    Phase 1: 0-10 s, Transition: 10-12 s, Phase 2: 12-22 s,
    Transition: 22-24 s, Phase 3: 24-30 s.
    """
    b1 = _cosine_blend(t, 10.0, 12.0)
    b2 = _cosine_blend(t, 22.0, 24.0)
    return v1 + b1 * (v2 - v1) + b2 * (v3 - v2)


def generate_flight_data(
    dt: float = 0.02, noise_scale: float = 0.0
) -> dict[str, list[dict[str, float]]]:
    """Generate synthetic flight log data matching parse_log() output format.

    Args:
        dt: Sample period in seconds (default 50 Hz).
        noise_scale: Multiplier for sensor noise (0 = no noise).

    Returns:
        Dict with ATT, IMU, GPS, CTUN message lists matching parse_log() format.
    """
    times = np.arange(0, 30.0 + dt, dt)
    n = len(times)

    # --- Build flight parameter profiles ---
    v_ground = _piecewise_profile(times, 15.0, 20.0, 15.0)
    alpha_deg = _piecewise_profile(times, 5.0, 3.0, 8.0)
    throttle_pct = _piecewise_profile(times, 50.0, 40.0, 75.0)
    vz = _piecewise_profile(times, 0.0, 0.0, 2.0)  # climb rate, positive up

    alpha_rad = np.radians(alpha_deg)

    # NED velocities (flying north, wings level)
    v_north = v_ground
    v_east = np.zeros(n)
    v_down = -vz  # NED convention: down is positive

    # Pitch angle (theta) from desired alpha and velocity components
    # Derived from: alpha = atan2(w, u) with body velocities from NED rotation
    # tan(theta) = (tan(alpha) * v_n - v_d) / (v_n + tan(alpha) * v_d)
    tan_a = np.tan(alpha_rad)
    theta = np.arctan2(tan_a * v_north - v_down, v_north + tan_a * v_down)

    # Altitude: integrate climb rate
    altitude = 100.0 + np.cumsum(vz * dt)
    altitude = np.concatenate(([100.0], altitude[:-1]))  # shift so t=0 starts at 100

    # --- IMU specific force ---
    # a_specific = R_body_NED * (a_NED - g_NED)
    # For phi=0, psi=0:
    #   AccX = cos(theta) * dvn/dt - sin(theta) * (dvd/dt - g)
    #   AccZ = sin(theta) * dvn/dt + cos(theta) * (dvd/dt - g)
    dvn_dt = np.gradient(v_north, times)
    dvd_dt = np.gradient(v_down, times)
    ct = np.cos(theta)
    st = np.sin(theta)

    acc_x = ct * dvn_dt - st * (dvd_dt - G)
    acc_y = np.zeros(n)
    acc_z = st * dvn_dt + ct * (dvd_dt - G)

    # Gyroscope: only pitch rate is nonzero (wings level, heading constant)
    gyr_x = np.zeros(n)
    gyr_y = np.gradient(theta, times)
    gyr_z = np.zeros(n)

    # --- Add sensor noise ---
    rng = np.random.default_rng(42)
    if noise_scale > 0:
        acc_x = acc_x + rng.normal(0, 0.01 * noise_scale, n)
        acc_y = acc_y + rng.normal(0, 0.01 * noise_scale, n)
        acc_z = acc_z + rng.normal(0, 0.01 * noise_scale, n)
        gyr_x = gyr_x + rng.normal(0, 0.001 * noise_scale, n)
        gyr_y = gyr_y + rng.normal(0, 0.001 * noise_scale, n)
        gyr_z = gyr_z + rng.normal(0, 0.001 * noise_scale, n)
        v_ground = v_ground + rng.normal(0, 0.05 * noise_scale, n)
        altitude = altitude + rng.normal(0, 0.2 * noise_scale, n)

    # GPS ground speed and course
    ground_speed = np.abs(v_ground)  # ensure non-negative after noise
    ground_course = np.zeros(n)  # heading north = 0°

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
            "Spd": float(ground_speed[i]),
            "GCrs": float(ground_course[i]),
            "VZ": float(vz[i]),
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

    return {"ATT": att_rows, "IMU": imu_rows, "GPS": gps_rows, "CTUN": ctun_rows}


def main() -> int:
    # Load aircraft config
    config_path = Path(__file__).resolve().parent.parent / "aircraft_fixed_wing.json"
    with open(config_path) as f:
        config_data = json.load(f)

    aircraft_with_thrust = AircraftConfig(**config_data)
    config_no_thrust = {k: v for k, v in config_data.items() if k != "max_thrust"}
    aircraft_no_thrust = AircraftConfig(**config_no_thrust)
    atmosphere = AtmosphereConfig()

    # Generate synthetic flight data
    log_data = generate_flight_data(dt=0.02, noise_scale=0.0)

    # Run pipeline: without and with thrust correction
    result_nt = compute_from_log(log_data, aircraft_no_thrust, atmosphere)
    result_wt = compute_from_log(log_data, aircraft_with_thrust, atmosphere)

    # --- Print comparison table ---
    t = result_nt.time
    phase_samples = [
        ("Cruise (5s)", np.argmin(np.abs(t - 5.0))),
        ("Fast (17s)", np.argmin(np.abs(t - 17.0))),
        ("Climb (27s)", np.argmin(np.abs(t - 27.0))),
    ]

    print("=" * 90)
    print("Synthetic Fixed-Wing Flight Data — Aero Pipeline Validation")
    print("=" * 90)
    print()

    hdr = (
        f"{'Phase':<16} {'alpha':>6} {'beta':>6} {'V_TAS':>6} {'q':>7}"
        f"  {'CL_nt':>7} {'CD_nt':>7} {'CL_wt':>7} {'CD_wt':>7}"
        f"  {'Cm_nt':>7} {'Cn_nt':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for label, idx in phase_samples:
        print(
            f"{label:<16} "
            f"{np.degrees(result_nt.alpha[idx]):6.2f} "
            f"{np.degrees(result_nt.beta[idx]):6.2f} "
            f"{result_nt.airspeed[idx]:6.2f} "
            f"{result_nt.dynamic_pressure[idx]:7.1f}"
            f"  {result_nt.cl[idx]:7.4f} {result_nt.cd[idx]:7.4f}"
            f" {result_wt.cl[idx]:7.4f} {result_wt.cd[idx]:7.4f}"
            f"  {result_nt.cm[idx]:7.4f} {result_nt.cn[idx]:7.4f}"
        )

    print()
    print("  _nt = no thrust correction    _wt = with thrust correction")
    print()

    # --- Verification checks ---
    print("Verification checks (cruise phase at t=5 s):")
    idx = np.argmin(np.abs(t - 5.0))
    cl_nt = result_nt.cl[idx]
    cd_nt = result_nt.cd[idx]
    cl_wt = result_wt.cl[idx]
    cd_wt = result_wt.cd[idx]
    alpha_val = np.degrees(result_nt.alpha[idx])
    beta_val = np.degrees(result_nt.beta[idx])
    cm_val = result_nt.cm[idx]
    cn_val = result_nt.cn[idx]

    checks = [
        ("CL (no thrust) ~ 0.40", abs(cl_nt - 0.40) < 0.02, cl_nt),
        ("CD (no thrust) ~ 0.00", abs(cd_nt) < 0.005, cd_nt),
        ("CL (with thrust) ~ 0.40", abs(cl_wt - 0.40) < 0.02, cl_wt),
        ("CD (with thrust) ~ 0.07", abs(cd_wt - 0.07) < 0.01, cd_wt),
        ("alpha = 5 deg", abs(alpha_val - 5.0) < 0.1, alpha_val),
        ("beta ~ 0 deg", abs(beta_val) < 0.1, beta_val),
        ("Cm ~ 0 (steady)", abs(cm_val) < 0.01, cm_val),
        ("Cn ~ 0 (steady)", abs(cn_val) < 0.01, cn_val),
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

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
