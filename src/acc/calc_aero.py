"""CLI command for computing aerodynamic coefficients from ArduPilot logs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from acc import plotting
from acc.aero import compute_from_log
from acc.log_parser import parse_log
from acc.models import AircraftConfig, AtmosphereConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute aerodynamic coefficients from ArduPilot .bin log files.",
    )
    parser.add_argument("logfile", help="Path to the .bin log file")
    parser.add_argument("config", help="Path to aircraft config JSON file")
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Fixed air density (kg/m³). Default: use ISA model.",
    )
    parser.add_argument(
        "--temperature-offset",
        type=float,
        default=0.0,
        help="ISA temperature offset delta-T (K). Default: 0.0",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive coefficient plots",
    )
    args = parser.parse_args()

    # Validate log file
    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Error: File '{args.logfile}' not found.")
        sys.exit(1)

    # Load aircraft config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    with open(config_path) as f:
        config_data = json.load(f)

    aircraft = AircraftConfig(**config_data)
    atmosphere = AtmosphereConfig(
        rho=args.rho, temperature_offset=args.temperature_offset
    )

    # Parse log (need ATT, IMU, GPS)
    print(f"Parsing '{log_path.name}'...")
    log_data = parse_log(str(log_path))

    # Compute coefficients
    print("Computing aerodynamic coefficients...")
    result = compute_from_log(log_data, aircraft, atmosphere)

    # Summary statistics (ignoring NaN from low-q masking)
    print(f"\nResults: {len(result.time)} time steps")
    valid = np.isfinite(result.cl)
    n_valid = np.sum(valid)
    print(f"Valid data points (q >= 10 Pa): {n_valid}")

    if n_valid > 0:
        print(f"  CL: {np.nanmean(result.cl):+.4f} ± {np.nanstd(result.cl):.4f}")
        print(f"  CD: {np.nanmean(result.cd):+.4f} ± {np.nanstd(result.cd):.4f}")
        print(f"  CY: {np.nanmean(result.cy):+.4f} ± {np.nanstd(result.cy):.4f}")
        cl_m = np.nanmean(result.c_roll)
        cl_s = np.nanstd(result.c_roll)
        print(f"  Cl: {cl_m:+.4f} ± {cl_s:.4f}")
        print(f"  Cm: {np.nanmean(result.cm):+.4f} ± {np.nanstd(result.cm):.4f}")
        print(f"  Cn: {np.nanmean(result.cn):+.4f} ± {np.nanstd(result.cn):.4f}")

    # Save CSV
    if args.output:
        header = "time,cl,cd,cy,c_roll,cm,cn,alpha,beta,q_dyn,airspeed"
        data = np.column_stack(
            [
                result.time,
                result.cl,
                result.cd,
                result.cy,
                result.c_roll,
                result.cm,
                result.cn,
                result.alpha,
                result.beta,
                result.dynamic_pressure,
                result.airspeed,
            ]
        )
        np.savetxt(args.output, data, delimiter=",", header=header, comments="")
        print(f"\nResults saved to '{args.output}'.")

    # Plot
    if args.plot:
        time = result.time

        plotting.subplots(
            time,
            {"CL": result.cl, "CD": result.cd, "CY": result.cy},
            title="Force Coefficients",
        )
        plotting.subplots(
            time,
            {
                "Cl (roll)": result.c_roll,
                "Cm (pitch)": result.cm,
                "Cn (yaw)": result.cn,
            },
            title="Moment Coefficients",
        )
        plotting.time_series(
            time,
            {"alpha (rad)": result.alpha, "beta (rad)": result.beta},
            title="Angles of Attack & Sideslip",
            ylabel="Angle (rad)",
        )
        plotting.show()


if __name__ == "__main__":
    main()
