"""ArduPilot .bin dataflash log parser and flight state extraction."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pymavlink import mavutil

from acc.log_parser.common import extract_time_and_field
from acc.model.aero_coefficients import AeroCoefficients
from acc.model.aircraft_model import AircraftModel
from acc.model.atmosphere_model import AtmosphereModel
from acc.model.flight_state import FlightState


def parse_log(
    filepath: str, message_type: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Parse an ArduPilot .bin dataflash log file.

    Args:
        filepath: Path to the .bin log file.
        message_type: Optional message type filter (e.g. "GPS", "ATT").

    Returns:
        Dict keyed by message type, where each value is a list of dicts
        (one per message, field name to value).
    """
    mlog = mavutil.mavlink_connection(filepath)
    result: dict[str, list[dict[str, Any]]] = {}

    while True:
        msg = mlog.recv_match(type=message_type, blocking=False)
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            continue
        fields = msg.get_fieldnames()
        row = {field: getattr(msg, field) for field in fields}
        if hasattr(msg, "TimeMS") and "TimeMS" not in row:
            row["TimeMS"] = msg.TimeMS
        result.setdefault(msg_type, []).append(row)

    return result


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
    t_imu, acc_x = extract_time_and_field(imu_rows, "AccX")
    _, acc_y = extract_time_and_field(imu_rows, "AccY")
    _, acc_z = extract_time_and_field(imu_rows, "AccZ")
    _, gyr_x = extract_time_and_field(imu_rows, "GyrX")
    _, gyr_y = extract_time_and_field(imu_rows, "GyrY")
    _, gyr_z = extract_time_and_field(imu_rows, "GyrZ")

    acc_x_i = np.interp(t_att, t_imu, acc_x)
    acc_y_i = np.interp(t_att, t_imu, acc_y)
    acc_z_i = np.interp(t_att, t_imu, acc_z)
    gyr_x_i = np.interp(t_att, t_imu, gyr_x)
    gyr_y_i = np.interp(t_att, t_imu, gyr_y)
    gyr_z_i = np.interp(t_att, t_imu, gyr_z)

    # GPS fields — interpolate to ATT time grid
    t_gps = np.array([float(r["T"]) / 1e3 for r in gps_rows])
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

    result = {
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

    # CTUN throttle (optional)
    if "CTUN" in log_data and log_data["CTUN"]:
        ctun_rows = log_data["CTUN"]
        t_ctun, thr_out = extract_time_and_field(ctun_rows, "ThrOut")
        throttle_i = np.interp(t_att, t_ctun, thr_out / 100.0)
        result["throttle"] = throttle_i

    # ARSP airspeed (optional) — used by the wind observer (Johansen 2015).
    # ArduPilot 4.x uses TimeUS (microseconds); older firmware uses TimeMS.
    # The logged value is total airspeed V_a (m/s) from the pitot tube.
    # For a fixed-mount pitot at typical UAV AOA (< 15°), V_a ≈ u_r (body-x
    # airspeed component) and is used directly as u_r^m in the KF.
    if "ARSP" in log_data and log_data["ARSP"]:
        arsp_rows = log_data["ARSP"]
        sample = arsp_rows[0]
        if "TimeUS" in sample:
            t_arsp, airspeed = extract_time_and_field(
                arsp_rows, "Airspeed", time_field="TimeUS", time_scale=1e-6
            )
        else:
            t_arsp, airspeed = extract_time_and_field(arsp_rows, "Airspeed")
        result["pitot_airspeed"] = np.interp(t_att, t_arsp, airspeed)

    return result


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


def compute_from_log(
    log_data: dict[str, list[dict[str, Any]]],
    aircraft: AircraftModel,
    atmosphere: AtmosphereModel,
) -> AeroCoefficients:
    """Convenience end-to-end wrapper: log data -> aerodynamic coefficients."""
    from acc.aero.compute_coefficients import compute_coefficients

    state = extract_flight_state(log_data)
    return compute_coefficients(state, aircraft, atmosphere)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("logfile", help="Path to the .bin log file")
    parser.add_argument(
        "-t",
        "--type",
        help="Filter by message type (e.g. GPS, ATT, IMU, BARO)",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        help="Number of messages to display (default: 20)",
        default=20,
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Display all messages (overrides --count)",
    )


def run(args: argparse.Namespace) -> None:
    path = Path(args.logfile)
    if not path.exists():
        print(f"Error: File '{args.logfile}' not found.")
        sys.exit(1)

    if not path.suffix.lower() == ".bin":
        print(f"Warning: Expected a .bin file, got '{path.suffix}'")

    print(f"Parsing '{path.name}'...")

    data = parse_log(str(path), message_type=args.type)

    if not data:
        if args.type:
            print(f"No '{args.type}' messages found in the log.")
        else:
            print("No messages found in the log file.")
        return

    counts: Counter[str] = Counter()
    for msg_type, rows in data.items():
        counts[msg_type] = len(rows)

    total = sum(counts.values())

    # Print summary
    print("=" * 60)
    print(f"Log file: {path.name}")
    print(f"Total messages: {total}")
    print(f"Message types: {len(counts)}")
    print("=" * 60)

    print("\nMessage type counts:")
    print("-" * 40)
    for msg_type, count in counts.most_common():
        print(f"  {msg_type:<25} {count:>8}")

    # Flatten messages in original order for display
    messages: list[tuple[str, dict[str, Any]]] = []
    for msg_type, rows in data.items():
        for row in rows:
            messages.append((msg_type, row))

    # Print messages
    display_count = len(messages) if args.all else min(args.count, len(messages))
    print(f"\n{'=' * 60}")
    if args.all:
        print(f"All {display_count} messages:")
    else:
        print(f"First {display_count} of {len(messages)} messages:")
    print("=" * 60)

    for msg_type, row in messages[:display_count]:
        print(f"\n[{msg_type}]")
        for field, value in row.items():
            print(f"  {field}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ArduPilot .bin dataflash log files."
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
