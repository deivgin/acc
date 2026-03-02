"""CLI command for plotting ArduPilot .bin dataflash log data."""

import argparse
import sys
from pathlib import Path

from acc import plotting
from acc.log_parser import parse_log

PRESETS = {
    "att": {
        "type": "ATT",
        "fields": ["Roll", "Pitch", "Yaw"],
        "title": "Attitude (Roll / Pitch / Yaw)",
        "ylabel": "Degrees",
        "plot": "time_series",
    },
    "gps": {
        "type": "GPS",
        "fields": ["Lat", "Lng", "Alt"],
        "title": "3D GPS Trajectory",
        "plot": "trajectory_3d",
    },
    "imu": {
        "type": "IMU",
        "fields": ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        "title": "IMU Readings",
        "plot": "subplots",
    },
    "baro": {
        "type": "BARO",
        "fields": ["Alt"],
        "title": "Barometric Altitude",
        "ylabel": "Altitude (m)",
        "plot": "time_series",
    },
}


def _extract_field(rows: list[dict], field: str) -> list[float]:
    """Extract a single field from a list of message dicts as floats."""
    return [float(row[field]) for row in rows]


def _time_index(rows: list[dict]) -> list[float]:
    """Build a time axis from message rows.

    Uses the TimeMS field (milliseconds) if available, otherwise falls back
    to a simple 0-based integer index.
    """
    if rows and "TimeMS" in rows[0]:
        return [float(row["TimeMS"]) / 1e3 for row in rows]
    return list(range(len(rows)))


def _plot_preset(data: dict, preset_name: str) -> plotting.Figure:
    preset = PRESETS[preset_name]
    msg_type = preset["type"]

    if msg_type not in data:
        print(f"Error: No '{msg_type}' messages found in the log.")
        sys.exit(1)

    rows = data[msg_type]

    if preset["plot"] == "trajectory_3d":
        lat = _extract_field(rows, "Lat")
        lon = _extract_field(rows, "Lng")
        alt = _extract_field(rows, "Alt")
        return plotting.trajectory_3d(lat, lon, alt, title=preset["title"])

    time = _time_index(rows)

    if preset["plot"] == "subplots":
        values = {f: _extract_field(rows, f) for f in preset["fields"]}
        return plotting.subplots(time, values, title=preset["title"])  # type: ignore

    # Default: time_series
    values = {f: _extract_field(rows, f) for f in preset["fields"]}
    return plotting.time_series(
        time,
        values,  # type: ignore
        title=preset["title"],
        ylabel=preset.get("ylabel", ""),
    )


def _plot_generic(data: dict, msg_type: str, fields: list[str]) -> plotting.Figure:
    if msg_type not in data:
        print(f"Error: No '{msg_type}' messages found in the log.")
        sys.exit(1)

    rows = data[msg_type]
    available = set(rows[0].keys()) if rows else set()
    missing = [f for f in fields if f not in available]
    if missing:
        print(f"Error: Fields {missing} not found in '{msg_type}' messages.")
        print(f"Available fields: {sorted(available)}")
        sys.exit(1)

    time = _time_index(rows)
    values = {f: _extract_field(rows, f) for f in fields}

    if len(fields) > 3:
        return plotting.subplots(
            time,
            values,  # type: ignore
            title=f"{msg_type} — {', '.join(fields)}",
        )
    return plotting.time_series(time, values, title=f"{msg_type} — {', '.join(fields)}")  # type: ignore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("logfile", help="Path to the .bin log file")
    parser.add_argument(
        "preset",
        nargs="?",
        choices=list(PRESETS.keys()),
        default=None,
        help="Preset plot mode: att, gps, imu, baro",
    )
    parser.add_argument(
        "-t",
        "--type",
        default=None,
        help="Message type for generic mode (e.g. IMU, GPS)",
    )
    parser.add_argument(
        "-f",
        "--fields",
        default=None,
        help="Comma-separated field names for generic mode (e.g. AccX,AccY,AccZ)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save plot to file instead of showing interactively",
    )


def run(args: argparse.Namespace) -> None:
    path = Path(args.logfile)
    if not path.exists():
        print(f"Error: File '{args.logfile}' not found.")
        sys.exit(1)

    # Determine which message type to parse
    if args.preset:
        filter_type = PRESETS[args.preset]["type"]
    elif args.type:
        filter_type = args.type
    else:
        print("Error: Provide a preset (att, gps, imu, baro) or --type with --fields.")
        sys.exit(2)

    print(f"Parsing '{path.name}'...")
    data = parse_log(str(path), message_type=filter_type)

    if not data:
        print(f"No '{filter_type}' messages found in the log.")
        sys.exit(1)

    if args.preset:
        fig = _plot_preset(data, args.preset)
    else:
        if not args.fields:
            print("Error: --fields is required when using --type.")
            sys.exit(2)
        fields = [f.strip() for f in args.fields.split(",")]
        fig = _plot_generic(data, args.type, fields)

    if args.save:
        plotting.save(fig, args.save)
        print(f"Plot saved to '{args.save}'.")
    else:
        plotting.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot data from ArduPilot .bin dataflash log files.",
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
