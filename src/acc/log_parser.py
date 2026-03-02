"""ArduPilot .bin dataflash log parser."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pymavlink import mavutil


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
