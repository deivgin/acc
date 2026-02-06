"""ArduPilot .bin dataflash log parser."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from pymavlink import mavutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ArduPilot .bin dataflash log files."
    )
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
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        print(f"Error: File '{args.logfile}' not found.")
        sys.exit(1)

    if not path.suffix.lower() == ".bin":
        print(f"Warning: Expected a .bin file, got '{path.suffix}'")

    mlog = mavutil.mavlink_connection(str(path))

    counts: Counter[str] = Counter()
    messages: list[object] = []

    print(f"Parsing '{path.name}'...")

    while True:
        msg = mlog.recv_match(type=args.type, blocking=False)
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            continue
        counts[msg_type] += 1
        messages.append(msg)

    if not messages:
        if args.type:
            print(f"No '{args.type}' messages found in the log.")
        else:
            print("No messages found in the log file.")
        return

    # Print summary
    print("=" * 60)
    print(f"Log file: {path.name}")
    print(f"Total messages: {sum(counts.values())}")
    print(f"Message types: {len(counts)}")
    print("=" * 60)

    print("\nMessage type counts:")
    print("-" * 40)
    for msg_type, count in counts.most_common():
        print(f"  {msg_type:<25} {count:>8}")

    # Print messages
    display_count = len(messages) if args.all else min(args.count, len(messages))
    print(f"\n{'=' * 60}")
    if args.all:
        print(f"All {display_count} messages:")
    else:
        print(f"First {display_count} of {len(messages)} messages:")
    print("=" * 60)

    for msg in messages[:display_count]:
        msg_type = msg.get_type()
        fields = msg.get_fieldnames()
        print(f"\n[{msg_type}]")
        for field in fields:
            value = getattr(msg, field)
            print(f"  {field}: {value}")


if __name__ == "__main__":
    main()
