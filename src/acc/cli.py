"""Unified CLI entry point for acc tools."""

import argparse

from acc import calc_aero, log_parser, plot_log


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="acc",
        description="ArduPilot aerodynamic coefficient tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # calc_aero
    p = subparsers.add_parser(
        "calc_aero",
        help="Compute aerodynamic coefficients from a .bin log",
    )
    calc_aero.add_arguments(p)
    p.set_defaults(func=calc_aero.run)

    # plot_log
    p = subparsers.add_parser(
        "plot_log",
        help="Plot data from a .bin dataflash log",
    )
    plot_log.add_arguments(p)
    p.set_defaults(func=plot_log.run)

    # parse_log
    p = subparsers.add_parser(
        "parse_log",
        help="Parse and inspect a .bin dataflash log",
    )
    log_parser.add_arguments(p)
    p.set_defaults(func=log_parser.run)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
