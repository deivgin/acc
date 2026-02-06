"""Graph visualization module for aerodynamic calculations and flight data."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def time_series(
    time: list[float] | np.ndarray,
    values: dict[str, list[float] | np.ndarray],
    *,
    title: str = "Time Series",
    xlabel: str = "Time (s)",
    ylabel: str = "",
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot one or more variables over time.

    Args:
        time: Time values for the x-axis.
        values: Mapping of label name to data array. Each entry becomes a line.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches (width, height).

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, data in values.items():
        ax.plot(time, data, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def xy_plot(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    *,
    title: str = "XY Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    style: str = "-",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot two variables against each other.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        style: Matplotlib line style string (e.g. '-', '--', 'o', '.-').
        figsize: Figure size in inches (width, height).

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, style)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def multi_xy_plot(
    datasets: list[dict[str, Any]],
    *,
    title: str = "XY Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot multiple XY datasets on the same axes.

    Args:
        datasets: List of dicts, each with keys 'x', 'y', and optional 'label'
                  and 'style'.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches (width, height).

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for ds in datasets:
        style = ds.get("style", "-")
        label = ds.get("label", None)
        ax.plot(ds["x"], ds["y"], style, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any("label" in ds for ds in datasets):
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def trajectory_3d(
    lat: list[float] | np.ndarray,
    lon: list[float] | np.ndarray,
    alt: list[float] | np.ndarray,
    *,
    title: str = "3D Flight Trajectory",
    figsize: tuple[float, float] = (12, 9),
    colorbar_label: str = "Altitude",
) -> Figure:
    """Plot a 3D flight trajectory colored by altitude.

    Args:
        lat: Latitude values.
        lon: Longitude values.
        alt: Altitude values.
        title: Plot title.
        figsize: Figure size in inches (width, height).
        colorbar_label: Label for the colorbar.

    Returns:
        The matplotlib Figure.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)

    fig = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]

    scatter = ax.scatter(lon, lat, alt, c=alt, cmap="viridis", s=1)
    ax.plot(lon, lat, alt, alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Altitude")  # type: ignore[attr-defined]
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=colorbar_label, shrink=0.6)
    fig.tight_layout()
    return fig


def subplots(
    time: list[float] | np.ndarray,
    values: dict[str, list[float] | np.ndarray],
    *,
    title: str = "Multi-panel Plot",
    xlabel: str = "Time (s)",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot each variable in its own subplot, sharing the x-axis.

    Args:
        time: Shared time values for the x-axis.
        values: Mapping of label name to data array. Each entry gets its own panel.
        title: Overall figure title.
        xlabel: X-axis label (shown on bottom panel only).
        figsize: Figure size. Defaults to (12, 3 * number of panels).

    Returns:
        The matplotlib Figure.
    """
    n = len(values)
    if figsize is None:
        figsize = (12, 3 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, values.items()):  # type: ignore[arg-type]
        ax.plot(time, data)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)  # type: ignore[index]
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def show() -> None:
    """Display all open figures. Convenience wrapper around plt.show()."""
    plt.show()


def save(fig: Figure, path: str, *, dpi: int = 150) -> None:
    """Save a figure to a file.

    Args:
        fig: The figure to save.
        path: Output file path (e.g. 'plot.png', 'plot.pdf').
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
