"""
In this notebook, we will:

- list all Type A rainfall `.prcp` files,
- read each rainfall file as a time series,
- extract rainfall time and rainfall intensity,
- compute basic rainfall statistics such as number of time steps, time interval, final simulation hour, peak rainfall intensity, and total rainfall depth,
- display a summary table for all rainfall events,
- plot selected rainfall hyetographs using step plots,
- compare different return-period rainfall events visually.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_type_a_prcp(prcp_path):
    """Read one spatially uniform temporal rainfall file."""

    data = np.loadtxt(prcp_path, ndmin=2).astype(np.float32)  # Load two columns: time_seconds and rain_mmhr.

    if data.size==0:
        raise ValueError(f"Rainfall file {prcp_path} is empty.")

    time_seconds = data[:, 0]  # First column is elapsed time in seconds.
    rain_mmhr = data[:, 1]  # Second column is rainfall intensity in mm/hr.
    return time_seconds, rain_mmhr  # Return the rainfall time series.


def total_rainfall_depth_mm(time_seconds, rain_mmhr):
    """Compute total rainfall depth in millimetres."""
    
    time_seconds = np.asarray(time_seconds, dtype=np.float32)  # Ensure time is a NumPy array.
    rain_mmhr = np.asarray(rain_mmhr, dtype=np.float32)  # Ensure rainfall is a NumPy array.

    interval_hours = np.diff(time_seconds) / 3600.0  # Duration of each interval in hours

    interval_rainfall_mmhr = rain_mmhr[:-1]  # Rainfall intensity for each interval (excluding the last point)
    rainfall_depth_mm = interval_rainfall_mmhr * interval_hours  # Depth = intensity * duration
    return rainfall_depth_mm.sum()  # Total rainfall depth in millimetres.

def plot_hyetograph(time_seconds, rain_mmhr, title):
    """Plot one temporal rainfall event."""

    fig, ax = plt.subplots(figsize=(7, 4))  # Create a figure and axes.
    ax.step(time_seconds / 3600.0, rain_mmhr, where="post", linewidth=2)  # Step plot matches block rainfall.
    ax.scatter(time_seconds / 3600.0, rain_mmhr, s=25)  # Points show the actual input records.
    ax.set_title(title)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Rainfall [mm/hr]")
    ax.grid(True, alpha=0.3)
    plt.show()


def list_rainfall_files(RAINFALL_FOLDER):
    """List all Type A rainfall files."""

    return sorted(path.name for path in RAINFALL_FOLDER.glob("*.prcp"))  # Find and sort only rainfall files.


def summarize_one_rainfall(path):
    """Create one summary row for a rainfall event."""

    time_seconds, rain_mmhr = read_type_a_prcp(path)  # Read rainfall file.
    return {
        "rainfall_name": path.name,
        "n_steps": len(time_seconds),
        "dt_hours": float((time_seconds[1] - time_seconds[0]) / 3600.0),
        "end_hour": float(time_seconds[-1] / 3600.0),
        "max_mmhr": float(np.max(rain_mmhr)),
        "total_depth_mm": total_rainfall_depth_mm(time_seconds, rain_mmhr),
    }

