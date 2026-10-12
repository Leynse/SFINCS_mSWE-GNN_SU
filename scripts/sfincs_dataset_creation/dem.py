"""
This .py file list all available `DEM_*.txt` files,
- find the available DEM files in the DEM folder,
- read one DEM file at a time,
- convert `x, y, elevation` points into a 2D grid,
- summarize the DEM elevations
- plot the DEM elevations.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def list_dem_files(DEM_FOLDER):
    """
    Find all DEM text files.
    
    DEM_FOLDER - Path to the DEM files
    """
    DEM_FOLDER = Path(DEM_FOLDER) # Convert the DEM folder into a Path
    dem_files = sorted(dem_path.name for dem_path in DEM_FOLDER.glob("DEM_*.txt"))  # Find and alphabetically sort DEM files.
    return dem_files


def read_dem_file(dem_path):
    """Read one DEM text file and convert it into a 2D elevation grid."""

    dem_path = Path(dem_path) #Convert the DEM file location into a Path
    dem_data = np.loadtxt(dem_path, ndmin=2)  # Load columns: x, y, elevation. ndmin=2 ensures that the output is always 2D, even if the file has only one row.

    if dem_data.size==0:
        raise ValueError(f"DEM file {dem_path} is empty.")

    x_coords = np.unique(dem_data[:, 0]).astype(np.float32)  # Unique x values define columns.
    y_coords = np.unique(dem_data[:, 1]).astype(np.float32)  # Unique y values define rows.

    nx = len(x_coords)  # Number of columns detected from file.
    ny = len(y_coords)  # Number of rows detected from file.

    if len(dem_data) != nx * ny:
        raise ValueError(f"DEM file {dem_path} does not contain a complete grid: expected {nx * ny} points, found {len(dem_data)}")

    sorting_order = np.lexsort((dem_data[:, 0], dem_data[:, 1]))  # Sort by increasing y and then increasing x.
    dem_grid = dem_data[sorting_order, 2].reshape(ny, nx).astype(np.float32)  # Convert the elevations into a 2D grid.
    
    return dem_grid, x_coords, y_coords


def summarize_dem(dem_path):
    """Return one summary row for one DEM."""

    dem_path = Path(dem_path)  # Convert the DEM file location into a Path.

    dem_grid, x_coords, y_coords = read_dem_file(dem_path)  # Load DEM.

    return {
        "dem_name": dem_path.name,
        "nx": int(dem_grid.shape[1]),
        "ny": int(dem_grid.shape[0]),
        "min_elevation_m": float(np.min(dem_grid)),
        "max_elevation_m": float(np.max(dem_grid)),
        "mean_elevation_m": float(np.mean(dem_grid)),
        "std_elevation_m": float(np.std(dem_grid)),
        "relief_m": float(np.max(dem_grid) - np.min(dem_grid)),
    }


def plot_dem_map(dem_grid, title):
    """Plot DEM as a terrain map."""

    fig, ax = plt.subplots(figsize=(6, 5))  # Create figure.
    image = ax.imshow(dem_grid, origin="lower", cmap="terrain")  # Show DEM grid.
    fig.colorbar(image, ax=ax, label="Elevation [m]")  # Add elevation scale.
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    plt.show()