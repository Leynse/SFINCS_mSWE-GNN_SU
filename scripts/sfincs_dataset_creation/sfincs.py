"""
Functions for preparing, running and saving Type A SFINCS simulations.
"""

from pathlib import Path
import json # Use json to save metadata in a simple text format.
import shutil # use shutil so we can delete and recreate folders.
import subprocess # Used to run external file such as sfincs.exe
import time # Used to measure how long SFINCS takes to run.

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr # Used to open sfincs_map.nc
from hydromt_sfincs import SfincsModel # Used to write sfincs.inp with HydroMT-SFINCS.

from scripts.sfincs_dataset_creation.dem import read_dem_file
from scripts.sfincs_dataset_creation.hyetograph import read_type_a_prcp

DTOUT_SECONDS = 1800 # SFINCS output interval in seconds.

#Create the case folder
def case_folder_type_A(dem_path, rainfall_name, CASE_ROOT, overwrite=True):
    """
    Create the folder for one Type A SFINCS simulation.

    dem_path: The path to the chosen DEM
    rainfall_name: The name of the chosen rainfall
    CASE_ROOT: CASE_ROOT/case_name will be the full path to the case folder
    """ 
    case_name = f"{Path(dem_path).stem}_{Path(rainfall_name).stem}_type_A"  # Create a clear name for this DEM-rainfall case.
    case_dir = CASE_ROOT / case_name  # Create the full path to the case folder.

    if case_dir.exists() and overwrite:  # Check whether the folder already exists and should be replaced.
        shutil.rmtree(case_dir)  # Delete the old case folder.

    case_dir.mkdir(parents=True, exist_ok=True)  # Create the new case folder.

    return case_name, case_dir  # Return the created case-folder path.


#Write elevation and mask files
def write_dep_and_mask(case_dir, dem_grid):
    """
    Write the SFINCS elevation and active-cell mask files.
    case_dir: SFINCS case folder
    dem_grid: 2D DEM elevation grid
    """
    mask_grid = np.isfinite(dem_grid).astype(np.uint8)  # Create a mask where finite elevations are active (1) and NaNs are inactive (0).

    np.savetxt(case_dir / "sfincs.dep", dem_grid, fmt="%.4f")  # Write DEM elevation to sfincs.dep.
    np.savetxt(case_dir / "sfincs.msk", mask_grid, fmt="%d")  # Write active-cell mask to sfincs.msk.

    return mask_grid  # Return the mask grid.


#Write sfincs.inp
def write_type_a_sfincs_inp(case_dir, dem_grid, x_coords, y_coords):
    """
    Write the configuration for one Type A SFINCS simulation.
    case_dir: SFINCS case folder
    dem_grid: 2D DEM elevation grid
    x_coords: DEM x coordinates
    y_coords: DEM y coordinates
    """
    dx = float(x_coords[1] - x_coords[0])  # Calculate x grid spacing.
    dy = float(y_coords[1] - y_coords[0])  # Calculate y grid spacing.

    sfincs = SfincsModel(
        root=str(case_dir), 
        mode="w", 
        config_fn="sfincs.inp", 
        write_gis=False)  # Create SFINCS model writer.

    sfincs.setup_config(
        mmax=dem_grid.shape[1],  # Set number of columns.
        nmax=dem_grid.shape[0],  # Set number of rows.
        dx=dx,  # Set x grid spacing.
        dy=dy,  # Set y grid spacing.
        x0=float(x_coords.min() - dx / 2.0),  # Set lower-left x coordinate.
        y0=float(y_coords.min() - dy / 2.0),  # Set lower-left y coordinate.
        rotation=0.0,  # Set grid rotation.
        epsg=32643,  # Set coordinate reference system.
        tref="20200101 000000",  # Set reference time.
        tstart="20200101 000000",  # Set simulation start time.
        tstop="20200102 210000",  # Set simulation stop time.
        dtout=DTOUT_SECONDS,  # Set map output interval.
        dthisout=DTOUT_SECONDS,  # Set history output interval.
        dtmaxout=10800,  # Set maximum output interval.
        alpha=0.5,  # Set numerical parameter alpha.
        theta=1.0,  # Set numerical parameter theta.
        huthresh=0.01,  # Set wet/dry threshold.
        manning=0.04,  # Set Manning roughness.
        qinf=0.0,  # Set infiltration to zero.
        zsini=-999.0,  # Start the model dry.
        inputformat="asc",  # Use ASCII input files.
        outputformat="net",  # Use NetCDF output.
        depfile="sfincs.dep",  # Tell SFINCS where the DEM file is.
        mskfile="sfincs.msk",  # Tell SFINCS where the mask file is.
        precipfile="sfincs.prcp",  # Tell SFINCS where the rainfall file is.
        storecumprcp=1,  # Store cumulative precipitation.
    )

    sfincs.write_config()  # Write sfincs.inp.

    return case_dir / "sfincs.inp"  # Return the sfincs.inp path.


#Run SFINCS
def run_sfincs(case_dir, case_name, log_folder, sfincs_exe, verbose=True):
    """
    Run SFINCS and save its terminal output.
    """
    log_folder = Path(log_folder)  # Ensure the log folder is a Path object.
    log_folder.mkdir(parents=True, exist_ok=True)  # Create the log folder if it doesn't exist.

    run_log = log_folder / f"{case_name}_run_log.txt"  # Define the run-log path.
    start_time = time.time()  # Save the start time.

    result = subprocess.run(
        [str(sfincs_exe)], 
        cwd=str(case_dir), 
        capture_output=True, 
        text=True)  # Run SFINCS inside the case folder.

    runtime_seconds = time.time() - start_time  # Calculate runtime in seconds.
    
    run_log.write_text(
        result.stdout + "\n\nSTDERR:\n" + result.stderr, 
        encoding="utf-8")  # Save the run log.

    if result.returncode != 0:  # Check whether SFINCS failed.
        raise RuntimeError(f"SFINCS failed. Check log: {run_log}")  # Stop if SFINCS failed.

    if not (case_dir / "sfincs_map.nc").exists():  # Check whether SFINCS produced the expected output file.
        raise RuntimeError(f"SFINCS did not produce sfincs_map.nc. Check log: {run_log}")  # Stop if the output file is missing.

    if verbose:
        print("Return code:", result.returncode)  # Print the SFINCS return code.
        print("Runtime seconds:", round(runtime_seconds, 2))  # Print the runtime.
        print("Run log:", run_log)  # Print the saved log path.

    return run_log, runtime_seconds  # Return log path and runtime.

#Plot the final water depth
def plot_final_water_depth(dataset, case_name):
    """Plot the final SFINCS water-depth map."""
    final_depth = dataset["h"].isel(time=-1).values # Select the final water depth.
    final_depth = np.squeeze(final_depth)   # Remove any singleton dimensions to ensure the array is 2D.
    final_depth = np.ma.masked_where(final_depth <= 0.01, final_depth)  # Mask dry cells.

    fig, ax = plt.subplots(figsize=(6, 5))  # Create a figure and axes.
    image = ax.imshow(final_depth, origin="lower", cmap="Blues", vmin=0.01)  # Plot final water depth.
    fig.colorbar(image, ax=ax, label="Water depth [m]")  # Add colorbar.
    ax.set_title(f"Final water depth: {case_name}")  # Add title.
    ax.set_xlabel("Grid column")  # Label x-axis.
    ax.set_ylabel("Grid row")  # Label y-axis.
    fig.tight_layout()
    plt.show()

#Plot flood evolution 
def plot_flood_evolution(
    dataset,
    case_name,
    interval_hours=3,
    dtout_seconds=1800,
):
    """Plot water depth at the selected interval."""

    water_depth = dataset["h"].transpose("time", ...).values 
    hours_per_frame = dtout_seconds / 3600 # Convert output interval to hours.
    frame_step = round(interval_hours / hours_per_frame) # Determine how many frames to skip for the desired interval.
    frames = list(range(0, len(water_depth), frame_step))  # Select frames to plot based on the interval.

    columns = 4
    rows = int(np.ceil(len(frames) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 2.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.flatten()
    max_depth = np.nanmax(water_depth)

    for axis, frame in zip(axes, frames):
        image = axis.imshow(
            water_depth[frame],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=max_depth,
        )

        time_hours = frame * hours_per_frame
        axis.set_title(f"{time_hours:g} hours")
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(frames):]:
        axis.axis("off")

    fig.colorbar(image, ax=axes.tolist(), label="Water depth [m]")
    fig.suptitle(case_name.replace("_", " "))
    plt.show()

    return fig


#Prepare and run one complete case
def run_type_A_case(dem_path, rainfall_path, case_root, log_folder, sfincs_exe, plot=False, verbose=True):
    """
    Prepare and run one Type A SFINCS simulation.
    """
    dem_path = Path(dem_path)  # Ensure DEM path is a Path object.
    rainfall_path = Path(rainfall_path)  # Ensure rainfall path is a Path object.

    case_name, case_dir = case_folder_type_A(
        dem_path, 
        rainfall_path,
        case_root,
    )

    dem_grid, x_coords, y_coords = read_dem_file(dem_path)  # Read the DEM file and convert to a grid.
    write_dep_and_mask(case_dir, dem_grid)  # Write the elevation and mask files.
    shutil.copyfile(rainfall_path, case_dir / "sfincs.prcp")  # Copy the rainfall file to the case directory.

    write_type_a_sfincs_inp(
        case_dir, 
        dem_grid, 
        x_coords, 
        y_coords
    )  # Write the SFINCS input file.

    run_log, runtime_seconds = run_sfincs(
        case_dir, 
        case_name, 
        log_folder,
        sfincs_exe, 
        verbose=verbose,
    )  # Run SFINCS and capture the log and runtime.

    metadata = {
        "case_name": case_name,
        "dem_name": dem_path.name,
        "rainfall_name": rainfall_path.name,
        "rainfall_type": "type_A",
        "dx": float(x_coords[1] - x_coords[0]),
        "dy": float(y_coords[1] - y_coords[0]),
        "runtime_seconds": float(runtime_seconds),
    }

    if plot:
        with xr.open_dataset(case_dir / "sfincs_map.nc") as dataset:
            print("h dimensions:", dataset["h"].dims)
            print("h shape:", dataset["h"].shape)
   
            plot_final_water_depth(dataset, case_name)  # Plot the final water depth.
            plot_flood_evolution(dataset, case_name, interval_hours=3, dtout_seconds=DTOUT_SECONDS)  # Plot the flood evolution.

    if verbose:
        print(f"Completed {case_name} in {runtime_seconds:.2f} seconds. Log saved to {run_log}.")

    return case_dir, metadata



#Save the mSWE-GNN training 
def save_training_npz_type_A(case_dir, processed_folder, metadata):
    """
    Save one SFINCS simulation for mSWE-GNN dataset creation.
    """
    case_dir = Path(case_dir)  # Ensure case_dir is a Path object.
    processed_folder = Path(processed_folder)  # Ensure processed_folder is a Path object.
    processed_folder.mkdir(parents=True, exist_ok=True)  # Create the processed folder if

    dem =np.loadtxt(case_dir / "sfincs.dep").astype(np.float32)  # Load the DEM elevation grid.
    rainfall_time_seconds, rain_mmhr = read_type_a_prcp(case_dir / "sfincs.prcp")  # Load the rainfall time series.

    """dataset[time].values - SFINCS stores date and time corresponding to every saved water-depth map."""
    with xr.open_dataset(case_dir / "sfincs_map.nc") as dataset:
        water_depth = (dataset["h"].transpose("time", ...).values.astype(np.float32))
        model_time = dataset["time"].values  # Load the model time series.

    water_depth = np.maximum(np.nan_to_num(water_depth), 0.0)  # Ensure water depth is non-negative.

    """model_time_seconds stores into simulation time [0, 1800, 3600, ...] seconds corresponding to each water-depth map."""
    model_time_seconds = ((model_time - model_time[0]) / np.timedelta64(1, "s")).astype(np.float32)  # Convert model time to seconds.

    processed_file = processed_folder / f"{metadata['case_name']}.npz"  # Define the output NPZ file path.

    np.savez_compressed(
        processed_file,
        dem=dem,
        water_depth=water_depth,
        model_time_seconds=model_time_seconds,
        rainfall_time_seconds=rainfall_time_seconds.astype(np.float32),
        rain_mmhr=rain_mmhr.astype(np.float32),
        metadata=json.dumps(metadata)  # Save metadata as a JSON string.
    )

    return processed_file  # Return the path to the saved NPZ file.