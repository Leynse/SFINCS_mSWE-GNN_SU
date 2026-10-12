"""
Stage 1 of 3: turn one SFINCS simulation (.npz) into one graph.

Each simulation has a DEM, a water-depth time series, and a rainfall time
series. This file builds a MultiScaleData graph at three resolutions -
fine (full), medium (2x2 pooled), coarse (4x4 pooled) - so the GNN can pass
messages over both short and long distances.

This step is expensive (grid building, pooling) but doesn't depend on
history length, prediction horizon, or model choice (GNN vs GraphSAGE) -
build it once per simulation and reuse it everywhere. The full, un-windowed
time series is stored here; windowing happens in temporal_learning.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data #torch_geometric is a library for graph neural networks, and Data is the basic graph container


# ============================================================
# 1. The graph container
# ============================================================

class MultiScaleData(Data):
    """
    Holds fine, medium, and coarse graphs in one PyG Data object.
    """
    def __inc__(self, key, value, *args, **kwargs):
        # When PyG glues multiple graphs into one batch, node indices must be
        # renumbered so they don't overlap. This tells PyG how much to shift
        # each of our multiscale index tensors by.
        index_increments = {
            "fine_edge_index": self.num_fine_nodes,
            "medium_edge_index": self.num_medium_nodes,
            "coarse_edge_index": self.num_coarse_nodes,
            "fine_to_medium": self.num_medium_nodes,
            "medium_to_coarse": self.num_coarse_nodes,
        }
        if key in index_increments:
            return index_increments[key]
        return super().__inc__(key, value, *args, **kwargs)


# ============================================================
# 2. Building one resolution level of the grid
# ============================================================

def build_grid_graph(number_of_columns, number_of_rows, dx, dy):
    """
    Build a simple grid graph: every cell is connected to its left/right and
    up/down neighbour (a "4-neighbour" grid), in both directions.

    Returns:
        edge_index    - which node connects to which node
        edge_distance - the physical distance (in metres) of every connection
        positions     - the [x, y] real-world coordinate of every node
    """

    # Give every cell a unique node number, arranged like the grid itself
    # torch.arange() creates a 1D tensor of numbers from 0 to (number_of_rows * number_of_columns - 1)
    node_grid = torch.arange(
        number_of_rows * number_of_columns, dtype=torch.long
    ).reshape(number_of_rows, number_of_columns)

    # Connect every cell to its right neighbour
    #.reshape(-1) flattens the 2D grid into a 1D array
    horizontal_edges = torch.stack(
        [node_grid[:, :-1].reshape(-1), node_grid[:, 1:].reshape(-1)]
    )
    # Connect every cell to the cell below it
    vertical_edges = torch.stack(
        [node_grid[:-1, :].reshape(-1), node_grid[1:, :].reshape(-1)]
    )

    #dim=1 means we concatenate the edges along the second dimension (columns), so we get a 2D tensor where each column is an edge
    forward_edges = torch.cat([horizontal_edges, vertical_edges], dim=1)
    # Add the reverse direction too, so messages can flow both ways
    #flip(0) flips the first dimension (rows) of the tensor, so we get the reverse edges
    edge_index = torch.cat([forward_edges, forward_edges.flip(0)], dim=1)

    # Work out the real-world (x, y) coordinate of the centre of every cell
    x_coordinates = (torch.arange(number_of_columns, dtype=torch.float32) + 0.5) * dx
    y_coordinates = (torch.arange(number_of_rows, dtype=torch.float32) + 0.5) * dy
    #indexing="ij" means we want the first dimension of the meshgrid to correspond to the y-coordinates and the second dimension to correspond to the x-coordinates
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    positions = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

    # The physical length of every edge (used by the GNN as an edge feature)
    edge_distance = torch.linalg.vector_norm(
        positions[edge_index[1]] - positions[edge_index[0]], dim=1, keepdim=True
    )

    return edge_index, edge_distance, positions


def mean_pool_2x2(array_2d):
    """Shrink a 2D array to half its size by averaging every 2x2 block."""
    number_of_rows, number_of_columns = array_2d.shape
    # // is integer division
    return (
        array_2d.reshape(number_of_rows // 2, 2, number_of_columns // 2, 2)
        #axis=(1, 3) means we average over the second and fourth dimensions (the 2x2 blocks)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def build_pooling_map(child_columns, child_rows):
    """
    Map each child node to its parent node after 2x2 pooling.
    Every 2x2 block of 4 children shares one parent ID.
    """
    parent_columns = child_columns // 2
    parent_ids = []

    for row in range(child_rows):
        for col in range(child_columns):
            parent_row = row // 2
            parent_col = col // 2
            parent_id = parent_row * parent_columns + parent_col
            parent_ids.append(parent_id)

    return torch.tensor(parent_ids, dtype=torch.long)


# ============================================================
# 3. Lining up rainfall with the water-depth timestamps
# ============================================================

def align_rainfall_to_model_time(rainfall_time_seconds, rain_mmhr, model_time_seconds):
    """
    Turn irregular rainfall readings into one averaged value per model
    timestep, based on how much of each rainfall interval overlaps it.
    """
    rainfall_end_seconds = np.append(rainfall_time_seconds[1:], model_time_seconds[-1])
    aligned_rainfall = np.zeros(len(model_time_seconds), dtype=np.float32)

    for step in range(len(model_time_seconds) - 1):
        step_start = model_time_seconds[step]
        step_end = model_time_seconds[step + 1]

        total_rain = 0.0
        # zip() bundles the three arrays together so we can iterate over them in parallel
        for rain_start, rain_end, rain_rate in zip(rainfall_time_seconds, rainfall_end_seconds, rain_mmhr):
            overlap_start = max(step_start, rain_start)
            overlap_end = min(step_end, rain_end)
            overlap = max(overlap_end - overlap_start, 0.0)
            total_rain += rain_rate * overlap

        aligned_rainfall[step] = total_rain / (step_end - step_start)

    return aligned_rainfall  # last frame has no "next step", so it stays 0


# ============================================================
# 4. Main function: turn one .npz file into one complete graph
# ============================================================

def convert_one_npz_to_graph(npz_path):
    """
    Read one saved SFINCS simulation (.npz) and build its multiscale graph.

    This is the ONE function you call per simulation. It is safe to call
    exactly once and reuse the result for every future experiment (different
    history length, different prediction horizon, different model
    architecture) - none of those choices affect this step.
    """

    npz_path = Path(npz_path)

    # --- Load everything saved by save_training_npz_type_A() ---
    with np.load(npz_path, allow_pickle=True) as npz:
        dem_2d = npz["dem"].astype(np.float32)
        water_depth_3d = npz["water_depth"].astype(np.float32)  # shape: [time, row, column]
        rainfall_time_seconds = npz["rainfall_time_seconds"].astype(np.float64)
        model_time_seconds = npz["model_time_seconds"].astype(np.float64)
        rain_mmhr = npz["rain_mmhr"].astype(np.float32)
        metadata = json.loads(str(npz["metadata"].item()))

    fine_rows, fine_columns = dem_2d.shape
    dx = float(metadata["dx"])
    dy = float(metadata["dy"])

    # Reshape water depth from [time, row, col] to [node, time] -
    # one row per grid cell, one column per timestep. This is the layout
    # the GNN and the temporal-windowing step both expect.
    water_depth = water_depth_3d.reshape(water_depth_3d.shape[0], -1).T

    # --- Build the medium and coarse DEMs by repeated 2x2 averaging ---
    medium_dem_2d = mean_pool_2x2(dem_2d)
    coarse_dem_2d = mean_pool_2x2(medium_dem_2d)
    medium_rows, medium_columns = medium_dem_2d.shape
    coarse_rows, coarse_columns = coarse_dem_2d.shape

    # --- Build the three grid graphs (edges + distances + coordinates) ---
    fine_graph = build_grid_graph(fine_columns, fine_rows, dx, dy)
    medium_graph = build_grid_graph(medium_columns, medium_rows, 2.0 * dx, 2.0 * dy)
    coarse_graph = build_grid_graph(coarse_columns, coarse_rows, 4.0 * dx, 4.0 * dy)

    # --- Rainfall: align to model timesteps, then repeat over every node ---
    # (Type A rainfall is spatially uniform, so every cell sees the same value)
    aligned_rainfall = align_rainfall_to_model_time(rainfall_time_seconds, rain_mmhr, model_time_seconds)
    rainfall = np.repeat(aligned_rainfall[None, :], fine_rows * fine_columns, axis=0)

    # --- Pack everything into one MultiScaleData object ---
    graph = MultiScaleData()

    graph.fine_z = torch.from_numpy(dem_2d.reshape(-1, 1))
    graph.medium_z = torch.from_numpy(medium_dem_2d.reshape(-1, 1))
    graph.coarse_z = torch.from_numpy(coarse_dem_2d.reshape(-1, 1))

    graph.fine_edge_index, graph.fine_edge_distance, graph.fine_pos = fine_graph
    graph.medium_edge_index, graph.medium_edge_distance, graph.medium_pos = medium_graph
    graph.coarse_edge_index, graph.coarse_edge_distance, graph.coarse_pos = coarse_graph

    graph.fine_to_medium = build_pooling_map(fine_columns, fine_rows)
    graph.medium_to_coarse = build_pooling_map(medium_columns, medium_rows)

    # Full, un-windowed time series - this is what makes reuse possible
    graph.water_depth = torch.from_numpy(water_depth)                       # [fine node, time]
    graph.rainfall = torch.from_numpy(rainfall)                             # [fine node, time]
    graph.model_time_seconds = torch.from_numpy(model_time_seconds.astype(np.float32))

    graph.num_fine_nodes = fine_rows * fine_columns
    graph.num_medium_nodes = medium_rows * medium_columns
    graph.num_coarse_nodes = coarse_rows * coarse_columns
    graph.num_nodes = graph.num_fine_nodes  # PyG uses this to build batches

    graph.fine_nx, graph.fine_ny = fine_columns, fine_rows
    graph.medium_nx, graph.medium_ny = medium_columns, medium_rows
    graph.coarse_nx, graph.coarse_ny = coarse_columns, coarse_rows
    graph.dx, graph.dy = dx, dy

    # Bookkeeping - useful for debugging and for labeling plots later
    graph.case_name = metadata["case_name"]
    graph.dem_name = metadata["dem_name"]
    graph.rainfall_name = metadata["rainfall_name"]
    graph.runtime_seconds = float(metadata.get("runtime_seconds", 0.0))

    return graph
