"""
A complete graph (Stage 1) holds the FULL water-depth/rainfall time series
for one simulation. A GNN needs short snippets instead, so this file slides
through each simulation one timestep at a time, cutting overlapping
snippets:

    INPUT  : water depth at the last `previous_steps` steps + rainfall now
    TARGET : water depth at the next `prediction_horizon` steps

Example (previous_steps=3, prediction_horizon=6):
    input  = [h(t-2), h(t-1), h(t), rain(t)]
    target = [h(t+1), ..., h(t+6)]

WHY THIS MATTERS FOR SWEEPS: `previous_steps`/`prediction_horizon` are
arguments, not hardcoded, so changing them just means re-running this file's
functions - no need to rebuild complete graphs from .npz files. This step
is cheap (slicing); Stage 1 (grids/pooling maps) is not.

Pattern: build complete_graphs ONCE outside the sweep loop, then call
build_temporal_dataset() with each run's chosen values.
"""

import torch

from scripts.sfincs_dataset_conversion.pytorch_geometric import MultiScaleData


# Sensible defaults - override these per experiment, no need to edit this file
DEFAULT_PREVIOUS_STEPS = 3
DEFAULT_PREDICTION_HORIZON = 6


def create_one_temporal_sample(graph, time_index, previous_steps, prediction_horizon):
    """
    Cut ONE training snippet out of a complete simulation graph, starting at
    the given time_index.

    graph              - a complete graph from convert_one_npz_to_graph()
    time_index         - which timestep to treat as "now" (t)
    previous_steps     - how many past water-depth frames to use as input
                          (e.g. 3 means we look at t-2, t-1, and t)
    prediction_horizon - how many future frames the model must predict
    """

    sample = MultiScaleData()

    # The static parts (elevation, grid structure) never change over time,
    # so we just copy them straight from the complete graph.
    sample.fine_z = graph.fine_z
    sample.medium_z = graph.medium_z
    sample.coarse_z = graph.coarse_z

    sample.fine_edge_index = graph.fine_edge_index
    sample.medium_edge_index = graph.medium_edge_index
    sample.coarse_edge_index = graph.coarse_edge_index

    sample.fine_edge_distance = graph.fine_edge_distance
    sample.medium_edge_distance = graph.medium_edge_distance
    sample.coarse_edge_distance = graph.coarse_edge_distance

    sample.fine_to_medium = graph.fine_to_medium
    sample.medium_to_coarse = graph.medium_to_coarse

    sample.num_fine_nodes = graph.num_fine_nodes
    sample.num_medium_nodes = graph.num_medium_nodes
    sample.num_coarse_nodes = graph.num_coarse_nodes
    sample.num_nodes = graph.num_fine_nodes

    # --- Build the model's INPUT ---
    # Grab the last `previous_steps` water-depth frames, ending at time_index (inclusive)
    history_start = time_index - previous_steps + 1
    history_depths = graph.water_depth[:, history_start:time_index + 1]   # [nodes, previous_steps]
    rainfall_now = graph.rainfall[:, time_index:time_index + 1]           # [nodes, 1]
    sample.fine_dynamic_x = torch.cat([history_depths, rainfall_now], dim=1)

    # --- Build the model's TARGET ---
    # The water depth at the next `prediction_horizon` steps: t+1 ... t+prediction_horizon
    sample.y = graph.water_depth[:, time_index + 1: time_index + 1 + prediction_horizon]

    # Rainfall over that same future window, in case the model needs to know it
    sample.rainfall_sequence = graph.rainfall[:, time_index: time_index + prediction_horizon]

    # How many real-world seconds pass between each predicted step
    """
    current_times and target_times are the same timestamp list shifted by one
    index, so subtracting them gives the real time gap before each predicted
    step. delta_time_seconds stores these gaps so the model can correctly scale
    predicted change even when SFINCS's save intervals aren't uniform.
    """
    current_times = graph.model_time_seconds[time_index: time_index + prediction_horizon]
    target_times = graph.model_time_seconds[time_index + 1: time_index + 1 + prediction_horizon]
    #.unsqueeze(0) adds a new dimension at the front, so we get a 1D tensor of shape [1, prediction_horizon] instead of [prediction_horizon]
    sample.delta_time_seconds = (target_times - current_times).unsqueeze(0)

    # Bookkeeping, useful for debugging and for the sweep config you used
    sample.time_index = time_index
    sample.previous_steps = previous_steps
    sample.prediction_horizon = prediction_horizon
    sample.case_name = graph.case_name

    return sample


def convert_one_graph_to_samples(graph, previous_steps=DEFAULT_PREVIOUS_STEPS,
                                  prediction_horizon=DEFAULT_PREDICTION_HORIZON):
    """
    Slide a window across ONE complete simulation and return every valid
    training snippet it can produce.
    """

    samples = []
    #.shape[1] is the number of columns (timesteps) in the water_depth array
    number_of_time_steps = graph.water_depth.shape[1]

    # The earliest valid "now" is once we have enough history (previous_steps - 1
    # steps before it). The latest valid "now" must leave room for the full
    # prediction horizon afterwards.
    first_valid_index = previous_steps - 1
    last_valid_index = number_of_time_steps - prediction_horizon

    for time_index in range(first_valid_index, last_valid_index):
        samples.append(create_one_temporal_sample(graph, time_index, previous_steps, prediction_horizon))

    return samples


def build_temporal_dataset(graphs, split_name,
                            previous_steps=DEFAULT_PREVIOUS_STEPS,
                            prediction_horizon=DEFAULT_PREDICTION_HORIZON):
    """
    Convert a LIST of complete simulation graphs (e.g. all your training
    simulations) into one big list of training snippets.

    Call this again with different previous_steps / prediction_horizon for
    every hyperparameter sweep run - it's fast, and you don't need to touch
    your .npz files or rebuild `graphs` to do it.
    """

    all_samples = []
    for graph_number, graph in enumerate(graphs, start=1):
        print(f"{split_name}: graph {graph_number}/{len(graphs)} - {graph.case_name} "
              f"(previous_steps={previous_steps}, prediction_horizon={prediction_horizon})")
        # .extend() adds all the elements of the list returned by convert_one_graph_to_samples() to all_samples
        all_samples.extend(convert_one_graph_to_samples(graph, previous_steps, prediction_horizon))

    print(f"{split_name}: total training snippets = {len(all_samples)}")
    return all_samples
