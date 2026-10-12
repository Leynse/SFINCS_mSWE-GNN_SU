"""
STAGE 3 of 3: Load everything and hand PyTorch Geometric DataLoaders to your
training script.

This file ties Stage 1 (pytorch_geometric.py) and Stage 2 (temporal_learning.py)
together, and is deliberately split into an EXPENSIVE step and a CHEAP step:

    EXPENSIVE (do once):
        load_complete_graphs()     - reads every .npz file and builds its
                                      multiscale graph. This is slow because
                                      it does grid construction and pooling.

    CHEAP (do as many times as you like):
        build_temporal_datasets()  - slices those already-built graphs into
                                      training snippets. This is just tensor
                                      slicing, so it's fast to redo with a
                                      different previous_steps / prediction_horizon.

HOW TO USE THIS FOR A WEIGHTS & BIASES SWEEP
----------------------------------------------
    # Run this ONCE, before your sweep starts:
    complete_graphs = load_all_complete_graphs(PROJECT_ROOT, "dataset_split.json")

    # Inside each sweep run (this part is fast, safe to repeat every trial):
    datasets = build_temporal_datasets(
        complete_graphs,
        previous_steps=wandb.config.previous_steps,       # e.g. 2, 3, or 4
        prediction_horizon=wandb.config.prediction_horizon,
    )
    loaders = create_data_loaders(datasets, batch_size=wandb.config.batch_size)

USING THE SAME DATA FOR GRAPHSAGE COMPARISON
-----------------------------------------------
No changes needed. Every sample already has both the multiscale fields
(fine_z, medium_z, coarse_z, ...) and the fine-only fields GraphSAGE needs
(fine_edge_index, fine_z, fine_dynamic_x, y). GraphSAGE's forward() just
reads the fine_* fields and ignores the rest - the dataset stays the same
regardless of model.
"""

import json
from pathlib import Path

from torch_geometric.loader import DataLoader

from scripts.sfincs_dataset_conversion.pytorch_geometric import convert_one_npz_to_graph
from scripts.sfincs_dataset_conversion.temporal_learning import (
    build_temporal_dataset,
    DEFAULT_PREVIOUS_STEPS,
    DEFAULT_PREDICTION_HORIZON,
)


# ============================================================
# EXPENSIVE STEP - run this once, reuse the result everywhere
# ============================================================

def read_dataset_split_file(dataset_split_path):
    """Read the JSON file listing which .npz paths belong to train/validation/test."""
    with Path(dataset_split_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_complete_graphs(project_root, npz_paths, split_name):
    """Convert a list of .npz simulation files into complete multiscale graphs."""

    project_root = Path(project_root)
    graphs = []

    for file_number, npz_path in enumerate(npz_paths, start=1):
        npz_path = Path(npz_path)
        if not npz_path.is_absolute():
            npz_path = project_root / npz_path

        print(f"{split_name}: loading file {file_number}/{len(npz_paths)} - {npz_path.name}")
        graphs.append(convert_one_npz_to_graph(npz_path))

    return graphs


def load_all_complete_graphs(project_root, dataset_split_path):
    """
    Build every complete graph for train, validation, and test.
    This is the slow step - run it ONCE per dataset and keep the result in
    memory (or save it, e.g. with torch.save) for reuse across experiments.
    """

    dataset_split = read_dataset_split_file(dataset_split_path)

    return {
        "train": load_complete_graphs(project_root, dataset_split["train"], "train"),
        "validation": load_complete_graphs(project_root, dataset_split["validation"], "validation"),
        "test": load_complete_graphs(project_root, dataset_split["test"], "test"),
    }


# ============================================================
# CHEAP STEP - safe to re-run for every hyperparameter sweep trial
# ============================================================

def build_temporal_datasets(complete_graphs,
                             previous_steps=DEFAULT_PREVIOUS_STEPS,
                             prediction_horizon=DEFAULT_PREDICTION_HORIZON):
    """
    Slice already-built complete graphs into training snippets, using
    whichever previous_steps / prediction_horizon this experiment needs.

    complete_graphs - the dictionary returned by load_all_complete_graphs()
    """

    training_samples = build_temporal_dataset(
        complete_graphs["train"], "train", previous_steps, prediction_horizon
    )
    validation_samples = build_temporal_dataset(
        complete_graphs["validation"], "validation", previous_steps, prediction_horizon
    )

    return {
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        # Kept as full, un-windowed graphs for rollout-style evaluation
        # (predicting many steps ahead in one continuous run, not fixed snippets)
        "validation_graphs": complete_graphs["validation"],
        "testing_graphs": complete_graphs["test"],
    }


# ============================================================
# Wrap everything into PyTorch Geometric DataLoaders
# ============================================================

def create_data_loaders(datasets, batch_size, number_of_workers=0):
    """Package the training/validation snippets into DataLoaders ready for a training loop."""

    training_loader = DataLoader(
        datasets["training_samples"], batch_size=batch_size, shuffle=True, num_workers=number_of_workers
    )
    validation_loader = DataLoader(
        datasets["validation_samples"], batch_size=batch_size, shuffle=False, num_workers=number_of_workers
    )

    return {"training": training_loader, "validation": validation_loader}
