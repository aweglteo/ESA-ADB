#!/usr/bin/env python3
"""
Test script for FunctionAdapter-based algorithm
"""
import os
from pathlib import Path

from timeeval import TimeEval, DatasetManager
from timeeval.metrics import DefaultMetrics
from timeeval.params import FixedParameters
from durations import Duration

from timeeval_experiments.algorithms import test_simple

# Set up paths
current_dir = os.path.dirname(os.path.realpath(__file__))
data_processed_folder = os.path.abspath(os.path.join(current_dir, "data", "preprocessed"))

print(f"Looking for preprocessed data in: {data_processed_folder}")

# Initialize dataset manager
dm = DatasetManager(data_processed_folder)

# Select a small subset of ESA-Mission1 data for testing
collection = "ESA-Mission1"
try:
    datasets = dm.select(collection=collection, dataset="3_months")  # Use smallest dataset
except Exception as e:
    print(f"Error selecting dataset: {e}")
    print("Available datasets:")
    print(dm.df.index.tolist())
    # Try to select manually
    datasets = dm.select(collection=collection)

print(f"Selected datasets: {datasets}")

# Define test algorithms with different parameters
algorithms = [
    test_simple(params=FixedParameters({"method": "random"})),
    test_simple(params=FixedParameters({"method": "mean_deviation"})),
    test_simple(params=FixedParameters({"method": "high_values"})),
]

# Use basic metrics for testing
metrics = [
    DefaultMetrics.ROC_AUC,
    DefaultMetrics.PR_AUC,
    DefaultMetrics.AVERAGE_PRECISION,
]

print(f"Using metrics: {[m.name for m in metrics]}")

# Set up TimeEval with shorter timeout for testing
timeeval = TimeEval(
    dm, 
    datasets, 
    algorithms,
    metrics=metrics,
    resource_constraints=None,  # No resource constraints for function adapter
    skip_invalid_combinations=True,
    distributed=False,
    force_training_type_match=False,
    force_dimensionality_match=False,
)

print("Starting TimeEval execution...")
timeeval.run()

print("Getting results...")
results = timeeval.get_results(aggregated=False)
print(f"Results shape: {results.shape}")
print(f"Results columns: {results.columns.tolist()}")

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(results[['algorithm', 'dataset', 'ROC_AUC', 'PR_AUC', 'status']])

print("\n" + "="*50)
print("RESULTS DETAILS")
print("="*50)
print(results)

print(f"\nResults saved to: {timeeval.results_path}")