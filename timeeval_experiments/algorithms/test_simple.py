from typing import Any, Dict
import numpy as np
from pathlib import Path

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.data_types import AlgorithmParameter
from timeeval.params import ParameterConfig


_test_simple_parameters: Dict[str, Dict[str, Any]] = {
    "method": {
        "defaultValue": "random",
        "description": "Method for generating anomaly scores: 'random', 'mean_deviation', 'high_values'",
        "name": "method",
        "type": "enum[random,mean_deviation,high_values]"
    },
    "target_channels": {
        "defaultValue": None,
        "description": "channels to detect anomalies in. If None, all channels in data are used",
        "name": "target_channels",
        "type": "List[String]"
    },
}


def _test_simple_function(data: AlgorithmParameter, params: Dict[str, Any]) -> np.ndarray:
    """
    Simple test algorithm that generates anomaly scores using different methods
    """
    print(f"[TEST_SIMPLE] Input data type: {type(data)}")
    print(f"[TEST_SIMPLE] Parameters: {params}")
    
    if isinstance(data, Path):
        print(f"[TEST_SIMPLE] Reading data from file: {data}")
        # Read data from CSV file (skip header, assume first column is timestamp)
        data_array = np.genfromtxt(data, delimiter=',', skip_header=1)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        # Remove timestamp column if present
        if data_array.shape[1] > 1:
            data_array = data_array[:, 1:]  # Skip first column (timestamp)
    else:
        data_array = np.array(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
    
    print(f"[TEST_SIMPLE] Data shape: {data_array.shape}")
    print(f"[TEST_SIMPLE] Data sample (first 5 rows): {data_array[:5]}")
    
    # Get parameters
    method = params.get("method", "random")
    target_channels = params.get("target_channels", None)
    
    print(f"[TEST_SIMPLE] Method: {method}")
    print(f"[TEST_SIMPLE] Target channels: {target_channels}")
    
    # Generate anomaly scores based on method
    if method == "random":
        # Random scores between 0 and 1
        scores = np.random.rand(data_array.shape[0])
        print("[TEST_SIMPLE] Generated random scores")
        
    elif method == "mean_deviation":
        # Deviation from mean across all channels
        if data_array.shape[1] > 1:
            # Multivariate: compute distance from mean
            mean_vals = np.mean(data_array, axis=0)
            scores = np.linalg.norm(data_array - mean_vals, axis=1)
        else:
            # Univariate: absolute deviation from mean
            mean_val = np.mean(data_array)
            scores = np.abs(data_array.flatten() - mean_val)
        
        # Normalize to [0, 1]
        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        print("[TEST_SIMPLE] Generated mean deviation scores")
        
    elif method == "high_values":
        # Higher values get higher anomaly scores
        if data_array.shape[1] > 1:
            # Multivariate: use max value across channels
            scores = np.max(data_array, axis=1)
        else:
            # Univariate: use values directly
            scores = data_array.flatten()
        
        # Normalize to [0, 1]
        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        print("[TEST_SIMPLE] Generated high values scores")
        
    else:
        # Default to random
        scores = np.random.rand(data_array.shape[0])
        print("[TEST_SIMPLE] Using default random scores")
    
    print(f"[TEST_SIMPLE] Generated scores shape: {scores.shape}")
    print(f"[TEST_SIMPLE] Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"[TEST_SIMPLE] Score sample (first 10): {scores[:10]}")
    
    return scores


def test_simple(params: ParameterConfig = None) -> Algorithm:
    """
    Create a simple test algorithm instance using FunctionAdapter
    """
    return Algorithm(
        name="TestSimple",
        main=FunctionAdapter(_test_simple_function),
        preprocess=None,
        postprocess=None,
        param_schema=_test_simple_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,  # We'll read from file to test file handling
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE
    )