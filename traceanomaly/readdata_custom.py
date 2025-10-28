"""
Custom data reading functions for processed trace data.

This module provides functions to read and process the custom trace data
that has been converted from JSON format to the TraceAnomaly format.
"""

import numpy as np
import random
import json
import os


def read_raw_vector_streaming(input_file, vc=None, max_samples=None, sample_rate=1.0):
    """
    Memory-efficient streaming read of raw vector data.
    
    Args:
        input_file: Path to the input file
        vc: Valid columns (optional)
        max_samples: Maximum number of samples to read
        sample_rate: Fraction of data to sample (1.0 for all)
        
    Returns:
        Tuple of (flows, vectors, valid_columns)
    """
    flows = list()
    vectors = list()
    
    # First pass: count total lines for sampling
    total_lines = 0
    with open(input_file, 'r') as fin:
        for line in fin:
            if line.strip():
                total_lines += 1
    
    # Calculate how many samples to actually read
    if max_samples is not None:
        target_samples = min(max_samples, int(total_lines * sample_rate))
    else:
        target_samples = int(total_lines * sample_rate)
    
    print(f"File has {total_lines} lines, sampling {target_samples} samples")
    
    # Second pass: read only the samples we need
    import random
    if sample_rate < 1.0:
        # Random sampling - select line indices to read
        selected_indices = set(random.sample(range(total_lines), target_samples))
    else:
        # Read all lines
        selected_indices = set(range(total_lines))
    
    line_idx = 0
    with open(input_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
                
            if line_idx in selected_indices:
                flows.append(line.split(':')[0])
                vectors.append([float(x) for x in line.split(':')[1].split(',')])
            
            line_idx += 1
            
            # Stop if we have enough samples
            if len(vectors) >= target_samples:
                break
    
    vectors = np.array(vectors)
    n = len(vectors)
    m = len(vectors[0])

    if vc is None:
        valid_column = list()
        for i in range(0, m):
            flag = False
            for j in range(0, n):
                if vectors[j, i] > 0:
                    flag = True
                    break
            if flag:
                valid_column.append(i)
    else:
        valid_column = vc

    vectors = vectors[:, valid_column]
    return flows, vectors, valid_column


def read_raw_vector(input_file, vc=None, shuffle=True, sample=False):
    """
    Read raw vector data from processed trace files.
    
    Args:
        input_file: Path to the input file
        vc: Valid columns (optional)
        shuffle: Whether to shuffle the data
        sample: Whether to sample a subset of data
        
    Returns:
        Tuple of (flows, vectors, valid_columns)
    """
    flows = list()
    vectors = list()
    
    # Read file line by line to avoid loading entire file into memory
    with open(input_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            flows.append(line.split(':')[0])
            vectors.append([float(x) for x in line.split(':')[1].split(',')])
    
    # Convert to numpy array immediately to avoid memory duplication
    vectors = np.array(vectors, dtype=np.float32)
    
    if shuffle is True:
        arr_index = np.arange(len(vectors))
        np.random.shuffle(arr_index)
        vectors = vectors[arr_index]
        flows = [flows[i] for i in arr_index]

    if sample is True:
        sample_size = min(50000, len(vectors))
        sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[sample_indices]
        flows = [flows[i] for i in sample_indices]

    n = len(vectors)
    m = len(vectors[0])

    if vc is None:
        valid_column = list()

        for i in range(0, m):
            flag = False
            for j in range(0, n):
                if vectors[j, i] > 0:
                    flag = True
                    break
            if flag:
                valid_column.append(i)
    else:
        valid_column = vc

    vectors = vectors[:, valid_column]
    return flows, vectors, valid_column


def get_mean_std(matrix):
    """Calculate mean and standard deviation for normalization."""
    mean = []
    std = []
    for item in np.transpose(matrix):
        mean.append(np.mean(item[item>0.00001]))
        std.append(max(1, np.std(item[item>0.00001])))
    
    return mean, std


def normalization(matrix, mean, std):
    """Apply z-score normalization with zero replacement."""
    n_mat = np.array(matrix, dtype=np.float32)
    n_mat = np.where(n_mat<0.00001, -1, (n_mat - mean) / std)
    return n_mat


def get_data_vae(train_file, normal_file, abnormal_file, valid_columns=None):
    """
    Get data for VAE training and testing.
    
    Args:
        train_file: Path to training data file
        normal_file: Path to normal test data file
        abnormal_file: Path to abnormal test data file
        valid_columns: Pre-computed valid columns (optional)
        
    Returns:
        Tuple of ((train_x, train_y), (test_x, test_y), test_flow)
    """
    if valid_columns is None:
        _, train_raw, valid_columns = read_raw_vector(train_file)
    else:
        _, train_raw, _ = read_raw_vector(train_file, valid_columns)
    
    flows1, normal_raw, _ = read_raw_vector(normal_file, valid_columns, shuffle=False)
    flows2, abnormal_raw, _ = read_raw_vector(abnormal_file, valid_columns, shuffle=False)

    train_mean, train_std = get_mean_std(train_raw)
    train_x = normalization(train_raw, train_mean, train_std)
    normal_x = normalization(normal_raw, train_mean, train_std)
    abnormal_x = normalization(abnormal_raw, train_mean, train_std)
    
    print('abnormal')
    for i in range(min(30, len(abnormal_x))):
        print(list(abnormal_x[i]))

    train_y = np.zeros(len(train_x), dtype=np.int32)
    normal_y = np.zeros(len(normal_x), dtype=np.int32)
    abnormal_y = np.ones(len(abnormal_x), dtype=np.int32)

    test_x = np.concatenate([normal_x, abnormal_x])
    test_y = np.concatenate([normal_y, abnormal_y])
    test_flow = flows1 + flows2

    return (train_x, train_y), (test_x, test_y), test_flow


def get_data_vae_custom(data_dir):
    """
    Get data for VAE training and testing from custom processed data.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Tuple of ((train_x, train_y), (test_x, test_y), test_flow)
    """
    train_file = os.path.join(data_dir, 'train')
    normal_file = os.path.join(data_dir, 'test_normal')
    abnormal_file = os.path.join(data_dir, 'test_abnormal')
    
    # Check if files exist
    for file_path in [train_file, normal_file, abnormal_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load metadata to get the original feature mapping
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        valid_columns = metadata.get('valid_columns', None)
        print(f"Using pre-computed valid columns from metadata: {len(valid_columns) if valid_columns else 'None'}")
    else:
        valid_columns = None
        print("No metadata found, will compute valid columns from training data")
    
    return get_data_vae(train_file, normal_file, abnormal_file, valid_columns)

def get_data_vae_test_only(data_dir, train_data_dir=None):
    """
    Get data for VAE testing from test-only processed data (no training data).
    
    Args:
        data_dir: Directory containing processed test data files
        train_data_dir: Directory containing training data for normalization (optional)
        
    Returns:
        Tuple of ((None, None), (test_x, test_y), test_flow)
    """
    normal_file = os.path.join(data_dir, 'test_normal')
    abnormal_file = os.path.join(data_dir, 'test_abnormal')
    
    # Check if test files exist
    for file_path in [normal_file, abnormal_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load metadata to get the original feature mapping
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        valid_columns = metadata.get('valid_columns', None)
        print(f"Using pre-computed valid columns from metadata: {len(valid_columns) if valid_columns else 'None'}")
    else:
        valid_columns = None
        print("No metadata found, will compute valid columns from test data")
    
    # Load test data
    if valid_columns is None:
        flows1, normal_raw, valid_columns = read_raw_vector(normal_file, shuffle=False)
    else:
        flows1, normal_raw, _ = read_raw_vector(normal_file, valid_columns, shuffle=False)
    flows2, abnormal_raw, _ = read_raw_vector(abnormal_file, valid_columns, shuffle=False)
    
    # Try to use training data normalization if available
    if train_data_dir:
        train_file = os.path.join(train_data_dir, 'train')
        if os.path.exists(train_file):
            try:
                # Load training data for normalization
                _, train_raw, _ = read_raw_vector(train_file, valid_columns, shuffle=False)
                train_mean, train_std = get_mean_std(train_raw)
                print("Using training data normalization parameters")
            except Exception as e:
                print(f"Warning: Could not load training normalization: {e}")
                # Fall back to test data normalization
                all_raw = np.concatenate([normal_raw, abnormal_raw])
                train_mean, train_std = get_mean_std(all_raw)
        else:
            # Fall back to test data normalization
            all_raw = np.concatenate([normal_raw, abnormal_raw])
            train_mean, train_std = get_mean_std(all_raw)
    else:
        # Use test data for normalization
        all_raw = np.concatenate([normal_raw, abnormal_raw])
        train_mean, train_std = get_mean_std(all_raw)
    
    normal_x = normalization(normal_raw, train_mean, train_std)
    abnormal_x = normalization(abnormal_raw, train_mean, train_std)
    
    normal_y = np.zeros(len(normal_x), dtype=np.int32)
    abnormal_y = np.ones(len(abnormal_x), dtype=np.int32)
    
    test_x = np.concatenate([normal_x, abnormal_x])
    test_y = np.concatenate([normal_y, abnormal_y])
    test_flow = flows1 + flows2
    
    return (None, None), (test_x, test_y), test_flow

def get_data_vae_unsupervised(data_dir, max_samples=None, sample_rate=1.0):
    """
    Get data for unsupervised VAE training from custom processed data.
    
    Args:
        data_dir: Directory containing processed data files
        max_samples: Maximum number of samples to load (None for all)
        sample_rate: Fraction of data to sample (1.0 for all, 0.1 for 10%)
        
    Returns:
        Tuple of (train_x, train_flow_ids)
    """
    train_file = os.path.join(data_dir, 'train')
    
    # Check if file exists
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Required file not found: {train_file}")
    
    # Load metadata to get the original feature mapping
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        valid_columns = metadata.get('valid_columns', None)
        print(f"Using pre-computed valid columns from metadata: {len(valid_columns) if valid_columns else 'None'}")
    else:
        valid_columns = None
        print("No metadata found, will compute valid columns from sampled data")
    
    # Use streaming approach for memory efficiency
    print(f"Using streaming data loading: max_samples={max_samples}, sample_rate={sample_rate}")
    flows, vectors, computed_valid_columns = read_raw_vector_streaming(train_file, vc=valid_columns, max_samples=max_samples, sample_rate=sample_rate)
    
    # Apply normalization for consistency with supervised training
    print("Applying normalization...")
    mean, std = get_mean_std(vectors)
    vectors = normalization(vectors, mean, std)
    print(f"Normalization complete. Data shape: {vectors.shape}")
    
    return vectors, flows


def load_metadata(data_dir):
    """
    Load metadata from the processed data directory.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Dictionary containing metadata
    """
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def load_feature_index_map(data_dir):
    """
    Load feature index map from the processed data directory.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Dictionary mapping feature patterns to indices
    """
    import pickle
    
    idx_file = os.path.join(data_dir, 'idx.pkl')
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"Feature index file not found: {idx_file}")
    
    with open(idx_file, 'rb') as f:
        feature_index_map = pickle.load(f)
    
    return feature_index_map


def get_z_dim(x_dim):
    """Calculate appropriate z dimension for VAE based on input dimension."""
    tmp = x_dim
    z_dim = 5
    while tmp > 20:
        z_dim *= 2
        tmp = tmp // 20
    return z_dim


def print_data_summary(data_dir):
    """
    Print a summary of the processed data.
    
    Args:
        data_dir: Directory containing processed data files
    """
    try:
        metadata = load_metadata(data_dir)
        feature_index_map = load_feature_index_map(data_dir)
        
        print("=== Processed Data Summary ===")
        print(f"Total features: {metadata['num_features']}")
        print(f"Valid features: {len(metadata['valid_columns'])}")
        print(f"Training samples: {metadata['train_samples']}")
        print(f"Test samples: {metadata['test_samples']}")
        print(f"  - Normal: {metadata['normal_test_samples']}")
        print(f"  - Anomalous: {metadata['anomalous_test_samples']}")
        print(f"Feature index map size: {len(feature_index_map)}")
        
        # Show some example features
        print("\nExample features:")
        for i, (pattern, idx) in enumerate(list(feature_index_map.items())[:10]):
            print(f"  {idx}: {pattern}")
        if len(feature_index_map) > 10:
            print(f"  ... and {len(feature_index_map) - 10} more")
            
    except Exception as e:
        print(f"Error loading data summary: {e}")


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print_data_summary(data_dir)
    else:
        print("Usage: python readdata_custom.py <data_directory>")
