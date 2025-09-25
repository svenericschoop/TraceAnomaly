#!/usr/bin/env python3
"""
Trace Data Processing Script for TraceAnomaly (Unsupervised Training)

This script converts JSON trace data (like 1809.jsonl.gz) into the format
expected by the TraceAnomaly system for unsupervised training.

The script performs the following steps:
1. Parse JSON trace data to extract service call patterns
2. Normalize span names using regex patterns and Drain3 template mining
3. Create feature vectors representing service call sequences
4. Apply the same normalization pipeline as the original train_ticket data
5. Save all processed data as training data for unsupervised learning

Usage:
    # Training mode (build templates)
    python process_trace_data.py --input 1809.jsonl.gz --output_dir processed_data --drain_state_file templates.pkl --training_mode
    
    # Inference mode (use existing templates)
    python process_trace_data.py --input test_data.jsonl.gz --output_dir processed_test --drain_state_file templates.pkl --inference_mode
"""

import json
import gzip
import argparse
import numpy as np
import pickle
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import random
from pathlib import Path
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence


class TraceProcessor:
    """Processes JSON trace data into TraceAnomaly format."""
    
    def __init__(self, min_feature_count: int = 2, drain_state_file: str = None, training_mode: bool = True):
        """
        Initialize the trace processor.
        
        Args:
            min_feature_count: Minimum number of occurrences for a feature to be included
            drain_state_file: Path to drain3 state file for template persistence
            training_mode: Boolean flag to control template building vs. inference
        """
        self.min_feature_count = min_feature_count
        self.feature_index_map = {}  # Maps service patterns to feature indices
        self.feature_counter = Counter()
        self.traces = []
        
        # Initialize Drain3 template miner
        if drain_state_file:
            persistence = FilePersistence(drain_state_file)
            self.template_miner = TemplateMiner(persistence)
        else:
            self.template_miner = TemplateMiner()
        
        self.training_mode = training_mode
        
        # Regex patterns for span name normalization
        self.regex_patterns = [
            (r'/_doc/\d+', '/_doc/'),
            (r'\d+', '<NUM>'),  # Replace numbers with <NUM>
            (r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>'),  # Replace UUIDs
            (r'/[a-fA-F0-9]{32,}', '/<HASH>'),  # Replace long hashes
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),  # Replace IP addresses
            (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),  # Replace dates
            (r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>'),  # Replace times
        ]
    
    def normalize_span_name(self, span_name: str) -> str:
        """
        Normalize span name using regex patterns.
        
        Args:
            span_name: Raw span name
            
        Returns:
            Normalized span name
        """
        normalized = span_name
        for pattern, replacement in self.regex_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized
    
    def parse_span_name_with_drain3(self, span_name: str) -> int:
        """
        Parse span name with Drain3 to get template cluster ID.
        
        Args:
            span_name: Normalized span name
            
        Returns:
            Template cluster ID
        """
        if self.training_mode:
            result = self.template_miner.add_log_message(span_name)
            return result["cluster_id"] if result else 0
        else:
            result = self.template_miner.match(span_name)
            return result.cluster_id if result else 0
        
    def extract_service_patterns(self, trace_data: Dict[str, Any]) -> List[str]:
        """
        Extract service call patterns from a trace using Drain3 template parsing.
        
        Args:
            trace_data: JSON trace data
            
        Returns:
            List of service call patterns
        """
        patterns = []
        
        def extract_from_span(span: Dict[str, Any], depth: int = 0) -> None:
            """Recursively extract patterns from spans."""
            if depth > 10:  # Prevent infinite recursion
                return
                
            # Extract service name and subtype
            name = span.get('name', 'unknown')
            subtype = span.get('subtype', 'unknown')
            
            # Normalize the span name using regex patterns
            normalized_name = self.normalize_span_name(name)
            
            # Parse with drain3 to get template ID
            template_id = self.parse_span_name_with_drain3(normalized_name)
            
            # Create pattern using template ID instead of raw name
            pattern = f"template_{template_id}#{subtype}"
            patterns.append(pattern)
            
            # Process children
            children = span.get('children', [])
            for child in children:
                extract_from_span(child, depth + 1)
        
        # Start from the root transaction
        if 'children' in trace_data:
            for child in trace_data['children']:
                extract_from_span(child)
        
        return patterns
    
    def create_sequence_patterns(self, patterns: List[str]) -> List[str]:
        """
        Create sequence patterns from individual service patterns.
        
        Args:
            patterns: List of individual service patterns
            
        Returns:
            List of sequence patterns
        """
        sequence_patterns = []
        
        # Create patterns of different lengths (1, 2, 3)
        for length in [1, 2, 3]:
            for i in range(len(patterns) - length + 1):
                sequence = '#'.join(patterns[i:i+length])
                sequence_patterns.append(sequence)
        
        return sequence_patterns
    
    def process_traces(self, input_file: str) -> None:
        """
        Process all traces from the input file.
        
        Args:
            input_file: Path to the input JSONL.gz file
        """
        print(f"Processing traces from {input_file}...")
        
        # First pass: collect all unique patterns and their frequencies
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} traces...")
                    
                try:
                    trace_data = json.loads(line.strip())
                    
                    # Extract service patterns
                    patterns = self.extract_service_patterns(trace_data)
                    sequence_patterns = self.create_sequence_patterns(patterns)
                    
                    # Count pattern frequencies
                    for pattern in sequence_patterns:
                        self.feature_counter[pattern] += 1
                    
                    # Store trace data for second pass
                    self.traces.append({
                        'id': trace_data.get('id', f'trace_{line_num}'),
                        'patterns': sequence_patterns,
                        'duration': trace_data.get('duration', 0),
                        'outcome': trace_data.get('outcome', 'unknown')
                    })
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed trace at line {line_num}: {e}")
                    continue
        
        print(f"Total traces processed: {len(self.traces)}")
        print(f"Total unique patterns found: {len(self.feature_counter)}")
        
        # Filter patterns by minimum frequency
        filtered_patterns = {
            pattern: count for pattern, count in self.feature_counter.items()
            if count >= self.min_feature_count
        }
        
        print(f"Patterns after filtering (min_count={self.min_feature_count}): {len(filtered_patterns)}")
        
        # Create feature index mapping
        sorted_patterns = sorted(filtered_patterns.keys())
        self.feature_index_map = {pattern: idx for idx, pattern in enumerate(sorted_patterns)}
        
        print(f"Created feature index map with {len(self.feature_index_map)} features")
    
    def create_feature_vectors(self) -> Tuple[List[str], np.ndarray]:
        """
        Create feature vectors from processed traces.
        
        Returns:
            Tuple of (flow_ids, feature_vectors)
        """
        print("Creating feature vectors...")
        
        num_features = len(self.feature_index_map)
        flow_ids = []
        feature_vectors = []
        
        for trace in self.traces:
            flow_id = trace['id']
            patterns = trace['patterns']
            
            # Create feature vector
            feature_vector = np.zeros(num_features, dtype=np.float32)
            
            # Count pattern occurrences
            pattern_counts = Counter(patterns)
            
            for pattern, count in pattern_counts.items():
                if pattern in self.feature_index_map:
                    idx = self.feature_index_map[pattern]
                    feature_vector[idx] = float(count)
            
            flow_ids.append(flow_id)
            feature_vectors.append(feature_vector)
        
        feature_vectors = np.array(feature_vectors)
        print(f"Created feature vectors: {feature_vectors.shape}")
        
        return flow_ids, feature_vectors
    
    def prepare_training_data(self, flow_ids: List[str], feature_vectors: np.ndarray) -> Dict[str, Any]:
        """
        Prepare all data for unsupervised training.
        
        Args:
            flow_ids: List of flow IDs
            feature_vectors: Feature vectors
            
        Returns:
            Dictionary containing training data
        """
        print("Preparing data for unsupervised training...")
        
        print(f"Training data: {len(feature_vectors)} samples")
        
        return {
            'flow_ids': flow_ids,
            'feature_vectors': feature_vectors,
            'feature_index_map': self.feature_index_map
        }


def apply_normalization_pipeline(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the same normalization pipeline as the original TraceAnomaly code.
    
    Args:
        data: Dictionary containing the training data
        
    Returns:
        Dictionary with normalized data
    """
    print("Applying normalization pipeline...")
    
    feature_vectors = data['feature_vectors']
    
    # Find valid columns (columns with at least one non-zero value)
    valid_columns = []
    for i in range(feature_vectors.shape[1]):
        if np.any(feature_vectors[:, i] > 0.00001):
            valid_columns.append(i)
    
    print(f"Valid columns: {len(valid_columns)} out of {feature_vectors.shape[1]}")
    
    # Filter to valid columns
    feature_vectors = feature_vectors[:, valid_columns]
    
    # Calculate mean and std for normalization (ignoring zeros)
    mean_values = []
    std_values = []
    
    for i in range(feature_vectors.shape[1]):
        non_zero_values = feature_vectors[:, i][feature_vectors[:, i] > 0.00001]
        if len(non_zero_values) > 0:
            mean_values.append(np.mean(non_zero_values))
            std_values.append(max(1, np.std(non_zero_values)))
        else:
            mean_values.append(0)
            std_values.append(1)
    
    # Apply normalization
    def normalize_matrix(matrix, mean_vals, std_vals):
        normalized = np.array(matrix, dtype=np.float32)
        for i in range(matrix.shape[1]):
            normalized[:, i] = np.where(
                normalized[:, i] < 0.00001, 
                -1,  # Replace zeros with -1 (inactive feature marker)
                (normalized[:, i] - mean_vals[i]) / std_vals[i]
            )
        return normalized
    
    feature_vectors_normalized = normalize_matrix(feature_vectors, mean_values, std_values)
    
    print(f"Normalized data shape: {feature_vectors_normalized.shape}")
    
    # Update data dictionary
    data['feature_vectors'] = feature_vectors_normalized
    data['valid_columns'] = valid_columns
    data['mean_values'] = mean_values
    data['std_values'] = std_values
    
    return data


def save_processed_data(data: Dict[str, Any], output_dir: str) -> None:
    """
    Save processed data in the format expected by TraceAnomaly for unsupervised training.
    
    Args:
        data: Dictionary containing processed data
        output_dir: Output directory path
    """
    print(f"Saving processed data to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, 'train')
    with open(train_file, 'w') as f:
        for flow_id, vector in zip(data['flow_ids'], data['feature_vectors']):
            vector_str = ','.join([str(x) for x in vector])
            f.write(f"{flow_id}:{vector_str}\n")
    
    # Save feature index map
    idx_file = os.path.join(output_dir, 'idx.pkl')
    with open(idx_file, 'wb') as f:
        pickle.dump(data['feature_index_map'], f)
    
    # Save metadata
    metadata = {
        'num_features': len(data['feature_index_map']),
        'valid_columns': data['valid_columns'],
        'mean_values': [float(x) for x in data['mean_values']],  # Convert to Python floats
        'std_values': [float(x) for x in data['std_values']],    # Convert to Python floats
        'total_samples': len(data['feature_vectors'])
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data saved successfully!")
    print(f"Training data: {metadata['total_samples']} samples")
    print(f"Features: {metadata['num_features']} total, {len(data['valid_columns'])} valid")


def main():
    """Main function to process trace data."""
    parser = argparse.ArgumentParser(description='Process trace data for TraceAnomaly')
    parser.add_argument('--input', required=True, help='Input JSONL.gz file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_feature_count', type=int, default=2, 
                       help='Minimum count for a feature to be included')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--drain_state_file', type=str, default=None,
                       help='Path to drain3 state file for template persistence')
    parser.add_argument('--training_mode', action='store_true', default=True,
                       help='Build templates (default: True)')
    parser.add_argument('--inference_mode', action='store_true', default=False,
                       help='Use existing templates')
    
    args = parser.parse_args()
    
    # Handle training/inference mode logic
    if args.inference_mode:
        training_mode = False
    else:
        training_mode = args.training_mode
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create processor with Drain3 configuration
    processor = TraceProcessor(
        min_feature_count=args.min_feature_count,
        drain_state_file=args.drain_state_file,
        training_mode=training_mode
    )
    
    # Process traces
    processor.process_traces(args.input)
    
    # Create feature vectors
    flow_ids, feature_vectors = processor.create_feature_vectors()
    
    # Prepare training data
    data = processor.prepare_training_data(flow_ids, feature_vectors)
    
    # Apply normalization
    data = apply_normalization_pipeline(data)
    
    # Save processed data
    save_processed_data(data, args.output_dir)
    
    print("Processing completed successfully!")


if __name__ == '__main__':
    main()
