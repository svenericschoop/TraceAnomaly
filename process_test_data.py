#!/usr/bin/env python3
"""
Test Data Processing Script for TraceAnomaly

This script processes labeled test data (jsonl.gz with is_anomaly field) into the format
expected by the TraceAnomaly evaluation system. It uses existing templates and feature
mappings from training to ensure consistency.

Usage:
    python process_test_data.py \
        --input test_data.jsonl.gz \
        --output_dir processed_test_data \
        --training_idx_file processed_training/idx.pkl \
        --drain_state_file templates.pkl
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


class TestDataProcessor:
    """Processes labeled test data into TraceAnomaly format."""
    
    def __init__(self, training_idx_file: str, drain_state_file: str = None):
        """
        Initialize the test data processor.
        
        Args:
            training_idx_file: Path to training data's idx.pkl file
            drain_state_file: Path to Drain3 state file for template persistence
        """
        self.training_idx_file = training_idx_file
        self.drain_state_file = drain_state_file
        
        # Load feature index mapping from training
        with open(training_idx_file, 'rb') as f:
            self.feature_index_map = pickle.load(f)
        
        print(f"Loaded feature index map with {len(self.feature_index_map)} features")
        
        # Initialize Drain3 template miner
        if drain_state_file and os.path.exists(drain_state_file):
            persistence = FilePersistence(drain_state_file)
            self.template_miner = TemplateMiner(persistence)
            print(f"Loaded Drain3 templates from {drain_state_file}")
        else:
            self.template_miner = None
            print("Warning: No Drain3 state file provided, using raw span names")
        
        # Storage for processed traces
        self.traces = []
        self.normal_traces = []
        self.abnormal_traces = []
    
    def normalize_span_name(self, span_name: str) -> str:
        """
        Normalize span name using regex patterns.
        
        Args:
            span_name: Raw span name
            
        Returns:
            Normalized span name
        """
        if not span_name:
            return "unknown"
            
        normalized = span_name
        
        # Use the same regex patterns as training script
        regex_patterns = [
            (r'/_doc/\d+', '/_doc/'),
            (r'\d+', '<NUM>'),  # Replace numbers with <NUM>
            (r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>'),  # Replace UUIDs
            (r'/[a-fA-F0-9]{32,}', '/<HASH>'),  # Replace long hashes
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),  # Replace IP addresses
            (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),  # Replace dates
            (r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>'),  # Replace times
        ]
        
        for pattern, replacement in regex_patterns:
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
        if self.template_miner:
            result = self.template_miner.match(span_name)
            return result.cluster_id if result else 0
        else:
            return 0
    
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
        Create sequence patterns from individual patterns.
        
        Args:
            patterns: List of individual patterns
            
        Returns:
            List of sequence patterns
        """
        sequence_patterns = []
        
        # Add individual patterns
        sequence_patterns.extend(patterns)
        
        # Create 2-grams
        for i in range(len(patterns) - 1):
            sequence_patterns.append(f"{patterns[i]}#{patterns[i+1]}")
        
        # Create 3-grams
        for i in range(len(patterns) - 2):
            sequence_patterns.append(f"{patterns[i]}#{patterns[i+1]}#{patterns[i+2]}")
        
        return sequence_patterns
    
    def process_traces(self, input_file: str) -> None:
        """
        Process all traces from the input file.
        
        Args:
            input_file: Path to the input JSONL.gz file
        """
        print(f"Processing test traces from {input_file}...")
        
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} traces...")
                    
                try:
                    trace_data = json.loads(line.strip())
                    
                    # Extract service patterns
                    patterns = self.extract_service_patterns(trace_data)
                    sequence_patterns = self.create_sequence_patterns(patterns)
                    
                    # Get anomaly label
                    is_anomaly = trace_data.get('is_anomaly', False)
                    
                    # Store trace data
                    trace_info = {
                        'id': trace_data.get('id', f'trace_{line_num}'),
                        'patterns': sequence_patterns,
                        'duration': trace_data.get('duration', 0),
                        'is_anomaly': is_anomaly
                    }
                    
                    self.traces.append(trace_info)
                    
                    # Split into normal/abnormal
                    if is_anomaly:
                        self.abnormal_traces.append(trace_info)
                    else:
                        self.normal_traces.append(trace_info)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed trace at line {line_num}: {e}")
                    continue
        
        print(f"Total traces processed: {len(self.traces)}")
        print(f"Normal traces: {len(self.normal_traces)}")
        print(f"Abnormal traces: {len(self.abnormal_traces)}")
    
    def create_feature_vectors(self, traces: List[Dict]) -> Tuple[List[str], np.ndarray]:
        """
        Create feature vectors from processed traces.
        
        Args:
            traces: List of trace dictionaries
            
        Returns:
            Tuple of (flow_ids, feature_vectors)
        """
        print(f"Creating feature vectors for {len(traces)} traces...")
        
        num_features = len(self.feature_index_map)
        flow_ids = []
        feature_vectors = []
        
        for trace in traces:
            flow_id = trace['id']
            patterns = trace['patterns']
            
            # Create feature vector
            feature_vector = np.zeros(num_features, dtype=np.float32)
            
            # Count pattern occurrences
            pattern_counts = Counter(patterns)
            
            for pattern, count in pattern_counts.items():
                if pattern in self.feature_index_map:
                    feature_vector[self.feature_index_map[pattern]] = count
            
            flow_ids.append(flow_id)
            feature_vectors.append(feature_vector)
        
        return flow_ids, np.array(feature_vectors)
    
    def save_processed_data(self, output_dir: str) -> None:
        """
        Save processed test data.
        
        Args:
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create feature vectors for normal and abnormal traces
        normal_flow_ids, normal_vectors = self.create_feature_vectors(self.normal_traces)
        abnormal_flow_ids, abnormal_vectors = self.create_feature_vectors(self.abnormal_traces)
        
        # Save normal test data
        normal_file = os.path.join(output_dir, 'test_normal')
        with open(normal_file, 'w') as f:
            for flow_id, vector in zip(normal_flow_ids, normal_vectors):
                vector_str = ','.join([str(x) for x in vector])
                f.write(f"{flow_id}:{vector_str}\n")
        
        # Save abnormal test data
        abnormal_file = os.path.join(output_dir, 'test_abnormal')
        with open(abnormal_file, 'w') as f:
            for flow_id, vector in zip(abnormal_flow_ids, abnormal_vectors):
                vector_str = ','.join([str(x) for x in vector])
                f.write(f"{flow_id}:{vector_str}\n")
        
        # Copy training idx.pkl to test directory for consistency
        import shutil
        shutil.copy2(self.training_idx_file, os.path.join(output_dir, 'idx.pkl'))
        
        # Create metadata
        metadata = {
            'test_samples': len(self.traces),
            'normal_test_samples': len(self.normal_traces),
            'anomalous_test_samples': len(self.abnormal_traces),
            'num_features': len(self.feature_index_map),
            'valid_columns': list(range(len(self.feature_index_map))),
            'training_idx_file': self.training_idx_file,
            'drain_state_file': self.drain_state_file
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved processed test data to {output_dir}")
        print(f"Normal test samples: {len(self.normal_traces)}")
        print(f"Abnormal test samples: {len(self.abnormal_traces)}")
        print(f"Feature dimension: {len(self.feature_index_map)}")


def main():
    """Main function to process test data."""
    parser = argparse.ArgumentParser(description='Process labeled test data for TraceAnomaly evaluation')
    parser.add_argument('--input', required=True, help='Input JSONL.gz file with is_anomaly field')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--training_idx_file', required=True, 
                       help='Path to training data idx.pkl file')
    parser.add_argument('--drain_state_file', type=str, default=None,
                       help='Path to drain3 state file for template persistence')
    
    args = parser.parse_args()
    
    # Create processor
    processor = TestDataProcessor(
        training_idx_file=args.training_idx_file,
        drain_state_file=args.drain_state_file
    )
    
    # Process traces
    processor.process_traces(args.input)
    
    # Save processed data
    processor.save_processed_data(args.output_dir)
    
    print("Test data processing completed successfully!")


if __name__ == '__main__':
    main()
