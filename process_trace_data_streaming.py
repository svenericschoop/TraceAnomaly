#!/usr/bin/env python3
"""
Memory-Efficient Trace Data Processing Script for TraceAnomaly

This script processes large JSON trace datasets in a memory-efficient way by:
1. Processing data in chunks
2. Using streaming for large datasets
3. Writing intermediate results to disk
4. Using sparse matrices for memory efficiency
"""

import json
import gzip
import argparse
import numpy as np
import pickle
import os
import tempfile
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import random
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
import gc
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import psutil
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from multiprocessing import cpu_count
import time


class StreamingTraceProcessor:
    """Memory-efficient trace processor that handles large datasets."""
    
    def __init__(self, min_feature_count: int = 50, chunk_size: int = 10000, 
                 drain_state_file: str = None, training_mode: bool = True, 
                 num_threads: int = None):
        """
        Initialize the streaming trace processor.
        
        Args:
            min_feature_count: Minimum number of occurrences for a feature to be included
            chunk_size: Number of traces to process in each chunk
            drain_state_file: Path to drain3 state file for template persistence
            training_mode: If True, builds templates. If False, uses existing templates.
            num_threads: Number of threads to use (default: all available cores)
        """
        self.min_feature_count = min_feature_count
        self.chunk_size = chunk_size
        self.feature_index_map = {}
        self.feature_counter = Counter()
        self.temp_dir = None
        self.chunk_files = []
        self.training_mode = training_mode
        self.num_threads = num_threads or cpu_count()
        
        # Thread-safe locks for shared resources
        self.counter_lock = threading.Lock()
        self.template_lock = threading.Lock()
        
        # Initialize drain3 template miner
        if drain_state_file:
            persistence = FilePersistence(drain_state_file)
            self.template_miner = TemplateMiner(persistence)
        else:
            # Use in-memory persistence if no file specified
            self.template_miner = TemplateMiner()
        
        # Regex patterns for span name normalization (from preprocessing.py)
        self.regex_patterns = [
            (r'/_doc/\d+', '/_doc/'),
            (r'\d+', '<NUM>'),  # Replace numbers with <NUM>
            (r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '<UUID>'),  # Replace UUIDs
            (r'/[a-fA-F0-9]{32,}', '/<HASH>'),  # Replace long hashes
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),  # Replace IP addresses
            (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),  # Replace dates
            (r'\b\d{2}:\d{2}:\d{2}\b', '<TIME>'),  # Replace times
        ]
    
    def log_memory_usage(self, step: str):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"[{step}] Memory usage: {memory_mb:.1f} MB")
        
        # Check if we're approaching memory limits
        if memory_mb > 8000:  # 8GB threshold
            print(f"WARNING: High memory usage detected ({memory_mb:.1f} MB)")
            print("Forcing garbage collection...")
            gc.collect()
    
    def normalize_span_name(self, span_name: str) -> str:
        """
        Normalize span name using regex patterns.
        
        Args:
            span_name: Original span name
            
        Returns:
            Normalized span name
        """
        normalized = span_name
        for pattern, replacement in self.regex_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized
    
    def parse_span_name_with_drain3(self, span_name: str) -> int:
        """
        Parse span name using drain3 template miner to get template ID.
        
        Args:
            span_name: Normalized span name
            
        Returns:
            Template cluster ID from drain3
        """
        with self.template_lock:
            if self.training_mode:
                # During training, use add_log_message to build template patterns
                result = self.template_miner.add_log_message(span_name)
                return result["cluster_id"] if result else 0
            else:
                # During testing, use match to find existing template patterns
                result = self.template_miner.match(span_name)
                return result.cluster_id if result else 0
        
    def extract_service_patterns(self, trace_data: Dict[str, Any]) -> List[str]:
        """Extract service call patterns from a trace using drain3 template parsing."""
        patterns = []
        
        def extract_from_span(span: Dict[str, Any], depth: int = 0) -> None:
            if depth > 10:  # Prevent infinite recursion
                return
                
            name = span.get('name', 'unknown')
            subtype = span.get('subtype', 'unknown')
            
            # Normalize the span name using regex patterns
            normalized_name = self.normalize_span_name(name)
            
            # Parse with drain3 to get template ID
            template_id = self.parse_span_name_with_drain3(normalized_name)
            
            # Create pattern using template ID instead of raw name
            pattern = f"template_{template_id}#{subtype}"
            patterns.append(pattern)
            
            children = span.get('children', [])
            for child in children:
                extract_from_span(child, depth + 1)
        
        if 'children' in trace_data:
            for child in trace_data['children']:
                extract_from_span(child)
        
        return patterns
    
    def create_sequence_patterns(self, patterns: List[str]) -> List[str]:
        """Create sequence patterns from individual service patterns."""
        sequence_patterns = []
        
        # Create patterns of different lengths (1, 2, 3)
        for length in [1, 2, 3]:
            for i in range(len(patterns) - length + 1):
                sequence = '#'.join(patterns[i:i+length])
                sequence_patterns.append(sequence)
        
        return sequence_patterns
    
    def process_single_trace(self, trace: Dict) -> Tuple[List[str], str]:
        """Process a single trace and return patterns and flow_id."""
        try:
            patterns = self.extract_service_patterns(trace['data'])
            sequence_patterns = self.create_sequence_patterns(patterns)
            return sequence_patterns, trace['id']
        except Exception as e:
            print(f"Warning: Skipping malformed trace {trace.get('id', 'unknown')}: {e}")
            return [], trace.get('id', 'unknown')
    
    def process_chunk_parallel(self, traces_chunk: List[Dict], chunk_idx: int) -> str:
        """Process a chunk of traces in parallel and save to temporary file."""
        print(f"Processing chunk {chunk_idx} with {len(traces_chunk)} traces using {self.num_threads} threads...")
        start_time = time.time()
        
        chunk_patterns = []
        chunk_flow_ids = []
        chunk_counter = Counter()
        
        # Process traces in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all trace processing tasks
            future_to_trace = {
                executor.submit(self.process_single_trace, trace): trace 
                for trace in traces_chunk
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_trace):
                sequence_patterns, flow_id = future.result()
                
                if sequence_patterns:  # Only add if processing was successful
                    chunk_patterns.append(sequence_patterns)
                    chunk_flow_ids.append(flow_id)
                    
                    # Count pattern frequencies for this chunk
                    for pattern in sequence_patterns:
                        chunk_counter[pattern] += 1
        
        # Update global counter in thread-safe manner
        with self.counter_lock:
            for pattern, count in chunk_counter.items():
                self.feature_counter[pattern] += count
        
        # Save chunk data to temporary file
        chunk_file = os.path.join(self.temp_dir, f'chunk_{chunk_idx}.pkl')
        with open(chunk_file, 'wb') as f:
            pickle.dump({
                'patterns': chunk_patterns,
                'flow_ids': chunk_flow_ids
            }, f)
        
        self.chunk_files.append(chunk_file)
        
        elapsed_time = time.time() - start_time
        print(f"Chunk {chunk_idx} completed in {elapsed_time:.2f} seconds ({len(traces_chunk)/elapsed_time:.1f} traces/sec)")
        
        return chunk_file
    
    def process_traces_streaming(self, input_file: str) -> None:
        """Process all traces from the input file in streaming mode."""
        print(f"Processing traces from {input_file} in streaming mode...")
        print(f"Using {self.num_threads} threads for parallel processing")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix='trace_processing_')
        print(f"Using temporary directory: {self.temp_dir}")
        
        # First pass: process in chunks and collect feature counts
        chunk_idx = 0
        current_chunk = []
        
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} traces...")
                    
                try:
                    trace_data = json.loads(line.strip())
                    current_chunk.append({
                        'id': trace_data.get('id', f'trace_{line_num}'),
                        'data': trace_data
                    })
                    
                    # Process chunk when it reaches the desired size
                    if len(current_chunk) >= self.chunk_size:
                        self.process_chunk_parallel(current_chunk, chunk_idx)
                        chunk_idx += 1
                        current_chunk = []
                        
                        # Force garbage collection
                        gc.collect()
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping malformed trace at line {line_num}: {e}")
                    continue
        
        # Process remaining traces
        if current_chunk:
            self.process_chunk_parallel(current_chunk, chunk_idx)
            chunk_idx += 1
        
        print(f"Total chunks created: {len(self.chunk_files)}")
        print(f"Total unique patterns found: {len(self.feature_counter)}")
        
        # Print drain3 template statistics
        if hasattr(self.template_miner, 'get_clusters'):
            clusters = self.template_miner.get_clusters()
            print(f"Drain3 templates created: {len(clusters)}")
            if self.training_mode:
                print("Training mode: Templates were built from span names")
            else:
                print("Inference mode: Using existing templates")
        
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
    
    def process_chunk_for_vectors(self, chunk_file: str) -> Tuple[List[int], List[int], List[float], List[str]]:
        """Process a single chunk file to extract feature vector data."""
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        
        patterns_list = chunk_data['patterns']
        flow_ids = chunk_data['flow_ids']
        
        row_indices = []
        col_indices = []
        data_values = []
        
        for row_idx, patterns in enumerate(patterns_list):
            # Count pattern occurrences
            pattern_counts = Counter(patterns)
            
            for pattern, count in pattern_counts.items():
                if pattern in self.feature_index_map:
                    idx = self.feature_index_map[pattern]
                    row_indices.append(row_idx)
                    col_indices.append(idx)
                    data_values.append(float(count))
        
        return row_indices, col_indices, data_values, flow_ids
    
    def create_feature_vectors_streaming(self) -> Tuple[List[str], str]:
        """Create feature vectors in streaming mode with parallel processing."""
        print(f"Creating feature vectors in streaming mode using {self.num_threads} threads...")
        self.log_memory_usage("Start feature vector creation")
        
        num_features = len(self.feature_index_map)
        all_flow_ids = []
        
        # Create temporary file for feature vectors
        vectors_file = os.path.join(self.temp_dir, 'feature_vectors.npz')
        
        # Process chunks in parallel
        row_indices = []
        col_indices = []
        data_values = []
        current_row = 0
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(self.process_chunk_for_vectors, chunk_file): chunk_file 
                for chunk_file in self.chunk_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_file = future_to_chunk[future]
                try:
                    chunk_row_indices, chunk_col_indices, chunk_data_values, flow_ids = future.result()
                    
                    # Adjust row indices to account for current position
                    adjusted_row_indices = [idx + current_row for idx in chunk_row_indices]
                    
                    row_indices.extend(adjusted_row_indices)
                    col_indices.extend(chunk_col_indices)
                    data_values.extend(chunk_data_values)
                    all_flow_ids.extend(flow_ids)
                    
                    current_row += len(flow_ids)
                    
                    print(f"Processed chunk: {chunk_file}")
                    self.log_memory_usage(f"Processed chunk {chunk_file}")
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_file}: {e}")
                    continue
                
                # Force garbage collection
                gc.collect()
        
        # Create sparse matrix
        print("Creating sparse matrix...")
        self.log_memory_usage("Before creating sparse matrix")
        
        feature_vectors = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(len(all_flow_ids), num_features),
            dtype=np.float32
        )
        
        # Save sparse matrix
        save_npz(vectors_file, feature_vectors)
        
        print(f"Created feature vectors: {feature_vectors.shape}")
        print(f"Saved to: {vectors_file}")
        self.log_memory_usage("End feature vector creation")
        
        return all_flow_ids, vectors_file
    
    def apply_normalization_streaming(self, vectors_file: str) -> str:
        """Apply normalization to the sparse matrix with memory-efficient processing."""
        print("Applying normalization pipeline...")
        self.log_memory_usage("Start normalization")
        
        # Load sparse matrix
        feature_vectors = load_npz(vectors_file)
        print(f"Loaded sparse matrix: {feature_vectors.shape}")
        self.log_memory_usage("After loading sparse matrix")
        
        # Find valid columns (memory-efficient way)
        print("Finding valid columns...")
        valid_columns = []
        for i in range(feature_vectors.shape[1]):
            if feature_vectors[:, i].nnz > 0:  # Check if column has any non-zero values
                valid_columns.append(i)
        
        print(f"Valid columns: {len(valid_columns)} out of {feature_vectors.shape[1]}")
        
        # Filter to valid columns
        feature_vectors = feature_vectors[:, valid_columns]
        self.log_memory_usage("After filtering columns")
        
        # Calculate statistics for normalization (memory-efficient)
        print("Calculating normalization statistics...")
        mean_values = []
        std_values = []
        
        # Process columns in batches to avoid memory issues
        batch_size = 500  # Reduced batch size for better memory management
        for batch_start in range(0, feature_vectors.shape[1], batch_size):
            batch_end = min(batch_start + batch_size, feature_vectors.shape[1])
            batch_cols = feature_vectors[:, batch_start:batch_end]
            
            for i in range(batch_cols.shape[1]):
                col_data = batch_cols[:, i].toarray().flatten()
                non_zero_values = col_data[col_data > 0.00001]
                
                if len(non_zero_values) > 0:
                    mean_values.append(np.mean(non_zero_values))
                    std_values.append(max(1, np.std(non_zero_values)))
                else:
                    mean_values.append(0)
                    std_values.append(1)
            
            # Force garbage collection after each batch
            gc.collect()
            if batch_start % 5000 == 0:
                self.log_memory_usage(f"Stats calculation batch {batch_start}")
        
        # Apply normalization in streaming fashion
        print("Applying normalization in streaming mode...")
        normalized_file = vectors_file.replace('.npz', '_normalized.npz')
        
        # Process in smaller chunks to manage memory better
        chunk_size = 200  # Further reduced chunk size for better memory management
        total_rows = feature_vectors.shape[0]
        
        # Create output file for streaming write
        output_vectors = None
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            
            # Process chunk
            chunk = feature_vectors[start:end].toarray()
            
            # Apply normalization
            for i in range(chunk.shape[1]):
                chunk[:, i] = np.where(
                    chunk[:, i] < 0.00001,
                    -1,  # Replace zeros with -1
                    (chunk[:, i] - mean_values[i]) / std_values[i]
                )
            
            # Convert to sparse and save incrementally
            chunk_sparse = csr_matrix(chunk)
            
            if output_vectors is None:
                # First chunk - create the file
                save_npz(normalized_file, chunk_sparse)
                output_vectors = load_npz(normalized_file)
            else:
                # Append to existing file
                temp_file = normalized_file + '.temp'
                save_npz(temp_file, chunk_sparse)
                temp_vectors = load_npz(temp_file)
                
                # Combine matrices
                combined = vstack([output_vectors, temp_vectors])
                save_npz(normalized_file, combined)
                output_vectors = combined
                
                # Clean up temp file
                os.remove(temp_file)
            
            # Force garbage collection
            gc.collect()
            
            if start % 10000 == 0:
                print(f"Normalized {start} rows...")
                self.log_memory_usage(f"Normalization progress {start}")
        
        print(f"Normalized data shape: {output_vectors.shape}")
        print(f"Saved to: {normalized_file}")
        self.log_memory_usage("End normalization")
        
        return normalized_file, valid_columns, mean_values, std_values
    
    def apply_normalization_ultra_efficient(self, vectors_file: str) -> str:
        """Ultra memory-efficient normalization that avoids loading the full matrix."""
        print("Applying ultra-efficient normalization pipeline...")
        self.log_memory_usage("Start ultra-efficient normalization")
        
        # Load sparse matrix
        feature_vectors = load_npz(vectors_file)
        print(f"Loaded sparse matrix: {feature_vectors.shape}")
        self.log_memory_usage("After loading sparse matrix")
        
        # Find valid columns
        print("Finding valid columns...")
        valid_columns = []
        for i in range(feature_vectors.shape[1]):
            if feature_vectors[:, i].nnz > 0:
                valid_columns.append(i)
        
        print(f"Valid columns: {len(valid_columns)} out of {feature_vectors.shape[1]}")
        
        # Filter to valid columns
        feature_vectors = feature_vectors[:, valid_columns]
        self.log_memory_usage("After filtering columns")
        
        # Calculate statistics without loading full matrix
        print("Calculating normalization statistics...")
        mean_values = []
        std_values = []
        
        # Process columns one by one to minimize memory usage
        for i in range(feature_vectors.shape[1]):
            col_data = feature_vectors[:, i].toarray().flatten()
            non_zero_values = col_data[col_data > 0.00001]
            
            if len(non_zero_values) > 0:
                mean_values.append(np.mean(non_zero_values))
                std_values.append(max(1, np.std(non_zero_values)))
            else:
                mean_values.append(0)
                std_values.append(1)
            
            # Clear column data immediately
            del col_data, non_zero_values
            
            if i % 100 == 0:
                gc.collect()
                self.log_memory_usage(f"Stats calculation column {i}")
        
        # Apply normalization in ultra-small chunks
        print("Applying ultra-efficient normalization...")
        normalized_file = vectors_file.replace('.npz', '_normalized.npz')
        
        # Process in very small chunks
        chunk_size = 100  # Ultra-small chunks
        total_rows = feature_vectors.shape[0]
        
        # Write directly to output file without keeping everything in memory
        output_file = open(normalized_file.replace('.npz', '.txt'), 'w')
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            
            # Process chunk
            chunk = feature_vectors[start:end].toarray()
            
            # Apply normalization
            for i in range(chunk.shape[1]):
                chunk[:, i] = np.where(
                    chunk[:, i] < 0.00001,
                    -1,  # Replace zeros with -1
                    (chunk[:, i] - mean_values[i]) / std_values[i]
                )
            
            # Write chunk to file immediately
            for row in chunk:
                row_str = ','.join([str(x) for x in row])
                output_file.write(f"{row_str}\n")
            
            # Clear chunk from memory
            del chunk
            gc.collect()
            
            if start % 10000 == 0:
                print(f"Normalized {start} rows...")
                self.log_memory_usage(f"Normalization progress {start}")
        
        output_file.close()
        
        # Convert text file to sparse matrix
        print("Converting to sparse matrix format...")
        self.convert_text_to_sparse(normalized_file.replace('.npz', '.txt'), normalized_file)
        
        print(f"Saved to: {normalized_file}")
        self.log_memory_usage("End ultra-efficient normalization")
        
        return normalized_file, valid_columns, mean_values, std_values
    
    def convert_text_to_sparse(self, text_file: str, output_file: str):
        """Convert text file to sparse matrix format."""
        print("Converting text file to sparse matrix...")
        
        # Read the text file and convert to sparse matrix
        data = []
        with open(text_file, 'r') as f:
            for line in f:
                row = [float(x) for x in line.strip().split(',')]
                data.append(row)
        
        # Convert to numpy array and then to sparse matrix
        data_array = np.array(data)
        sparse_matrix = csr_matrix(data_array)
        
        # Save sparse matrix
        save_npz(output_file, sparse_matrix)
        
        # Clean up text file
        os.remove(text_file)
        
        print(f"Converted to sparse matrix: {sparse_matrix.shape}")
    
    def save_processed_data_streaming(self, flow_ids: List[str], vectors_file: str, 
                                    valid_columns: List[int], mean_values: List[float], 
                                    std_values: List[float], output_dir: str) -> None:
        """Save processed data in the format expected by TraceAnomaly."""
        print(f"Saving processed data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load normalized vectors
        feature_vectors = load_npz(vectors_file).toarray()
        
        # Save training data
        train_file = os.path.join(output_dir, 'train')
        with open(train_file, 'w') as f:
            for flow_id, vector in zip(flow_ids, feature_vectors):
                vector_str = ','.join([str(x) for x in vector])
                f.write(f"{flow_id}:{vector_str}\n")
        
        # Save feature index map
        idx_file = os.path.join(output_dir, 'idx.pkl')
        with open(idx_file, 'wb') as f:
            pickle.dump(self.feature_index_map, f)
        
        # Save metadata
        metadata = {
            'num_features': len(self.feature_index_map),
            'valid_columns': valid_columns,
            'mean_values': [float(x) for x in mean_values],
            'std_values': [float(x) for x in std_values],
            'total_samples': len(feature_vectors)
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Data saved successfully!")
        print(f"Training data: {metadata['total_samples']} samples")
        print(f"Features: {metadata['num_features']} total, {len(valid_columns)} valid")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function to process trace data with streaming."""
    parser = argparse.ArgumentParser(description='Process large trace data for TraceAnomaly (streaming mode)')
    parser.add_argument('--input', required=True, help='Input JSONL.gz file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--min_feature_count', type=int, default=50, 
                       help='Minimum count for a feature to be included')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Number of traces to process in each chunk')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--drain_state_file', type=str, default=None,
                       help='Path to drain3 state file for template persistence')
    parser.add_argument('--training_mode', action='store_true', default=True,
                       help='Use training mode to build templates (default: True)')
    parser.add_argument('--inference_mode', action='store_true', default=False,
                       help='Use inference mode with existing templates')
    parser.add_argument('--ultra_efficient', action='store_true', default=False,
                       help='Use ultra-efficient normalization (recommended for very large datasets)')
    parser.add_argument('--num_threads', type=int, default=None,
                       help='Number of threads to use (default: all available cores)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine training mode
    training_mode = args.training_mode and not args.inference_mode
    
    # Create processor
    processor = StreamingTraceProcessor(
        min_feature_count=args.min_feature_count,
        chunk_size=args.chunk_size,
        drain_state_file=args.drain_state_file,
        training_mode=training_mode,
        num_threads=args.num_threads
    )
    
    try:
        # Process traces in streaming mode
        processor.process_traces_streaming(args.input)
        
        # Create feature vectors
        flow_ids, vectors_file = processor.create_feature_vectors_streaming()
        
        # Apply normalization
        if args.ultra_efficient:
            print("Using ultra-efficient normalization mode...")
            normalized_file, valid_columns, mean_values, std_values = processor.apply_normalization_ultra_efficient(vectors_file)
        else:
            print("Using standard streaming normalization mode...")
            normalized_file, valid_columns, mean_values, std_values = processor.apply_normalization_streaming(vectors_file)
        
        # Save processed data
        processor.save_processed_data_streaming(
            flow_ids, normalized_file, valid_columns, mean_values, std_values, args.output_dir
        )
        
        print("Processing completed successfully!")
        
    finally:
        # Clean up temporary files
        processor.cleanup()


if __name__ == '__main__':
    main()
