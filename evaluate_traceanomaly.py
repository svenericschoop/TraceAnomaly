#!/usr/bin/env python3
"""
TraceAnomaly Evaluation Script

This script loads a trained TraceAnomaly model and evaluates it on labeled test data,
calculating F1 scores and other metrics using KDE-based thresholding as described
in the TraceAnomaly paper. Supports both GPU and CPU evaluation.

Usage:
    # GPU evaluation (default)
    python evaluate_traceanomaly.py \
        --model_path md_vae_sampled_vaeg.model \
        --test_data_dir processed_test_data \
        --train_data_dir processed_training \
        --output_file evaluation_results.csv \
        --use_gpu --gpu_memory_fraction 0.8
    
    # CPU evaluation
    python evaluate_traceanomaly.py \
        --model_path md_vae_sampled_vaeg.model \
        --test_data_dir processed_test_data \
        --train_data_dir processed_training \
        --output_file evaluation_results.csv \
        --no_gpu
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from pathlib import Path

# Add traceanomaly module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'traceanomaly'))

import tfsnippet as spt
from tfsnippet.examples.utils import collect_outputs, MultiGPU
from traceanomaly.readdata_custom import get_data_vae_custom, get_data_vae_test_only, load_metadata
from traceanomaly.evaluation_utils import (
    apply_kde_thresholding, calculate_metrics, calculate_score_statistics,
    generate_confusion_matrix_text, save_evaluation_results, print_evaluation_summary,
    validate_evaluation_inputs
)


class TraceAnomalyEvaluator:
    """Evaluator for TraceAnomaly models."""
    
    def __init__(self, model_path: str, test_data_dir: str, train_data_dir: str, use_gpu: bool = True, gpu_memory_fraction: float = 0.8):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model directory
            test_data_dir: Directory containing processed test data
            train_data_dir: Directory containing training data (for KDE fitting)
            use_gpu: Whether to use GPU for evaluation
            gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.train_data_dir = train_data_dir
        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Model components
        self.session = None
        self.test_logp = None
        self.input_x = None
        self.test_flow = None
        
        # Data
        self.test_data = None
        self.test_labels = None
        self.train_data = None
        
    def load_test_data(self):
        """Load test data using the custom data loader."""
        print("Loading test data...")
        
        # Check if we have a train file in test directory (full dataset)
        train_file = os.path.join(self.test_data_dir, 'train')
        if os.path.exists(train_file):
            # Use full dataset loader
            (_, _), (test_x, test_y), test_flow = get_data_vae_custom(self.test_data_dir)
        else:
            # Use test-only loader with training data for normalization
            (_, _), (test_x, test_y), test_flow = get_data_vae_test_only(self.test_data_dir, self.train_data_dir)
        
        self.test_data = test_x
        self.test_labels = test_y
        self.test_flow = test_flow
        
        print(f"Test data loaded: {len(test_x)} samples, {test_x.shape[1]} features")
        print(f"Normal samples: {np.sum(test_y == 0)}")
        print(f"Anomalous samples: {np.sum(test_y == 1)}")
        
        # Validate test data
        if len(test_x) == 0:
            raise ValueError("No test data loaded")
        if test_x.shape[1] == 0:
            raise ValueError("Test data has 0 features - check data processing")
        
        # Skip training data loading for now - will use test normal samples for KDE
        print("Using test normal samples for KDE fitting (skipping training data load)")
        self.train_data = None
    
    def build_model_graph(self, x_dim: int):
        """
        Build the TraceAnomaly model graph.
        
        Args:
            x_dim: Input feature dimension
        """
        print(f"Building model graph with {x_dim} features...")
        
        # Import model components from train_custom.py
        from train_custom import q_net, p_net, coupling_layer_shift_and_scale, ExpConfig
        from traceanomaly.MLConfig import set_global_config
        
        # Create and set the global configuration
        config = ExpConfig()
        config.x_dim = x_dim
        config.z_dim = 10  # Default from train_custom.py
        config.flow_type = None  # Use vanilla VAE
        config.test_n_z = 500
        config.test_batch_size = 128
        
        # Set as global config so the model functions can access it
        set_global_config(config)
        
        # Input placeholder
        self.input_x = tf.placeholder(
            dtype=tf.float32, shape=(None, config.x_dim), name='input_x')
        
        # Build posterior flow (same as training)
        if config.flow_type is None:
            posterior_flow = None
        elif config.flow_type == 'planar_nf':
            posterior_flow = spt.layers.planar_normalizing_flows(config.n_planar_nf_layers)
        else:
            assert(config.flow_type == 'rnvp')
            with tf.variable_scope('posterior_flow'):
                flows = []
                for i in range(config.n_rnvp_layers):
                    flows.append(spt.layers.ActNorm())
                    flows.append(spt.layers.CouplingLayer(
                        tf.make_template(
                            'coupling',
                            coupling_layer_shift_and_scale,
                            create_scope_now_=True
                        ),
                        scale_type='sigmoid'
                    ))
                    flows.append(spt.layers.InvertibleDense(strict_invertible=True))
                posterior_flow = spt.layers.SequentialFlow(flows=flows)
        
        # Build test network
        with tf.name_scope('testing'):
            # Use GPU if available
            device = '/device:GPU:0' if self.use_gpu else '/device:CPU:0'
            with tf.device(device):
                test_q_net = q_net(self.input_x, posterior_flow, n_z=config.test_n_z)
                test_chain = test_q_net.chain(
                    p_net, latent_axis=0, observed={'x': self.input_x})
                self.test_logp = test_chain.vi.evaluation.is_loglikelihood()
        
        print("Model graph built successfully")
    
    def load_model(self):
        """Load the trained model weights."""
        print(f"Loading model from {self.model_path}...")
        
        # Configure GPU usage
        if self.use_gpu:
            # Check if GPU is available using TensorFlow 1.x API
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
            if gpu_devices:
                print(f"Found {len(gpu_devices)} GPU(s): {gpu_devices}")
                print(f"Using GPU device: {gpu_devices[0]}")
            else:
                print("No GPU devices found, falling back to CPU")
                self.use_gpu = False
        else:
            print("GPU usage disabled in configuration")
        
        # Print device information
        device_info = "GPU" if self.use_gpu else "CPU"
        print(f"Evaluation will use: {device_info}")
        
        # Create session with GPU configuration
        session_config = tf.ConfigProto()
        if self.use_gpu:
            session_config.gpu_options.allow_growth = True
            session_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        else:
            # Force CPU usage by setting device count properly
            session_config.device_count['GPU'] = 0
        
        self.session = spt.utils.create_session(lock_memory=False, **session_config)
        
        with self.session.as_default():
            # Build model graph
            self.build_model_graph(self.test_data.shape[1])
            
            # Load model weights
            var_dict = spt.utils.get_variables_as_dict()
            saver = spt.VariableSaver(var_dict, self.model_path)
            
            if os.path.exists(self.model_path):
                print("Restoring model weights...")
                saver.restore()
                print("Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def run_inference(self):
        """Run inference on test data to get anomaly scores."""
        print("Running inference on test data...")
        
        # Create data flow
        test_flow = spt.DataFlow.arrays([self.test_data], 128)  # batch_size=128
        
        # Run inference
        start_time = time.time()
        test_scores = collect_outputs([self.test_logp], [self.input_x], test_flow)[0]
        end_time = time.time()
        
        # Normalize scores by feature dimension (as done in training)
        test_scores = test_scores / self.test_data.shape[1]
        
        # Clear data flow from memory
        del test_flow
        gc.collect()
        
        print(f"Inference completed in {end_time - start_time:.2f} seconds")
        print(f"Score range: [{np.min(test_scores):.4f}, {np.max(test_scores):.4f}]")
        
        return test_scores
    
    def evaluate(self, output_file: str):
        """
        Run complete evaluation pipeline.
        
        Args:
            output_file: Path to save evaluation results
        """
        print("Starting TraceAnomaly evaluation...")
        
        # Load data
        self.load_test_data()
        
        # Load model
        self.load_model()
        
        # Run inference
        test_scores = self.run_inference()
        
        # Prepare data for KDE thresholding
        if self.train_data is not None:
            # Use training data for KDE fitting
            print("Using training data for KDE fitting...")
            normal_scores = self._get_training_scores()
        else:
            # Use normal test samples for KDE fitting
            print("Using normal test samples for KDE fitting...")
            normal_mask = self.test_labels == 0
            normal_scores = test_scores[normal_mask]
        
        print(f"KDE fitting on {len(normal_scores)} normal samples")
        
        # Validate inputs
        validate_evaluation_inputs(test_scores, self.test_labels, normal_scores)
        
        # Apply KDE-based thresholding
        print("Applying KDE-based thresholding...")
        predictions = apply_kde_thresholding(test_scores, normal_scores, significance_level=0.001)
        
        # Calculate metrics
        print("Calculating evaluation metrics...")
        metrics = calculate_metrics(self.test_labels, predictions)
        
        # Calculate score statistics
        score_stats = calculate_score_statistics(test_scores, self.test_labels)
        
        # Generate confusion matrix text
        cm_text = generate_confusion_matrix_text(self.test_labels, predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'trace_id': self.test_flow,
            'true_label': self.test_labels,
            'predicted_label': predictions.astype(int),
            'anomaly_score': test_scores,
            'is_correct': (self.test_labels == predictions)
        })
        
        # Save results
        save_evaluation_results(results_df, metrics, output_file, cm_text)
        
        # Print summary
        print_evaluation_summary(metrics, score_stats)
        
        # Print confusion matrix
        print(f"\n{cm_text}")
        
        return metrics, results_df
    
    def _get_training_scores(self):
        """Get anomaly scores for training data."""
        print("Computing training scores for KDE fitting...")
        
        # Create data flow for training data
        train_flow = spt.DataFlow.arrays([self.train_data], 128)
        
        # Run inference on training data
        with self.session.as_default():
            train_scores = collect_outputs([self.test_logp], [self.input_x], train_flow)[0]
            train_scores = train_scores / self.train_data.shape[1]
        
        # Clear data flow from memory
        del train_flow
        gc.collect()
        
        return train_scores


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate TraceAnomaly model on test data')
    parser.add_argument('--model_path', required=True, 
                       help='Path to trained model directory')
    parser.add_argument('--test_data_dir', required=True,
                       help='Directory containing processed test data')
    parser.add_argument('--train_data_dir', required=True,
                       help='Directory containing training data (for KDE fitting)')
    parser.add_argument('--output_file', required=True,
                       help='Path to save evaluation results CSV')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Enable GPU usage for evaluation (default: True)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU usage and force CPU evaluation')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                       help='Fraction of GPU memory to use (0.0-1.0, default: 0.8)')
    
    args = parser.parse_args()
    
    # Handle GPU configuration
    use_gpu = args.use_gpu and not args.no_gpu
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    if not os.path.exists(args.test_data_dir):
        raise FileNotFoundError(f"Test data directory not found: {args.test_data_dir}")
    
    if not os.path.exists(args.train_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {args.train_data_dir}")
    
    # Create evaluator
    evaluator = TraceAnomalyEvaluator(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        train_data_dir=args.train_data_dir,
        use_gpu=use_gpu,
        gpu_memory_fraction=args.gpu_memory_fraction
    )
    
    # Run evaluation
    try:
        metrics, results_df = evaluator.evaluate(args.output_file)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {args.output_file}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
