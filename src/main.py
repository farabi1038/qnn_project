import os
import sys
from datetime import datetime
import numpy as np
import json

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger
from src.pipeline import GPUPipeline, TestingManager
from src.data import load_cesnet_data
from src.core import AnomalyDetector, ZeroTrustFramework

# Initialize logger
logger = setup_logger()

def setup_environment():
    """Setup the environment and verify requirements."""
    try:
        # Verify CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Using GPU: {device_name}")
        else:
            logger.warning("CUDA is not available. Using CPU only.")
        
        # Verify other dependencies
        logger.debug("All required packages are available")
        
        # Create necessary directories
        required_dirs = ['data', 'logs', 'models', 'results', 'plots', 'checkpoints']
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
        logger.debug("Directory structure verified")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

def load_data():
    try:
        logger.info("Loading CESNET data...")
        X, y = load_cesnet_data()  # Directly unpack the tuple
        logger.info(f"Type of X is {type(X)}, type of y is {type(y)}")
        
        # Log shapes separately since y might be None
        logger.info(f"Data loaded successfully from load_cesnet_data inside load_data")
        if y is not None:
            logger.info(f"y shape: {y.shape}")
        else:
            logger.info("y is None")
            
        return X, y
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def run_training_pipeline(pipeline, data):
    """Run the main training pipeline."""
    try:
        logger.info("Starting training pipeline...")
        X = data['train_data']
        y = data['y_true']  # y is None in the data dictionary      
        # Initialize model and optimizer
        logger.info(f"Type of X is {type(X)}, type of y is {type(y)}")
        model = pipeline.initialize_model()
        optimizer = pipeline.initialize_optimizer(model)
        logger.debug("Model and optimizer initialized")
        
        # Train model
        costs, outputs = pipeline.train_model(model, optimizer, X)
        logger.info(f"Training completed. Final loss: {costs[-1]:.4f}")
        
        # Evaluate model
        test_outputs = model(X)
        test_metrics = pipeline.compute_metrics(X, 
                                             (test_outputs > 0.5).float())
        logger.info("Model evaluation completed")
        
        # Save everything
        logger.info("Saving model, results, and plots...")
        pipeline.save_model(model, optimizer, test_metrics)
        pipeline.save_results(
            metrics=test_metrics,
            history={'loss': costs},
            outputs=test_outputs,
            y_true=data['test_data']
        )
        logger.info("All artifacts saved successfully")
        
        return test_metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

def run_testing_pipeline(pipeline):
    """Run the testing pipeline."""
    try:
        logger.info("Starting testing pipeline...")
        testing_manager = TestingManager()
        
        # Run quick test
        test_metrics, costs, outputs, y_true = testing_manager.test_pipeline(pipeline)
        logger.info("Quick test completed")
        
        # Save test results
        logger.info("Saving test results and plots...")
        pipeline.save_results(
            metrics=test_metrics,
            history={'loss': costs},
            outputs=outputs,
            y_true=y_true
        )
        
        # Run performance test if configured
        if pipeline.config.get('testing', {}).get('run_performance_test', False):
            logger.info("Starting performance test...")
            avg_metrics, execution_times = testing_manager.run_performance_test(
                num_iterations=pipeline.config['testing'].get('num_iterations', 5)
            )
            logger.info("Performance test completed")
            
            # Save performance test results
            perf_results = {
                'avg_metrics': avg_metrics,
                'execution_times': execution_times.tolist(),
                'mean_execution_time': float(np.mean(execution_times)),
                'std_execution_time': float(np.std(execution_times))
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            perf_path = f"results/performance_{timestamp}.json"
            with open(perf_path, 'w') as f:
                json.dump(perf_results, f, indent=4)
            logger.info(f"Performance results saved to {perf_path}")
            
            return test_metrics, avg_metrics, execution_times
        
        return test_metrics, None, None
        
    except Exception as e:
        logger.error(f"Testing pipeline failed: {str(e)}")
        raise

def main():
    """Main entry point for the pipeline."""
    try:
        start_time = datetime.now()
        logger.info("Starting pipeline execution")
        
        # Setup environment
        if not setup_environment():
            logger.error("Environment setup failed. Exiting.")
            sys.exit(1)
        
        # Initialize pipeline
        logger.info("Initializing GPU Pipeline...")
        pipeline = GPUPipeline()
        
        # Determine execution mode
        mode = pipeline.config.get('execution_mode')
        logger.info(f"Execution mode: {mode}")
        
        if mode == 'train':
            # Load and process data
            logger.info("Loading data...")
            X, y = load_data()
            # Run training pipeline
            metrics = run_training_pipeline(pipeline, {'train_data': X, 'test_data': X, 'y_true': y})
            logger.info("Training pipeline completed successfully")
            logger.info(f"Final metrics: {metrics}")
            
        elif mode == 'test':
            # Run testing pipeline
            metrics, avg_metrics, execution_times = run_testing_pipeline(pipeline)
            logger.info("Testing pipeline completed successfully")
            logger.info(f"Test metrics: {metrics}")
            if avg_metrics:
                logger.info(f"Average performance metrics: {avg_metrics}")
                logger.info(f"Average execution time: {np.mean(execution_times):.2f} seconds")
        
        else:
            logger.error(f"Unknown execution mode: {mode}")
            sys.exit(1)
        
        execution_time = datetime.now() - start_time
        logger.info(f"Total execution time: {execution_time}")
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 