import os
import sys
from datetime import datetime

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
        import pandas as pd
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

def load_data(pipeline):
    """Load and preprocess the data."""
    try:
        logger.info("Loading CESNET data...")
        data = load_cesnet_data(pipeline.config['data']['csv_path'])
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def run_training_pipeline(pipeline, data):
    """Run the main training pipeline."""
    try:
        logger.info("Starting training pipeline...")
        
        # Initialize model and optimizer
        model = pipeline.initialize_model()
        optimizer = pipeline.initialize_optimizer(model)
        logger.debug("Model and optimizer initialized")
        
        # Train model
        costs, outputs = pipeline.train_model(model, optimizer, data['X_train'], data['y_train'])
        logger.info(f"Training completed. Final loss: {costs[-1]:.4f}")
        
        # Evaluate model
        test_metrics = pipeline.compute_metrics(data['y_test'], 
                                             (model(data['X_test']) > 0.5).float())
        logger.info("Model evaluation completed")
        
        # Save results
        pipeline.save_results(test_metrics, {'loss': costs}, outputs, data['y_train'])
        pipeline.save_model(model, optimizer, test_metrics)
        logger.info("Results and model saved")
        
        return test_metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

def run_testing_pipeline(pipeline):
    """Run the testing pipeline."""
    try:
        logger.info("Starting testing pipeline...")
        testing_manager = TestingManager(pipeline)
        
        # Run quick test
        test_metrics = testing_manager.test_pipeline()
        logger.info("Quick test completed")
        
        # Run performance test if configured
        if pipeline.config.get('testing', {}).get('run_performance_test', False):
            logger.info("Starting performance test...")
            avg_metrics, execution_times = testing_manager.run_performance_test(
                num_iterations=pipeline.config['testing'].get('num_iterations', 5)
            )
            logger.info("Performance test completed")
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
        mode = pipeline.config.get('execution_mode', 'test')
        logger.info(f"Execution mode: {mode}")
        
        if mode == 'train':
            # Load and process data
            data = load_data(pipeline)
            
            # Run training pipeline
            metrics = run_training_pipeline(pipeline, data)
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