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
        X,y = load_cesnet_data(pipeline.config['data']['csv_path'])
        logger.info(f"Data loaded successfully. Shape of X: {X.shape}")
        return [X,y]
        
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
        logger.info("Model and optimizer initialized")
        
        # Train model
        costs, outputs = pipeline.train_model(model, optimizer, data['X'], data['y'])
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

def run_testing_pipeline(config_path: str = 'config.yml') -> None:
    """Run the testing pipeline."""
    try:
        logger.info("Starting testing pipeline...")
        
        # Initialize pipeline and testing manager
        pipeline = GPUPipeline(config_path)
        testing_manager = TestingManager(pipeline.config)
        
        # Run the test pipeline
        testing_manager.test_pipeline(pipeline)
        
        logger.info("Testing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Testing pipeline failed: {str(e)}")
        raise

def main():
    """Main entry point."""
    try:
        # Load configuration
        config_path = 'config.yml'
        pipeline = GPUPipeline(config_path)
        
        # Determine execution mode
        mode = pipeline.config.get('execution_mode', 'test')
        logger.info(f"Execution mode: {mode}")
        
        # Execute pipeline based on mode
        if mode == 'train':
            X,y = load_data(pipeline)
            run_training_pipeline(pipeline, {'X': X, 'y': y})
        elif mode == 'test':
            run_testing_pipeline(config_path)
        else:
            raise ValueError(f"Invalid execution mode: {mode}")
            
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()