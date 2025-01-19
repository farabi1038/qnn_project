import os
import sys
from loguru import logger

def setup_logger():
    """Initialize and configure the logger."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure log file path
        log_file = os.path.join(log_dir, 'system.log')
        
        # Remove default handler
        logger.remove()
        
        # Add file handler
        logger.add(log_file,
                  rotation="500 MB",
                  retention="10 days",
                  level="DEBUG")
        
        # Add console handler
        logger.add(sys.stderr, level="INFO")
        
        logger.info(f"Log directory created/verified at: {log_dir}")
        logger.info(f"Log file verified at: {log_file}")
        logger.info(f"Logger initialized successfully at: {log_file}")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        raise

# Initialize logger
logger = setup_logger()

# Export logger instance
__all__ = ['logger']
