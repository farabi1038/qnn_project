import logging
import sys
from typing import Optional

class LoggerConfig:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: str = 'casnet.log',
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Configure and return a logger instance
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
            format_string: Custom format string for logs
        """
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_string))
        
        # Add handlers if they don't exist
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        logger.info(f"Logger '{name}' initialized with level {level}")
        return logger 