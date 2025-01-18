from loguru import logger
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure Loguru
logger.add(
    "logs/system.log",  # Log file location
    rotation="500 MB",  # Rotate logs when the file reaches 500 MB
    retention="10 days",  # Keep logs for 10 days
    level="DEBUG",  # Minimum logging level
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
