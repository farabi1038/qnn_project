import logging
import os

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/qnn_project.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

# Create a logger instance for the project
logger = logging.getLogger("qnn_project")
