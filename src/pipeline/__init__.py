"""Pipeline module initialization."""

__version__ = "1.0.0"

# Import only the pipeline-specific components
from .gpu_pipeline import GPUPipeline
from .testing import TestingManager

# Public API for the pipeline package
__all__ = [
    "GPUPipeline",
    "TestingManager",
]
