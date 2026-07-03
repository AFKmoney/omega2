"""OMEGA2 package init."""
from omega2.core import Config, load_config
from omega2.orchestrator import Orchestrator

__version__ = "2.0.0"
__all__ = ["Config", "load_config", "Orchestrator", "__version__"]
