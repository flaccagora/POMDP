"""
Base environment class defining the interface for all environments.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from src.config import EnvironmentConfig

class BaseEnvironment(ABC):
    """Base class for all environments in the system."""
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration object
        """
        # check if config is EnvironmentConfig
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an instance of EnvironmentConfig")
        self.config = config
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple containing:
                - observation: New state observation
                - reward: Reward received
                - done: Whether episode is done
                - info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render the current state of the environment."""
        pass 