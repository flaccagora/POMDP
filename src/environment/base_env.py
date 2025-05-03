from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from collections import OrderedDict

from src.config import EnvironmentConfig

class BaseEnvironment(ABC, gym.Env):
    """Base class for all environments in the system."""
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the base environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Initialize state
        self.state = None
        self.transformation_matrix = None
        
        # Set up observation and action spaces
        self._setup_spaces()
        
    @abstractmethod
    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing the current state
        """
        pass
        
    @abstractmethod
    def set_state(self, state: Dict[str, Any]):
        """
        Set the environment state.
        
        Args:
            state: Dictionary containing the state to set
        """
        pass
        
    @abstractmethod
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        pass
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Tuple of (initial_state, info)
        """
        pass
        
    @abstractmethod
    def render(self):
        """Render the current state of the environment."""
        pass
        
    def close(self):
        """Clean up resources."""
        pass
        
    def seed(self, seed: Optional[int] = None):
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            
    def get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Current observation as numpy array
        """
        state = self.get_state()
        return state['obs']
        
    def get_position(self) -> np.ndarray:
        """
        Get the current position.
        
        Returns:
            Current position as numpy array
        """
        state = self.get_state()
        return state['pos']
        
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the current transformation matrix.
        
        Returns:
            Current transformation matrix as numpy array
        """
        return self.transformation_matrix
        
    def set_transformation_matrix(self, matrix: np.ndarray):
        """
        Set the transformation matrix.
        
        Args:
            matrix: New transformation matrix
        """
        self.transformation_matrix = matrix
        
    def is_terminal(self) -> bool:
        """
        Check if the current state is terminal.
        
        Returns:
            True if the current state is terminal, False otherwise
        """
        return False
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary containing additional information
        """
        return {} 