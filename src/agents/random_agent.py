"""
Random agent implementation.

This module provides a simple random agent that selects actions uniformly at random.
It serves as a baseline for comparison with more sophisticated agents and can be
used for testing environments.
"""

import numpy as np
from typing import Any, Dict, Tuple

from src.agents import BaseAgent
from src.config import AgentConfig
from src.environment import POMDPDeformedGridworld

class RandomAgent(BaseAgent):
    """
    A random agent that selects actions uniformly at random.
    
    This agent implements a simple random policy that selects actions uniformly
    at random from the available action space. It is useful as a baseline for
    comparison with more sophisticated agents and for testing environments.
    
    Attributes:
        num_actions (int): Number of possible actions
        rng (np.random.RandomState): Random number generator for action selection
    
    Methods:
        predict: Select a random action
        update: No-op for random agent
        reset: No-op for random agent
    """
    
    def __init__(self, pomdp_env: POMDPDeformedGridworld, config: AgentConfig):
        """
        Initialize the random agent.
        
        Args:
            config: Configuration object containing agent parameters
        """
        super().__init__(None, pomdp_env, config)
        self.num_actions = 4  # Fixed number of actions for grid environment
        self.rng = np.random.RandomState(config.seed if hasattr(config, 'seed') else 42)
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Select a random action.
        
        This method selects an action uniformly at random from the available
        action space. The observation is ignored as this is a random policy.
        
        Args:
            observation: Current state observation (ignored)
            
        Returns:
            int: A random action from the action space
        """
        return self.rng.randint(0, self.num_actions), {}
    
    def update(self, observation: np.ndarray, action: int, 
              reward: float, next_observation: np.ndarray, 
              done: bool) -> None:
        """
        No-op for random agent.
        
        The random agent does not learn from experience, so this method does nothing.
        
        Args:
            observation: Previous state observation (ignored)
            action: Action taken (ignored)
            reward: Reward received (ignored)
            next_observation: New state observation (ignored)
            done: Whether episode is done (ignored)
        """
        pass
    
    def reset(self) -> None:
        """
        No-op for random agent.
        
        The random agent does not maintain any state between episodes,
        so this method does nothing.
        """
        pass 