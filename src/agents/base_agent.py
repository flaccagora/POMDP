"""
Base agent module defining the interface for all agents in the system.

This module provides the abstract base class that all agent implementations must inherit from.
The base class defines the core interface that agents must implement, including prediction,
update, and reset functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import torch

from src.config import AgentConfig
from src.utils import BetaVariationalBayesianInference, BayesianParticleFilter

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    This class defines the interface that all agent implementations must follow.
    It provides the basic structure for agents that can interact with environments,
    make predictions, and update their internal state based on observations.
    
    Attributes:
        config (AgentConfig): Configuration object containing agent parameters
        is_training (bool): Flag indicating whether the agent is in training mode
    
    Methods:
        predict: Make a prediction based on the current state
        update: Update the agent's internal state based on observations
        reset: Reset the agent to its initial state
        get_config: Get the agent's configuration
        train: Set the agent to training mode
        eval: Set the agent to evaluation mode
    """
    
    def __init__(self, mdp_agent, pomdp_env, config: AgentConfig, obs_model=None):
        """
        Initialize the base agent.
        
        Args:
            mdp_agent: The underlying MDP agent (PPO or DQN)
            pomdp_env: The POMDP environment
            config: Agent configuration
            obs_model: Optional observation model
        """
        self.pomdp_env = pomdp_env
        self.mdp_agent = mdp_agent
        self.config = config
        self.obs_model = obs_model
        self.device = self.config.device
        
        if obs_model is not None:
            # Initialize belief state
            self._initialize_belief_state()
            
            # History tracking
            self.X_history = [self.pomdp_env.get_state()['pos']]
            self.y_history = [self.pomdp_env.get_state()['obs']]
            self.entropy = None
        
        self.is_training = True
        
    def _initialize_belief_state(self):
        """Initialize the belief state based on the update method."""
        if self.config.update_method in ['discrete', 'discrete_exact']:
            self._initialize_discrete_belief()
        elif self.config.update_method == 'variational':
            self._initialize_variational_belief()
        elif self.config.update_method == 'particlefilters':
            self._initialize_particle_filter()
        else:
            raise ValueError(f"Invalid update method: {self.config.update_method}")
            
    def _initialize_discrete_belief(self):
        """Initialize discrete belief state."""
        stretch = np.linspace(.4, 1, self.config.discretization)
        shear = np.linspace(-.2, .2, self.config.discretization)
        xa, ya, yb, xb = np.meshgrid(stretch, shear, shear, stretch)
        positions = np.column_stack([xa.ravel(), ya.ravel(), yb.ravel(), xb.ravel()])
        self.belief_points = torch.tensor(positions, dtype=torch.float32, device=self.device)
        self.belief_values = torch.ones(self.belief_points.shape[0], dtype=torch.float32, device=self.device) / len(positions)
        
    def _initialize_variational_belief(self):
        """Initialize variational belief state."""
        if self.obs_model is None:
            raise ValueError("Observation model required for variational belief update")
        self.VI = BetaVariationalBayesianInference(
            self.obs_model, 
            input_dim=2, 
            latent_dim=self.config.theta_dim, 
            debug=self.config.debug
        )
        
    def _initialize_particle_filter(self):
        """Initialize particle filter belief state."""
        if self.obs_model is None:
            raise ValueError("Observation model required for particle filter belief update")
        self.PF = BayesianParticleFilter(
            f=self.obs_model,
            n_particles=self.config.n_particles,
            theta_dim=self.config.theta_dim
        )
        self.belief_points, self.belief_values = self.PF.initialize_particles()
        
    @abstractmethod
    def predict(self, state: Dict[str, Any], deterministic: bool = True) -> Tuple[int, Any]:
        """
        Predict the next action given the current state.
        
        Args:
            state: Current state dictionary
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        pass
        
    def update_belief(self, state: Dict[str, Any]):
        """Update the belief state based on the current state."""
        if self.config.update_method == 'discrete':
            self._update_discrete_belief(state)
        elif self.config.update_method == 'variational':
            self._update_variational_belief(state)
        elif self.config.update_method == 'particlefilters':
            self._update_particle_filter(state)
            
    def _update_discrete_belief(self, state: Dict[str, Any]):
        """Update discrete belief state."""
        if self.obs_model is None:
            raise ValueError("Observation model required for discrete belief update")
        # Implementation specific to discrete belief update
        pass
        
    def _update_variational_belief(self, state: Dict[str, Any]):
        """Update variational belief state."""
        self.X_history.append(state['pos'])
        self.y_history.append(state['obs'])
        X = torch.stack(self.X_history)
        y = torch.stack(self.y_history)
        self.VI.fit(X, y, n_epochs=10, lr=0.05)
        
    def _update_particle_filter(self, state: Dict[str, Any]):
        """Update particle filter belief state."""
        self.X_history.append(state['pos'])
        self.y_history.append(state['obs'])
        X = torch.stack(self.X_history[-1:])
        y = torch.stack(self.y_history[-1:])
        self.belief_points, self.belief_values = self.PF.update(X, y)
        
    def reset(self):
        """Reset the agent's state."""
        self.X_history = [self.pomdp_env.get_state()['pos']]
        self.y_history = [self.pomdp_env.get_state()['obs']]
        self.entropy = None
        
        if self.config.update_method == 'discrete':
            self.belief_values = torch.ones(self.belief_points.shape[0], dtype=torch.float32, device=self.device) / len(self.belief_points)
        elif self.config.update_method == 'variational':
            self._initialize_variational_belief()
        elif self.config.update_method == 'particlefilters':
            self._initialize_particle_filter()
            
    def get_entropy(self) -> float:
        """Calculate the current entropy of the belief state."""
        if self.config.update_method == 'discrete':
            return torch.distributions.Categorical(probs=self.belief_values).entropy().item()
        elif self.config.update_method == 'variational':
            return self.VI.entropy()
        elif self.config.update_method == 'particlefilters':
            return self.PF.entropy()
        return 0.0 
    def on_precidt_callback(self):
        """Callback function to be called after prediction."""
        pass

    def get_config(self) -> AgentConfig:
        """
        Get the agent's configuration.
        
        Returns:
            AgentConfig: The agent's configuration object
        """
        return self.config
    
    def train(self) -> None:
        """
        Set the agent to training mode.
        
        This method sets the agent's internal state to training mode, which may
        affect how the agent behaves (e.g., exploration vs exploitation).
        """
        self.is_training = True
    
    def eval(self) -> None:
        """
        Set the agent to evaluation mode.
        
        This method sets the agent's internal state to evaluation mode, which may
        affect how the agent behaves (e.g., no exploration, only exploitation).
        """
        self.is_training = False
