import torch
from collections import OrderedDict
from typing import Dict, Any, Tuple
from stable_baselines3 import PPO
from src.environment import POMDPDeformedGridworld
from src.agents import BaseAgent
from src.config import AgentConfig

class MLS(BaseAgent):
    """Maximum Likelihood Sampling agent implementation."""
    
    def __init__(self, mdp_agent: PPO, pomdp_env: POMDPDeformedGridworld, config: AgentConfig, obs_model=None):
        """
        Initialize the Maximum Likelihood Sampling agent.
        
        Args:
            mdp_agent: The underlying MDP agent (PPO)
            pomdp_env: The POMDP environment
            config: Agent configuration
        """
        super().__init__(mdp_agent, pomdp_env, config, obs_model)
        
    def predict(self, state: Dict[str, Any], deterministic: bool = True) -> Tuple[int, Any]:
        """
        Predict the next action using Maximum Likelihood Sampling.
        
        MLS works as follows:
        1. Find the theta with maximum likelihood in the current belief distribution
        2. Use the maximum likelihood theta to predict the next action
        
        Args:
            state: Current state dictionary
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        # Update belief state
        self.update_belief(state)
        
        # Get maximum likelihood theta
        if self.config.update_method == 'discrete_exact' or self.config.update_method == 'discrete':
            theta = self.belief_points[torch.argmax(self.belief_values).item()]
        elif self.config.update_method == 'variational':
            theta = self.VI.sample_latent(1).squeeze().clone().detach().numpy()
        elif self.config.update_method == 'particlefilters':
            mean, _ = self.PF.estimate_posterior()
            theta = torch.tensor(mean, dtype=torch.float32)
            
        # Create state with maximum likelihood theta
        pos = state['pos']
        state = OrderedDict({'pos': pos, 'theta': theta})
        
        # Get action from MDP agent
        action = self.mdp_agent.predict(state, deterministic=deterministic)
        
        # Call prediction callback if debug is enabled
        self.on_precidt_callback()
        
        return action 