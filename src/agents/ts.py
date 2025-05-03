import torch
from collections import OrderedDict
from typing import Dict, Any, Tuple
from stable_baselines3 import PPO
from src.environment import POMDPDeformedGridworld
from src.agents import BaseAgent
from src.config import AgentConfig

class TS(BaseAgent):
    """Thompson Sampling agent implementation."""
    
    def __init__(self, mdp_agent: PPO, pomdp_env: POMDPDeformedGridworld, config: AgentConfig, obs_model=None):
        """
        Initialize the Thompson Sampling agent.
        
        Args:
            mdp_agent: The underlying MDP agent (PPO)
            pomdp_env: The POMDP environment
            config: Agent configuration
        """
        super().__init__(mdp_agent, pomdp_env, config, obs_model)
        
    def predict(self, state: Dict[str, Any], deterministic: bool = True) -> Tuple[int, Any]:
        """
        Predict the next action using Thompson Sampling.
        
        Thompson Sampling works as follows:
        1. Sample a theta from the current belief distribution
        2. Use the sampled theta to predict the next action
        
        Args:
            state: Current state dictionary
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        # Update belief state
        self.update_belief(state)
        
        # Sample theta from belief distribution
        if self.config.update_method == 'discrete_exact' or self.config.update_method == 'discrete':
            theta = self.belief_points[torch.multinomial(self.belief_values, 1).item()]
        elif self.config.update_method == 'variational':
            theta = self.VI.sample_latent(1).squeeze().clone().detach().numpy()
        elif self.config.update_method == 'particlefilters':
            mean, var = self.PF.estimate_posterior()
            theta = torch.distributions.Normal(
                torch.tensor(mean), 
                torch.tensor(var).sqrt() + 1e-6
            ).sample().squeeze()
            
        # Create state with sampled theta
        pos = state['pos']
        state = OrderedDict({'pos': pos, 'theta': theta})
        
        # Get action from MDP agent
        action = self.mdp_agent.predict(state, deterministic=deterministic)
        
        # Call prediction callback if debug is enabled
        self.on_precidt_callback()
        
        return action 
    
    