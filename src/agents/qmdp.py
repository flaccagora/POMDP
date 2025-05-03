import torch
from collections import OrderedDict
from typing import Dict, Any, Tuple
from stable_baselines3 import DQN
from src.environment import POMDPDeformedGridworld
from src.agents import BaseAgent
from src.config import AgentConfig

class QMDP(BaseAgent):
    """QMDP agent implementation."""
    
    def __init__(self, mdp_agent: DQN, pomdp_env: POMDPDeformedGridworld, config: AgentConfig, obs_model=None):
        """
        Initialize the QMDP agent.
        
        Args:
            mdp_agent: The underlying MDP agent (DQN)
            pomdp_env: The POMDP environment
            config: Agent configuration
        """
        super().__init__(mdp_agent, pomdp_env, config, obs_model)
        
        # Store Q-network reference
        self.Q = self.mdp_agent.policy.q_net
        
        # Move observation model to correct device
        if self.obs_model is not None:
            self.obs_model.to(self.mdp_agent.device)
            
    def predict(self, state: Dict[str, Any], deterministic: bool = True) -> Tuple[int, Any]:
        """
        Predict the next action using QMDP.
        
        QMDP works as follows:
        1. Compute Q(s,a) for all belief points and actions
        2. Compute QMDP(s) = argmax_a sum_{theta} b(theta) * Q(s,a)
        3. Return QMDP(s)
        
        Args:
            state: Current state dictionary
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        # Update belief state
        self.update_belief(state)
        
        # Move belief points to device if needed
        belief_points = self.belief_points.to(self.mdp_agent.device)
        belief_values = self.belief_values.to(self.mdp_agent.device)
        
        # Step 1: Compute Q(s,a) for all belief points and actions
        B = belief_values.shape[0]  # number of belief points
        state = OrderedDict({
            'pos': state['pos'].expand(B, -1).to(self.mdp_agent.device),
            'theta': belief_points
        })
        
        # Get Q-values for all actions
        q_values = self.Q(state)
        
        # Step 2: Compute QMDP(s) = argmax_a \sum_{theta} b(theta) * Q(s,a)
        actions = torch.einsum("s,sa->a", belief_values, q_values)
        
        if deterministic:
            return torch.argmax(actions).item(), actions
        else:
            # Implement stochastic policy
            probs = torch.softmax(actions / self.config.temperature, dim=0)
            return torch.multinomial(probs, 1).item(), actions 