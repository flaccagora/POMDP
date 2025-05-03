import torch
from collections import OrderedDict
from typing import Dict, Any, Tuple
from stable_baselines3 import DQN
from src.environment.env import POMDPDeformedGridworld
from src.agents import BaseAgent
from src.config import AgentConfig

class Infotaxis(BaseAgent):
    def __init__(self,mdp_agent:DQN, pomdp_env: POMDPDeformedGridworld, config, obs_model=None):
        """
        Infotaxis agent implementation.
        Args:
            mdp_agent: The underlying MDP agent (DQN)
            pomdp_env: The POMDP environment
            config: Agent configuration
        """
        # Initialize the base class
        super().__init__(mdp_agent, pomdp_env,config, obs_model)

        self.pomdp_bis = POMDPDeformedGridworld(
            render_mode="rgb_array",
            obs_type='cardinal'
        )
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        def make_env():
            pomdp_bis = POMDPDeformedGridworld(
                render_mode="rgb_array",
                obs_type='cardinal'
            )
            return pomdp_bis
        
        self.pomdp_bis = SubprocVecEnv([make_env]*4)
    
    def get_entropy(self, belief):
        if self.config.update_method == 'discrete':
            return torch.distributions.Categorical(probs=belief).entropy()
        elif self.config.update_method == 'particlefilters':
            # return self.PF.entropy()
            raise NotImplementedError('Particle filters not implemented still missing belief tmp')
        elif self.config.update_method == 'variational':
            raise NotImplementedError('Variational Infotaxis not implemented')
        
    def update_discrete_belief_tmp(self, b, s):

        """discrete belief update"""
        pos = s['pos']
        obs = s['obs']

        batch_pos = pos.expand(len(self.belief_points),-1)

        model_device = next(self.obs_model.parameters()).device
        with torch.no_grad():
            predictions = self.obs_model(batch_pos.to(model_device),self.belief_points.to(model_device))

        likelihood = torch.distributions.Bernoulli(predictions).log_prob(obs.to(model_device))
        if len(likelihood.shape) == 2:
            likelihood = likelihood.sum(dim=1)
        likelihood = likelihood.exp()

        tmp = likelihood.squeeze() * b.to(model_device)
        b = tmp  / tmp.sum()
        
        return b
        
    def update_discrete_belief_tmp_batched(self, b, s):
        """discrete belief update for batched inputs"""
        pos = torch.tensor(s['pos'], dtype=torch.float32)
        obs = torch.tensor(s['obs'], dtype=torch.float32)
        batch_size = pos.shape[0]
        
        # Expand dimensions for proper broadcasting
        # belief_points: [num_belief_points, belief_dim]
        # pos: [batch_size, pos_dim]
        # We need: [batch_size, num_belief_points, pos_dim]
        expanded_pos = pos.unsqueeze(1).expand(batch_size, len(self.belief_points), -1)
        expanded_belief_points = self.belief_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        model_device = next(self.obs_model.parameters()).device
        
        with torch.no_grad():
            # Reshape for the model
            # The model expects: [batch_size * num_belief_points, pos_dim + belief_dim]
            # reshaped_input = torch.cat([
            #     expanded_pos.reshape(batch_size * len(self.belief_points), -1),
            #     expanded_belief_points.reshape(batch_size * len(self.belief_points), -1)
            # ], dim=1).to(model_device)
            
            predictions = self.obs_model(expanded_pos.reshape(batch_size * len(self.belief_points), -1).to(model_device), expanded_belief_points.reshape(batch_size * len(self.belief_points), -1).to(model_device))
            
            # Reshape predictions to [batch_size, num_belief_points, obs_dim]
            predictions = predictions.reshape(batch_size, len(self.belief_points), -1)
            
            # Expand obs to match prediction shape for computing log probability
            expanded_obs = obs.unsqueeze(1).expand(batch_size, len(self.belief_points), -1).to(model_device)
            
            # Compute log probabilities
            likelihood = torch.distributions.Bernoulli(predictions).log_prob(expanded_obs)
            
            if len(likelihood.shape) > 2:
                likelihood = likelihood.sum(dim=2)  # Sum over observation dimensions
                
            likelihood = likelihood.exp()  # Shape: [batch_size, num_belief_points]
            
            # Expand belief to match the batch dimension
            expanded_b = b.unsqueeze(0).expand(batch_size, -1).to(model_device)
            
            # Update beliefs for each item in batch
            tmp = likelihood * expanded_b
            updated_b = tmp / tmp.sum(dim=1, keepdim=True)
            
        return updated_b  # Shape: [batch_size, num_belief_points]

    def predict(self, s, deterministic=True):
        self.update_belief(s)
        pos = s['pos']
        obs = s['obs']

        H_b = self.get_entropy(self.belief_values)
        print(f'Entropy: {H_b}')

        # print(self.pomdp_bis.env_method.__getattribute__("set_deformation"))

        G = torch.zeros(self.pomdp_env.action_space.n, dtype=torch.float32,device=self.belief_values.device)
        deform = list(self.pomdp_env.transformation_matrix)
        for t, theta in enumerate(self.belief_points):
            self.pomdp_bis.env_method("set_deformation",[theta[0], theta[3]],[theta[1],theta[2]]) # stretch, shear format
            self.pomdp_bis.env_method("set_position",(pos.tolist()))
            
            s_prime, reward, done, info = self.pomdp_bis.step([i for i in range(4)])
            H_b_t_a_o = self.get_entropy(self.update_discrete_belief_tmp_batched(self.belief_values, s_prime))
            G += H_b_t_a_o * self.belief_values[t].to(H_b_t_a_o.device)
        
        return torch.argmax(H_b-G).item(), G

