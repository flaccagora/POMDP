"""
Tests for all agent implementations.

This module contains tests that verify the basic functionality of all agent implementations
by running them in the environment for a fixed number of steps.
"""
import pytest
import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from src.agents import (
    Infotaxis,
    MLS,
    QMDP,
    TS
)
from src.config import AgentConfig, EnvironmentConfig, ObservationModelConfig
from src.environment import Grid, POMDPDeformedGridworld
from src.obs_model import (
    singleNN,
    cardinalNN
)

# Define which agents to test and their configurations
AGENTS_TO_TEST = [
    # {
    #     'name': 'Infotaxis',
    #     'class': Infotaxis,
    #     'base_agent': 'dqn',
    #     'requires_obs_model': True
    # },
    {
        'name': 'MLS',
        'class': MLS,
        'base_agent': 'ppo',
        'requires_obs_model': True
    },
    {
        'name': 'QMDP',
        'class': QMDP,
        'base_agent': 'dqn',
        'requires_obs_model': True
    },
    {
        'name': 'TS',
        'class': TS,
        'base_agent': 'ppo',
        'requires_obs_model': True
    }
]

@pytest.fixture
def env_config():
    """Create a basic environment configuration."""
    return EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4),
        step_size=0.1,
        observation_radius=1.0,
        shear_range=(-0.2, 0.2),
        stretch_range=(0.4, 1.0),
        seed=42
    )

@pytest.fixture
def agent_config():
    """Create a basic agent configuration."""
    return AgentConfig(
        discretization=10,
        update_method='discrete',
        n_particles=100,
        theta_dim=4,
        seed=42,
        debug=True,
        device='cpu',
    )

@pytest.fixture
def obs_model_config():
    """Create a basic observation model configuration."""
    return ObservationModelConfig(
        model_type='cardinal',
        latent_dim=128,
        condition_dim=4,
        seed=42
    )

@pytest.fixture
def env(env_config):
    """Create a POMDP environment instance."""
    return POMDPDeformedGridworld(env_config)

@pytest.fixture
def mdp_env():
    """Create a MDP environment instance."""
    return Grid()

@pytest.fixture
def dqn_agent(mdp_env):
    """Create a DQN agent for testing."""
    return DQN('MultiInputPolicy', mdp_env, verbose=0)

@pytest.fixture
def ppo_agent(mdp_env):
    """Create a PPO agent for testing."""
    return PPO('MultiInputPolicy', mdp_env, verbose=0)

@pytest.fixture
def obs_model(obs_model_config):
    """Create an observation model instance."""
    if obs_model_config.model_type == 'single':
        return singleNN()
    else:
        return cardinalNN()

def run_agent_steps(agent, env, num_steps=5):
    """
    Run an agent in the environment for a specified number of steps.
    
    Args:
        agent: The agent to test
        env: The environment to run in
        num_steps: Number of steps to run
        
    Returns:
        list: List of rewards received
    """
    rewards = []
    obs = env.reset()
    
    for _ in range(num_steps):
        action = agent.predict(env.get_state())
        if isinstance(action, tuple):
            action = action[0]  # Some agents return (action, info)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
            
        obs = next_obs
        
    return rewards

# def test_infotaxis_agent(env, agent_config, dqn_agent, obs_model):
#     """Test the Infotaxis agent implementation."""
#     agent = Infotaxis(dqn_agent, env, agent_config, obs_model)
#     rewards = run_agent_steps(agent, env)
    
#     assert len(rewards) > 0
#     assert all(isinstance(r, float) for r in rewards)

def test_mls_agent(env, agent_config, ppo_agent, obs_model):
    """Test the MLS agent implementation."""
    agent = MLS(ppo_agent, env, agent_config, obs_model)
    rewards = run_agent_steps(agent, env)
    
    assert len(rewards) > 0
    assert all(isinstance(r, float) for r in rewards)

def test_qmdp_agent(env, agent_config, dqn_agent, obs_model):
    """Test the QMDP agent implementation."""
    agent = QMDP(dqn_agent, env, agent_config, obs_model)
    rewards = run_agent_steps(agent, env)
    
    assert len(rewards) > 0
    assert all(isinstance(r, float) for r in rewards)

def test_ts_agent(env, agent_config, ppo_agent, obs_model):
    """Test the TS agent implementation."""
    agent = TS(ppo_agent, env, agent_config, obs_model)
    rewards = run_agent_steps(agent, env)
    
    assert len(rewards) > 0
    assert all(isinstance(r, float) for r in rewards)

def test_agent_reset(env, agent_config, dqn_agent, ppo_agent, obs_model):
    """Test that agents can be reset properly."""
    agents = []
    for a in AGENTS_TO_TEST:
        base_agent = dqn_agent if a['base_agent'] == 'dqn' else ppo_agent
        agents.append(a['class'](base_agent, env, agent_config, obs_model) if a['requires_obs_model'] else a['class'](base_agent, env, agent_config))
    
    for agent in agents:
        # Run first episode
        rewards1 = run_agent_steps(agent, env)
        
        # Reset agent and environment
        agent.reset()
        env.reset()
        
        # Run second episode
        rewards2 = run_agent_steps(agent, env)
        
        # Check that both episodes produced rewards
        assert len(rewards1) > 0
        assert len(rewards2) > 0

def test_agent_training_modes(env, agent_config, dqn_agent, ppo_agent, obs_model):
    """Test that agents handle training/evaluation modes correctly."""
    agents = []
    for a in AGENTS_TO_TEST:
        base_agent = dqn_agent if a['base_agent'] == 'dqn' else ppo_agent
        agents.append(a['class'](base_agent, env, agent_config, obs_model) if a['requires_obs_model'] else a['class'](base_agent, env, agent_config))
    
    for agent in agents:
        # Test training mode
        agent.train()
        assert agent.is_training
        
        # Test evaluation mode
        agent.eval()
        assert not agent.is_training

def test_agent_config_access(env, agent_config, dqn_agent, ppo_agent, obs_model):
    """Test that agents can access their configuration."""
    agents = []
    for a in AGENTS_TO_TEST:
        base_agent = dqn_agent if a['base_agent'] == 'dqn' else ppo_agent
        agents.append(a['class'](base_agent, env, agent_config, obs_model) if a['requires_obs_model'] else a['class'](base_agent, env, agent_config))
    
    for agent in agents:
        config = agent.get_config()
        assert isinstance(config, AgentConfig)
        assert config.discretization == agent_config.discretization
        assert config.update_method == agent_config.update_method
        assert config.n_particles == agent_config.n_particles
        assert config.theta_dim == agent_config.theta_dim 