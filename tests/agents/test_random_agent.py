"""
Tests for the RandomAgent class.
"""
import pytest
import numpy as np

from src.agents import RandomAgent
from src.config import AgentConfig
from src.environment import POMDPDeformedGridworld

def test_random_agent_initialization():
    """Test RandomAgent initialization."""
    config = AgentConfig()
    env = POMDPDeformedGridworld(config)    
    agent = RandomAgent(env, config)
    
    assert agent.num_actions == 4
    assert isinstance(agent.rng, np.random.RandomState)
    assert agent.is_training is True

def test_random_agent_predict():
    """Test RandomAgent prediction."""
    config = AgentConfig()
    env = POMDPDeformedGridworld(config)
    agent = RandomAgent(env, config)
    
    # Test multiple predictions
    state = {'observation': np.array([0, 0])}
    actions = []
    for _ in range(100):
        action, info = agent.predict(state)
        actions.append(action)
        assert isinstance(info, dict)
        assert len(info) == 0
    
    # Check that all actions are valid
    assert all(0 <= action < 4 for action in actions)
    
    # Check that predictions are deterministic with same seed
    env2 = POMDPDeformedGridworld(config)
    agent2 = RandomAgent(env2, config)
    actions2 = []
    for _ in range(100):
        action, _ = agent2.predict(state)
        actions2.append(action)
    assert actions == actions2

def test_random_agent_reset():
    """Test RandomAgent reset method."""
    config = AgentConfig()
    env = POMDPDeformedGridworld(config)
    agent = RandomAgent(env, config)
    
    # Reset should not raise any errors
    agent.reset()

def test_random_agent_train_eval():
    """Test RandomAgent train/eval modes."""
    config = AgentConfig()
    env = POMDPDeformedGridworld(config)
    agent = RandomAgent(env, config)
    
    assert agent.is_training is True
    
    agent.eval()
    assert agent.is_training is False
    
    agent.train()
    assert agent.is_training is True 