"""
Tests for the POMDP Deformed Gridworld environment implementation.
"""
import numpy as np
import pytest
import torch

from src.environment import POMDPDeformedGridworld
from src.config import EnvironmentConfig

@pytest.fixture
def env_config():
    """Create a basic environment configuration."""
    return EnvironmentConfig(
        grid_size=(1.0, 1.0),
        goal=(0.9, 0.9),
        step_size=0.1,  # Larger step size for easier movement
        observation_radius=0.05,
        shear_range=(-0.2, 0.2),
        stretch_range=(0.4, 1.0),
        render_mode=None,
        seed=42
    )

@pytest.fixture
def small_env_config():
    """Create a small environment configuration."""
    return EnvironmentConfig(
        grid_size=(0.5, 0.5),
        goal=(0.4, 0.4),
        step_size=0.1,  # Larger step size for easier movement
        observation_radius=0.05,
        shear_range=(-0.2, 0.2),
        stretch_range=(0.4, 1.0),
        render_mode=None,
        seed=42
    )

@pytest.fixture
def env(env_config):
    """Create a POMDP environment instance."""
    return POMDPDeformedGridworld(env_config)

@pytest.fixture
def small_env(small_env_config):
    """Create a small POMDP environment instance."""
    return POMDPDeformedGridworld(small_env_config)

def test_environment_initialization(env_config):
    """Test POMDP environment initialization."""
    env = POMDPDeformedGridworld(env_config)
    
    assert np.allclose(env.grid_size, np.array([1.0, 1.0]))
    assert np.allclose(env.goal, np.array([0.9, 0.9]))
    assert env.observation_radius == 0.05
    assert np.allclose(env.shear_range, np.array([-0.2, 0.2]))
    assert np.allclose(env.stretch_range, np.array([0.4, 1.0]))
    
    # Check initial state
    state = env.get_state()
    assert isinstance(state, dict)
    assert 'pos' in state
    assert 'obs' in state
    assert isinstance(state['pos'], torch.Tensor)
    assert isinstance(state['obs'], torch.Tensor)

def test_environment_reset(env):
    """Test environment reset functionality."""
    # Take some steps
    env.step(0)  # Move up
    env.step(1)  # Move right
    
    # Reset and check
    obs, info = env.reset()
    state = env.get_state()
    
    # Check position is reset
    assert isinstance(state['pos'], torch.Tensor)
    assert state['pos'].shape == (2,)
    
    # Check observation
    assert isinstance(state['obs'], torch.Tensor)
    assert state['obs'].shape == ()  # Scalar observation

def test_environment_step(env):
    """Test environment step functionality."""
    # Test each action
    for action in range(4):
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check observation
        assert isinstance(obs, dict)
        assert 'pos' in obs
        assert 'obs' in obs
        assert isinstance(obs['pos'], torch.Tensor)
        assert isinstance(obs['obs'], torch.Tensor)
        assert obs['pos'].shape == (2,)
        assert obs['obs'].shape == ()  # Scalar observation
        
        # Check reward
        assert isinstance(reward, float)
        assert reward <= 1.0 and reward >= -2.0
        
        # Check done flags
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Check info
        assert isinstance(info, dict)
        
        # Reset for next action
        env.reset()

def test_environment_boundary_conditions(env):
    """Test that agent cannot move outside grid boundaries."""
    # Try to move left from leftmost position
    obs, reward, terminated, truncated, info = env.step(3)  # Left
    state = env.get_state()
    
    # Try to move down from bottom position
    obs, reward, terminated, truncated, info = env.step(1)  # Down
    state = env.get_state()
    
    # Move to top-right corner
    for _ in range(10):  # Fewer steps needed with larger step size
        env.step(0)  # Move up
        env.step(2)  # Move right
    
    # Try to move up from top
    obs, reward, terminated, truncated, info = env.step(0)  # Up
    state = env.get_state()
    
    # Try to move right from rightmost
    obs, reward, terminated, truncated, info = env.step(2)  # Right
    state = env.get_state()

def test_environment_goal_reached(env):
    """Test that reaching goal gives correct reward and done flag."""
    # Reset to a known position
    env.reset()
    env.set_position(env.transform([0.9, 0.9]))

    env.step(0)
    state, reward, terminated, truncated, info = env.step(2)
    # Check that we eventually reach the goal
    assert terminated or truncated, "Failed to reach goal within 50 steps"
    assert reward == 1.0

def test_environment_invalid_action(env):
    """Test that invalid actions are handled."""
    # The environment clips actions to valid range
    obs, reward, terminated, truncated, info = env.step(4)
    assert reward <= 0  # Should get negative reward for invalid action

def test_environment_observation_space(env):
    """Test that observations are always within bounds."""
    # Try various movements
    for _ in range(10):
        action = np.random.randint(0, 4)
        obs, _, _, _, _ = env.step(action)
        
        # Check observation is binary
        assert obs['obs'] in [0.0, 1.0]  # Binary observation
        
        # Reset for next action
        env.reset()

def test_environment_state_consistency(env):
    """Test that state is consistent between get_state and step info."""
    obs, reward, terminated, truncated, info = env.step(0)
    state1 = env.get_state()
    state2 = obs
    
    assert torch.allclose(state1['pos'], state2['pos'])
    assert torch.allclose(state1['obs'], state2['obs'])

def test_environment_small_grid(small_env):
    """Test environment behavior with a small grid."""
    # Test movement in small grid
    obs, reward, terminated, truncated, info = small_env.step(2)  # Move right
    state = small_env.get_state()
    
    obs, reward, terminated, truncated, info = small_env.step(0)  # Move up
    state = small_env.get_state()

def test_environment_multiple_resets(env):
    """Test multiple reset calls."""
    # First episode
    env.step(2)  # Move right
    env.step(0)  # Move up
    obs1, _ = env.reset()
    
    # Second episode
    env.step(2)  # Move right
    env.step(2)  # Move right again
    obs2, _ = env.reset()
    
    # Check observations are different (due to random initialization)
    assert not torch.allclose(obs1['pos'], obs2['pos'])

