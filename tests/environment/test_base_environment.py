"""
Tests for the base environment class.
"""
import pytest
import numpy as np
from typing import Tuple, Dict, Any

from src.environment import BaseEnvironment
from src.config import EnvironmentConfig

class DummyEnvironment(BaseEnvironment):
    """A dummy environment for testing the base class."""
    
    def reset(self) -> np.ndarray:
        return np.array([0, 0])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return np.array([0, 0]), 0.0, False, {}
    
    def render(self) -> None:
        pass

class InvalidEnvironment(BaseEnvironment):
    """An invalid environment that doesn't implement all methods."""
    pass

def test_base_environment_subclass():
    """Test that a proper subclass can be instantiated."""
    config = EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4)
    )
    
    env = DummyEnvironment(config)
    assert isinstance(env, BaseEnvironment)
    assert env.config == config

def test_base_environment_abstract_methods():
    """Test that all abstract methods are properly defined."""
    config = EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4)
    )
    
    env = DummyEnvironment(config)
    
    # Test reset
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    
    # Test step
    obs, reward, done, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Test render
    env.render()  # Should not raise any errors 

def test_base_environment_invalid_subclass():
    """Test that an invalid subclass cannot be instantiated."""
    config = EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4)
    )
    
    with pytest.raises(TypeError):
        InvalidEnvironment(config)

def test_base_environment_config_immutability():
    """Test that environment config cannot be modified after initialization."""
    config = EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4)
    )
    
    env = DummyEnvironment(config)
    
    # Try to modify config
    with pytest.raises(AttributeError):
        env.config.grid_size = (3, 3)

def test_base_environment_abstract_methods_inheritance():
    """Test that abstract methods are properly inherited."""
    assert hasattr(DummyEnvironment, 'reset')
    assert hasattr(DummyEnvironment, 'step')
    assert hasattr(DummyEnvironment, 'render')
    
    # Check that methods are not abstract in DummyEnvironment
    assert not getattr(DummyEnvironment.reset, '__isabstractmethod__', False)
    assert not getattr(DummyEnvironment.step, '__isabstractmethod__', False)
    assert not getattr(DummyEnvironment.render, '__isabstractmethod__', False)

def test_base_environment_config_validation():
    """Test that environment validates its config."""
    # Test with invalid config type
    with pytest.raises(TypeError):
        DummyEnvironment({"grid_size": (5, 5)})  # Should be EnvironmentConfig
    
    # Test with None config
    with pytest.raises(TypeError):
        DummyEnvironment(None)

def test_base_environment_method_return_types():
    """Test that all methods return correct types."""
    config = EnvironmentConfig(
        grid_size=(5, 5),
        goal=(4, 4)
    )
    
    env = DummyEnvironment(config)
    
    # Test reset return type
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    
    # Test step return types
    obs, reward, done, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Test render return type
    result = env.render()
    assert result is None 