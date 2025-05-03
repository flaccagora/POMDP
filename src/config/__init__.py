from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch

@dataclass(frozen=True)
class BaseConfig:
    """Base configuration class for all components."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    debug: bool = False
    seed: int = 42

@dataclass(frozen=True)
class EnvironmentConfig(BaseConfig):
    """Configuration for environment settings."""
    grid_size: Tuple[float, float] = (1.0, 1.0)
    step_size: float = 0.02
    goal: Tuple[float, float] = (0.9, 0.9)
    observation_radius: float = 0.2
    render_mode: Optional[str] = None
    obstacles: List[Dict[str, Any]] = None
    stretch_range: Tuple[float, float] = (0.4, 1.0)
    shear_range: Tuple[float, float] = (-0.2, 0.2)
    stretch: Tuple[float, float] = (1.0, 1.0)
    shear: Tuple[float, float] = (0.0, 0.0)
    max_timesteps: int = 500

@dataclass(frozen=True)
class AgentConfig(BaseConfig):
    """Configuration for agent settings."""
    discretization: int = 10
    update_method: str = 'discrete'  # Options: 'discrete', 'discrete_exact', 'variational', 'particlefilters'
    n_particles: int = 10
    theta_dim: int = 4
    temperature: float = 1.0

@dataclass(frozen=True)
class ObservationModelConfig(BaseConfig):
    """Configuration for observation model settings."""
    model_type: str = 'cardinal'  # Options: 'single', 'cardinal'
    latent_dim: int = 128
    condition_dim: int = 4 