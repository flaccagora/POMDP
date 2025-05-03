import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import torch
import pygame
from collections import OrderedDict
from .cpp_env_continous import gridworld
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union

from src.environment import BaseEnvironment
from src.config import EnvironmentConfig

DEFAULT_OBSTACLES = [((0.14625, 0.3325), (0.565, 0.55625)), 
             ((0.52875, 0.5375), (0.7375, 0.84125)), 
             ((0.0, 0.00125), (0.01625, 0.99125)), 
             ((0.0075, 0.00125), (0.99875, 0.04)), 
             ((0.98875, 0.0075), (0.99875, 1.0)), 
             ((0.00125, 0.9825), (0.99875, 1.0))]

class ObservableDeformedGridworld(BaseEnvironment):
    """A deformable gridworld environment with partial observability."""
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__(config)
        
        # Initialize state variables
        self.state = None
        self.transformation_matrix = None
        self.inverse_transformation_matrix = None
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Initialize rendering if needed
        if self.config.render_mode == "human":
            self._setup_rendering()
            
    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "pos": spaces.Box(low=1.0, high=1.0, shape=(2,), dtype=float),
            "theta": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
        })
        
    def _setup_rendering(self):
        """Set up the rendering environment."""
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Deformable Gridworld Environment")
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        return {
            "pos": self.state,
            "theta": self.transformation_matrix.flatten()
        }
        
    def set_state(self, state: Dict[str, Any]):
        """Set the environment state."""
        self.state = state["pos"]
        self.transformation_matrix = state["theta"].reshape(2, 2)
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)
        
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Map actions to movements
        moves = [
            np.array([0, self.config.step_size]),   # Move up
            np.array([0, -self.config.step_size]),  # Move down
            np.array([self.config.step_size, 0]),   # Move right
            np.array([-self.config.step_size, 0])   # Move left
        ]
        
        # Get movement vector
        move = moves[action]
        
        # Update state
        next_state = self.state + move
        
        # Check for collisions and boundaries
        collision = self._check_collision(next_state)
        out_of_bounds = not self._is_in_bounds(next_state)
        reached_goal = self._is_at_goal(next_state)
        
        # Determine reward and termination
        if reached_goal:
            reward = 1.0
            terminated = True
            info = {"collision": False, "out": False, "goal": True}
        elif out_of_bounds:
            reward = -2.0
            terminated = False
            info = {"out": True}
            next_state = self.state
        elif collision:
            reward = -2.0
            terminated = False
            info = {"collision": True}
        else:
            reward = -0.5
            terminated = False
            info = {"collision": False, "out": False, "goal": False}
            
        # Update state
        self.state = next_state
        
        # Render if needed
        if self.config.render_mode == "human":
            self.render()
            
        return self.get_state(), reward, terminated, False, info
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to its initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        # Sample random deformation
        stretch = self._sample_range(self.config.stretch_range)
        shear = self._sample_range(self.config.shear_range)
        self.set_deformation(stretch, shear)
        
        # Sample random initial position
        self.state = self._sample_valid_position()
        
        return self.get_state(), {}
        
    def render(self):
        """Render the current state of the environment."""
        if self.config.render_mode != "human":
            return
            
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid and obstacles
        self._draw_grid()
        self._draw_obstacles()
        self._draw_agent()
        self._draw_goal()
        
        # Update display
        pygame.display.flip()
        
    def _check_collision(self, position: np.ndarray) -> bool:
        """Check if a position collides with any obstacle."""
        for obs in self.config.obstacles:
            if self._is_in_obstacle(position, obs):
                return True
        return False
        
    def _is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if a position is within the environment bounds."""
        return (0 <= position[0] <= self.config.grid_size[0] and 
                0 <= position[1] <= self.config.grid_size[1])
        
    def _is_at_goal(self, position: np.ndarray) -> bool:
        """Check if a position is at the goal."""
        return np.linalg.norm(position - self.config.goal) < self.config.observation_radius
        
    def _sample_range(self, range_tuple: Tuple[float, float]) -> np.ndarray:
        """Sample from a range."""
        low, high = range_tuple
        return low + np.random.rand(2) * (high - low)
        
    def _sample_valid_position(self) -> np.ndarray:
        """Sample a valid initial position."""
        while True:
            pos = np.random.rand(2) * self.config.grid_size
            if not self._check_collision(pos):
                return pos
                
    def _draw_grid(self):
        """Draw the grid lines."""
        # Implementation details...
        pass
        
    def _draw_obstacles(self):
        """Draw the obstacles."""
        # Implementation details...
        pass
        
    def _draw_agent(self):
        """Draw the agent."""
        # Implementation details...
        pass
        
    def _draw_goal(self):
        """Draw the goal."""
        # Implementation details...
        pass

class Grid(gridworld.ObservableDeformedGridworld,gym.Env):
   
    def __init__(self, grid_size=(1.0, 1.0), step_size=0.02, goal=(0.9, 0.9), obstacles=DEFAULT_OBSTACLES,
                  stretch=(1.0, 1.0), shear=(0.0, 0.0), observation_radius=0.05, shear_range=(-0.2,0.2), 
                  stretch_range=(0.4,1), render_mode=None,max_timesteps=500):
        super().__init__(grid_size, step_size, goal, obstacles, stretch, shear, observation_radius, shear_range, stretch_range)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  gym.spaces.Dict({
            "pos": gym.spaces.Box(low=.0, high=1.0, shape=(2,),dtype=float),
            "theta": gym.spaces.Box(low=.0, high=1.0, shape=(4,),dtype=float), # deformation is a 2x2 tensor
        })    
    
    def render(self):
        """
        Render the deformed gridworld environment along with the original gridworld.
        The original gridworld serves as a reference background.
        """
        import pygame  # Ensure Pygame is imported

        # Define colors
        WHITE = (255, 255, 255)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        PINK = (255, 105, 180)  
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)

        # Initialize the screen
        if not hasattr(self, "screen"):
            self.screen_width = 1000
            self.screen_height = 1000
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Deformed and Original Gridworld")

        # Fill background with white
        self.screen.fill(WHITE)

        # Compute the bounding box of the deformed grid
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        x_coords, y_coords = zip(*transformed_corners)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Define scaling factors to fit the deformed grid within the screen
        scale_x = self.screen_width / (max_x - min_x)
        scale_y = self.screen_height / (max_y - min_y)
        scale = min(scale_x, scale_y)  # Uniform scaling to maintain aspect ratio

        # Add upward translation offset
        y_translation = max(0, -min_y * scale)

        # Transform helper for rendering
        def to_screen_coords(pos):
            """
            Map transformed coordinates to screen coordinates, scaled and shifted to fit the screen.
            """
            x, y = pos
            x_screen = int((x - min_x) * scale)
            y_screen = int((max_y - y) * scale + y_translation)  # Flip y-axis and add upward translation
            return x_screen, y_screen
        
        # Draw the un-deformed grid (background)
        for i in range(int(self.grid_size[0]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((i, 0)),
                            to_screen_coords((i, self.grid_size[1])), width=1)
        for j in range(int(self.grid_size[1]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((0, j)),
                            to_screen_coords((self.grid_size[0], j)), width=1)

        # Draw the deformed grid boundaries
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        pygame.draw.polygon(self.screen, BLACK, [to_screen_coords(corner) for corner in transformed_corners], width=3)

        # Draw the obstacles in both grids
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            # Original obstacle
            pygame.draw.rect(self.screen, PINK,
                            (*to_screen_coords((x_min, y_max)),  # Top-left corner
                            int((x_max - x_min) * scale),      # Width
                            int((y_max - y_min) * scale)),    # Height
                            width=0)

            # Transformed obstacle
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            pygame.draw.polygon(self.screen, RED, [
                to_screen_coords(bottom_left),
                to_screen_coords(bottom_right),
                to_screen_coords(top_right),
                to_screen_coords(top_left)
            ])

        # Draw the agent in both grids
        agent_position = self.state
        transformed_agent_position = agent_position
        pygame.draw.circle(self.screen, BLUE, to_screen_coords(agent_position), 10)  # Original
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(transformed_agent_position), 10)  # Transformed

        # Draw the goal in both grids
        goal_position = self.goal
        transformed_goal_position = self.transform(goal_position)
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(goal_position), 12)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_goal_position), 12)  # Transformed

        # Draw observation radius as a dashed circle around the agent
        observation_radius = self.observation_radius # stays the same in both grids
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(agent_position), 
                        int(self.observation_radius * scale), 1)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_agent_position), 
                        int(observation_radius * scale), 1)  # Transformed

        # Update the display
        pygame.display.flip()

        # Handle key events
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'w' to quit
                elif event.key == pygame.K_w:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    return self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    return self.step(2)  # Right action
                elif event.key == pygame.K_UP:
                    return self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    return self.step(1)  # Down action
        return None, None, None, None, None
    
    def render_with_agentbelief(self, render_agent_belief=None):
        """
        Render the deformed gridworld environment along with the original gridworld.
        The original gridworld serves as a reference background.
        """
        import pygame  # Ensure Pygame is imported

        # Define colors
        WHITE = (255, 255, 255)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        PINK = (255, 105, 180)  
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)

        # Initialize the screen
        if not hasattr(self, "screen"):
            self.screen_width = 1000
            self.screen_height = 1000
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Deformed and Original Gridworld")

        # Fill background with white
        self.screen.fill(WHITE)

    # Compute the bounding box of the deformed grid
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        x_coords, y_coords = zip(*transformed_corners)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Define scaling factors to fit the deformed grid within the screen
        scale_x = self.screen_width / (max_x - min_x)
        scale_y = self.screen_height / (max_y - min_y)
        scale = min(scale_x, scale_y)  # Uniform scaling to maintain aspect ratio

        # Add upward translation offset
        y_translation = max(0, -min_y * scale)

        # Transform helper for rendering
        def to_screen_coords(pos):
            """
            Map transformed coordinates to screen coordinates, scaled and shifted to fit the screen.
            """
            x, y = pos
            x_screen = int((x - min_x) * scale)
            y_screen = int((max_y - y) * scale + y_translation)  # Flip y-axis and add upward translation
            return x_screen, y_screen
        
        # Draw the un-deformed grid (background)
        for i in range(int(self.grid_size[0]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((i, 0)),
                            to_screen_coords((i, self.grid_size[1])), width=1)
        for j in range(int(self.grid_size[1]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((0, j)),
                            to_screen_coords((self.grid_size[0], j)), width=1)

        # Draw the deformed grid boundaries
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        pygame.draw.polygon(self.screen, BLACK, [to_screen_coords(corner) for corner in transformed_corners], width=3)

        # Draw the obstacles in both grids
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            # Original (undeformed) obstacle
            pygame.draw.rect(self.screen, PINK,
                            (*to_screen_coords((x_min, y_max)),  # Top-left corner
                            int((x_max - x_min) * scale),      # Width
                            int((y_max - y_min) * scale)),    # Height
                            width=0)

            # Transformed obstacle
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            pygame.draw.polygon(self.screen, RED, [
                to_screen_coords(bottom_left),
                to_screen_coords(bottom_right),
                to_screen_coords(top_right),
                to_screen_coords(top_left)
            ])

            # Draw agent current belief deformation if requested
            if render_agent_belief is not None:
                bottom_left = self.transform(np.array([x_min, y_min]), render_agent_belief)
                bottom_right = self.transform(np.array([x_max, y_min]), render_agent_belief)
                top_left = self.transform(np.array([x_min, y_max]), render_agent_belief)
                top_right = self.transform(np.array([x_max, y_max]), render_agent_belief)
                pygame.draw.polygon(self.screen, BLACK, [
                    to_screen_coords(bottom_left),
                    to_screen_coords(bottom_right),
                    to_screen_coords(top_right),
                    to_screen_coords(top_left)
                ])
        
        # Draw the agent in both grids
        agent_position = self.state
        transformed_agent_position = agent_position
        pygame.draw.circle(self.screen, BLUE, to_screen_coords(agent_position), 10)  # Original
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(transformed_agent_position), 10)  # Transformed

        # Draw the goal in both grids
        goal_position = self.goal
        transformed_goal_position = self.transform(goal_position)
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(goal_position), 12)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_goal_position), 12)  # Transformed

        # Draw observation radius as a dashed circle around the agent
        observation_radius = self.observation_radius # stays the same in both grids
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(agent_position), 
                        int(self.observation_radius * scale), 1)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_agent_position), 
                        int(observation_radius * scale), 1)  # Transformed

        # Update the display
        pygame.display.flip()

        # Handle key events
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'w' to quit
                elif event.key == pygame.K_w:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    return self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    return self.step(2)  # Right action
                elif event.key == pygame.K_UP:
                    return self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    return self.step(1)  # Down action
        return None, None, None, None, None
    
    def reset(self, seed=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        
        state = OrderedDict({
            "pos": np.array(self.state),
            "theta": np.array(self.transformation_matrix).flatten(),
        })

        return state, {}
    
    def step(self, action):
        if self.render_mode == "human":
            self.render()

        return super().step(action)
      
    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()

    def create_obstacles_interactively(grid_size=(1.0, 1.0)):
        """
        Open a pygame window that allows the user to draw rectangular obstacles.
        The user can click and drag to create obstacles, press 'z' to undo the last obstacle,
        and press 'enter' to finish and return the list of obstacles.
        
        Returns:
            list: A list of obstacles in the format [((x_min, y_min), (x_max, y_max)), ...]
        """
        import pygame
        import numpy as np
        
        # Initialize pygame
        pygame.init()
        screen_width, screen_height = 1000, 1000
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Draw Obstacles - Click and drag to create, Z to undo, Enter to finish")
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        
        # Scale factors
        scale_x = screen_width / grid_size[0]
        scale_y = screen_height / grid_size[1]
        
        # Convert screen coordinates to grid coordinates
        def to_grid_coords(pos):
            x_screen, y_screen = pos
            x_grid = x_screen / scale_x
            y_grid = (screen_height - y_screen) / scale_y  # Flip y-axis
            return max(0, min(x_grid, grid_size[0])), max(0, min(y_grid, grid_size[1]))
        
        # Convert grid coordinates to screen coordinates
        def to_screen_coords(pos):
            x_grid, y_grid = pos
            x_screen = int(x_grid * scale_x)
            y_screen = int(screen_height - (y_grid * scale_y))  # Flip y-axis
            return x_screen, y_screen
        
        # Draw grid lines
        def draw_grid():
            screen.fill(WHITE)
            
            # Draw grid lines
            for i in range(int(grid_size[0]) + 1):
                x = int(i * scale_x)
                pygame.draw.line(screen, BLACK, (x, 0), (x, screen_height), 1)
            
            for j in range(int(grid_size[1]) + 1):
                y = int(screen_height - (j * scale_y))
                pygame.draw.line(screen, BLACK, (0, y), (screen_width, y), 1)
        
        # Function to draw existing obstacles
        def draw_obstacles():
            for obs in obstacles:
                (x_min, y_min), (x_max, y_max) = obs
                top_left = to_screen_coords((x_min, y_max))
                width = int((x_max - x_min) * scale_x)
                height = int((y_max - y_min) * scale_y)
                pygame.draw.rect(screen, RED, (top_left[0], top_left[1], width, height))
        
        # Show instructions
        font = pygame.font.SysFont(None, 24)
        
        def draw_instructions():
            instructions = [
                "Click and drag to create obstacles",
                "Press Z to undo the last obstacle",
                "Press Enter to finish",
                f"Current obstacles: {len(obstacles)}"
            ]
            
            for i, text in enumerate(instructions):
                text_surface = font.render(text, True, BLACK)
                screen.blit(text_surface, (10, 10 + i * 25))
        
        # Main loop
        running = True
        drawing = False
        start_pos = None
        current_rect = None
        obstacles = []
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        drawing = True
                        start_pos = to_grid_coords(event.pos)
                        current_rect = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if drawing:
                        current_pos = to_grid_coords(event.pos)
                        # Create rectangle coordinates
                        x_min = min(start_pos[0], current_pos[0])
                        y_min = min(start_pos[1], current_pos[1])
                        x_max = max(start_pos[0], current_pos[0])
                        y_max = max(start_pos[1], current_pos[1])
                        current_rect = ((x_min, y_min), (x_max, y_max))
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and drawing:
                        drawing = False
                        if current_rect:
                            (x_min, y_min), (x_max, y_max) = current_rect
                            # Only add if the rectangle has some size
                            if x_max > x_min and y_max > y_min:
                                obstacles.append(current_rect)
                        current_rect = None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:  # Undo
                        if obstacles:
                            obstacles.pop()
                    
                    elif event.key == pygame.K_RETURN:  # Finish
                        running = False
            
            # Draw background and grid
            draw_grid()
            
            # Draw existing obstacles
            draw_obstacles()
            
            # Draw current rectangle being created
            if drawing and current_rect:
                (x_min, y_min), (x_max, y_max) = current_rect
                top_left = to_screen_coords((x_min, y_max))
                width = int((x_max - x_min) * scale_x)
                height = int((y_max - y_min) * scale_y)
                pygame.draw.rect(screen, BLUE, (top_left[0], top_left[1], width, height), 2)
            
            # Draw instructions
            draw_instructions()
            
            # Update display
            pygame.display.flip()
        
        pygame.quit()
        return obstacles

class POMDPDeformedGridworld(Grid):
    def __init__(self, render_mode='human', obs_type = 'single'):
        super(POMDPDeformedGridworld, self).__init__(render_mode=render_mode)
        
        assert obs_type in ['single', 'cardinal']
        if obs_type == 'single':
            self.observe = super().is_collision
        elif obs_type == 'cardinal':
            self.observe = super().is_collision_cardinal

        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Discrete(2),
            'pos': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
    
    def reset(self, seed=None):
        state, _ = super().reset(seed=seed)
        pomdp_state = {
            'obs': torch.tensor(self.observe(state['pos']), dtype=torch.float32),
            'pos': torch.tensor(state['pos'], dtype=torch.float32)
            }
        return pomdp_state, {}
    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        pomdp_state = {
            'obs': torch.tensor(self.observe(state['pos']), dtype=torch.float32),
            'pos': torch.tensor(state['pos'], dtype=torch.float32)
            }
        return pomdp_state, reward, terminated, truncated, info
    
    def get_state(self):
        pomdp_state = {
            'obs': torch.tensor(self.observe(self.state), dtype=torch.float32),
            'pos': torch.tensor(self.state, dtype=torch.float32)
            }
        return pomdp_state
    
    def render_bis(self):
        """
        Render the deformed gridworld environment along with the original gridworld.
        The original gridworld serves as a reference background.
        """
        self.render = self.render_bis
        import pygame  # Ensure Pygame is imported

        # Define colors
        WHITE = (255, 255, 255)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        PINK = (255, 105, 180)  
        YELLOW = (255, 255, 0)
        BLACK = (0, 0, 0)

        # Initialize the screen
        if not hasattr(self, "screen"):
            self.screen_width = 1000
            self.screen_height = 1000
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Deformed and Original Gridworld")

        # Fill background with white
        self.screen.fill(WHITE)

    # Compute the bounding box of the deformed grid
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        x_coords, y_coords = zip(*transformed_corners)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Define scaling factors to fit the deformed grid within the screen
        scale_x = self.screen_width / (max_x - min_x)
        scale_y = self.screen_height / (max_y - min_y)
        scale = min(scale_x, scale_y)  # Uniform scaling to maintain aspect ratio

        # Add upward translation offset
        y_translation = max(0, -min_y * scale)

        # Transform helper for rendering
        def to_screen_coords(pos):
            """
            Map transformed coordinates to screen coordinates, scaled and shifted to fit the screen.
            """
            x, y = pos
            x_screen = int((x - min_x) * scale)
            y_screen = int((max_y - y) * scale + y_translation)  # Flip y-axis and add upward translation
            return x_screen, y_screen
        
        # Draw the un-deformed grid (background)
        for i in range(int(self.grid_size[0]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((i, 0)),
                            to_screen_coords((i, self.grid_size[1])), width=1)
        for j in range(int(self.grid_size[1]) + 1):
            pygame.draw.line(self.screen, LIGHT_GRAY,
                            to_screen_coords((0, j)),
                            to_screen_coords((self.grid_size[0], j)), width=1)

        # Draw the deformed grid boundaries
        corners = [
            np.array([0, 0]),
            np.array([self.grid_size[0], 0]),
            self.grid_size,
            np.array([0, self.grid_size[1]]),
        ]
        transformed_corners = [self.transform(corner) for corner in corners]
        pygame.draw.polygon(self.screen, BLACK, [to_screen_coords(corner) for corner in transformed_corners], width=3)

        # Draw the obstacles in both grids
        for obs in self.obstacles:
            (x_min, y_min), (x_max, y_max) = obs
            # Original obstacle
            # pygame.draw.rect(self.screen, PINK,
            #                 (*to_screen_coords((x_min, y_max)),  # Top-left corner
            #                 int((x_max - x_min) * scale),      # Width
            #                 int((y_max - y_min) * scale)),    # Height
            #                 width=0)

            # Transformed obstacle
            bottom_left = self.transform(np.array([x_min, y_min]))
            bottom_right = self.transform(np.array([x_max, y_min]))
            top_left = self.transform(np.array([x_min, y_max]))
            top_right = self.transform(np.array([x_max, y_max]))
            pygame.draw.polygon(self.screen, BLACK, [
                to_screen_coords(bottom_left),
                to_screen_coords(bottom_right),
                to_screen_coords(top_right),
                to_screen_coords(top_left)
            ])

        # Draw the agent in both grids
        agent_position = self.state
        transformed_agent_position = agent_position
        pygame.draw.circle(self.screen, BLUE, to_screen_coords(agent_position), 10)  # Original
        pygame.draw.circle(self.screen, RED, to_screen_coords(transformed_agent_position), 10)  # Transformed

        # Draw the goal in both grids
        goal_position = self.goal
        transformed_goal_position = self.transform(goal_position)
        # pygame.draw.circle(self.screen, GREEN, to_screen_coords(goal_position), 12)  # Original
        pygame.draw.circle(self.screen, GREEN, to_screen_coords(transformed_goal_position), 12)  # Transformed

        # Draw observation radius as a dashed circle around the agent
        observation_radius = self.observation_radius # stays the same in both grids
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(agent_position), 
                        int(self.observation_radius * scale), 1)  # Original
        pygame.draw.circle(self.screen, YELLOW, to_screen_coords(transformed_agent_position), 
                        int(observation_radius * scale), 4)  # Transformed

        # Update the display
        pygame.display.flip()

        # Handle key events
        # Handle key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Press 'r' to reset environment
                if event.key == pygame.K_r:
                    self.reset()
                # Press 'w' to quit
                elif event.key == pygame.K_w:
                    pygame.quit()
                    return
                # Press 's' to save current state
                elif event.key == pygame.K_s:
                    self.save_state()
                # Press space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.pause()
                # Press arrow keys for manual control
                elif event.key == pygame.K_LEFT:
                    return self.step(3)  # Left action
                elif event.key == pygame.K_RIGHT:
                    return self.step(2)  # Right action
                elif event.key == pygame.K_UP:
                    return self.step(0)  # Up action
                elif event.key == pygame.K_DOWN:
                    return self.step(1)  # Down action
        return None, None, None, None, None
    
class BeliefSpacePOMDP(POMDPDeformedGridworld):
    def __init__(self, obs_model, render_mode='human', obs_type = 'single', discretization=2000):
        super(BeliefSpacePOMDP, self).__init__(render_mode=render_mode, obs_type=obs_type)
        
        self.obs_model = obs_model
        
        from utils.belief import BayesianParticleFilter
        self.obs_model = obs_model
        self.n_particles = discretization
        self.PF = BayesianParticleFilter(f = obs_model, n_particles=self.n_particles, theta_dim=4)
        self.belief_points, self.belief_values = self.PF.initialize_particles()

        self.observation_space = Dict({
            'obs': Box(low=0, high=1, shape=(4,), dtype=np.int32),
            'pos': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'belief': Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })
 
    def reset(self, seed=None):
        pomdp_state, _ = super().reset(seed=seed)
        
        pomdp_state['belief'] = self.PF.estimate_posterior()[0]
        
        return pomdp_state, {}
    
    def step(self, action):
        pomdp_state, reward, terminated, truncated, info = super().step(action)
        
        self.belief_update()
        pomdp_state['belief'] = self.PF.estimate_posterior()[0]

        return pomdp_state, reward, terminated,truncated, info
    
    def belief_update(self):
        pomdp_state = self.get_state()
        X, y = pomdp_state['pos'].unsqueeze(0), pomdp_state['obs'].unsqueeze(0)

        self.PF.update(X, y)


if __name__ == "__main__":

    def load_obs_model(obs_type):
        from src.observation_model import singleNN, cardinalNN

        if obs_type == 'single':
            obs_model = singleNN()
            obs_model.load_state_dict(torch.load("observation_model/obs_model_4.pth", weights_only=True,map_location=torch.device('cpu')))
        elif obs_type == 'cardinal':
            obs_model = cardinalNN()
            obs_model.load_state_dict(torch.load("observation_model/obs_model_cardinal_4.pth", weights_only=True))
        else:
            raise ValueError("Observation type not recognized")
        
        return obs_model
    
    obs_model = load_obs_model('cardinal')

    env = BeliefSpacePOMDP(obs_model=obs_model, obs_type='cardinal', render_mode='rgb_array')

    # env = POMDPDeformedGridworld(obs_type='cardinal', render_mode='rgb_array')

    env.reset()
    print(env.step(int(0)))
    print(env.observation_space.sample())


