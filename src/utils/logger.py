import os
import csv
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class Logger:
    """Enhanced logger for tracking experiments and debugging."""
    
    def __init__(self, log_dir: str, name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store logs
            name: Optional name for the logger
        """
        self.log_dir = Path(log_dir)
        self.name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file paths
        self.txt_path = self.log_dir / f"{self.name}_log.txt"
        self.csv_path = self.log_dir / f"{self.name}_performance.csv"
        self.json_path = self.log_dir / f"{self.name}_config.json"
        
        # Set up logging
        self._setup_logging()
        
        # Set up CSV writer
        self._setup_csv()
        
    def _setup_logging(self):
        """Set up the logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.txt_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.name)
        
    def _setup_csv(self):
        """Set up the CSV writer for performance metrics."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        fieldnames = ['timestamp', 'episode', 'reward', 'entropy', 'steps', 'success']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        with open(self.json_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Configuration saved to {self.json_path}")
        
    def log(self, message: str, level: str = 'info'):
        """
        Log a message with the specified level.
        
        Args:
            message: Message to log
            level: Logging level (debug, info, warning, error, critical)
        """
        log_func = getattr(self.logger, level.lower())
        log_func(message)
        
    def log_performance(self, episode: int, reward: float, entropy: float, steps: int, success: bool):
        """
        Log performance metrics for an episode.
        
        Args:
            episode: Episode number
            reward: Total reward
            entropy: Current entropy
            steps: Number of steps
            success: Whether the episode was successful
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'reward': reward,
            'entropy': entropy,
            'steps': steps,
            'success': success
        }
        self.writer.writerow(data)
        self.csv_file.flush()
        
        # Log summary
        self.logger.info(
            f"Episode {episode} - Reward: {reward:.2f}, "
            f"Entropy: {entropy:.2f}, Steps: {steps}, Success: {success}"
        )
        
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error with optional context.
        
        Args:
            error: Exception to log
            context: Optional context information
        """
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f"\nContext: {json.dumps(context, indent=2)}"
        self.logger.error(error_msg)
        
    def close(self):
        """Close all file handlers."""
        self.csv_file.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
