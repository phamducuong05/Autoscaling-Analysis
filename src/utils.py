import yaml
import os

def load_config(config_path="config/autoscaling_config.yaml"):
    """
    Load parameters from a YAML configuration file.
    If using absolute path, provide it directly. 
    Otherwise it looks relative to the project root.
    """
    # Simple check to find the file if called from different directories
    if not os.path.exists(config_path):
        # Try going up one level if called from src/
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        # Try going up two levels
        elif os.path.exists(os.path.join("..", "..", config_path)):
            config_path = os.path.join("..", "..", config_path)
            
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
