import yaml
import os

def load_config(config_path="config/autoscaling_config.yaml"):
    if not os.path.exists(config_path):
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        elif os.path.exists(os.path.join("..", "..", config_path)):
            config_path = os.path.join("..", "..", config_path)
            
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
