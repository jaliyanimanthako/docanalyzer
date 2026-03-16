import yaml
from pathlib import Path

def load_config(config_path):
    """Loads a YAML configuration file safely."""
    with open(config_path, 'r') as file:
        try:
            # safe_load prevents execution of arbitrary code within the YAML
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML: {exc}")
            return None