import yaml
from pathlib import Path


class Config:
    def __init__(self, relative_path="conf/config.yaml"):
        # Resolve path relative to this file, not the working directory
        base_dir = Path(__file__).resolve().parent
        config_path = (base_dir / relative_path).resolve()

        with open(config_path, "r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.data[key]


# Singleton pattern to avoid rereading the config every time
_config_instance = None


def get_config(relative_path="conf/config.yaml"):
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(relative_path)
    return _config_instance
