import json
import os
from typing import Any, Dict
from pathlib import Path


class Config:
    """Configuration loader and manager for the seed generation system."""

    _instance = None
    _config_data: Dict[str, Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config_data is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        config_path = Path(__file__).parent.parent / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config_data = json.load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'models.seed_generation')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config_data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def reload(self):
        """Reload configuration from file."""
        self._config_data = None
        self._load_config()

    @property
    def models(self) -> Dict[str, str]:
        """Get model configurations."""
        return self.get('models', {})

    @property
    def seed_generation(self) -> Dict[str, Any]:
        """Get seed generation configurations."""
        return self.get('seed_generation', {})

    @property
    def text_generation(self) -> Dict[str, Any]:
        """Get text generation configurations."""
        return self.get('text_generation', {})

    @property
    def multi_stage(self) -> Dict[str, Any]:
        """Get multi-stage generation configurations."""
        return self.get('multi_stage', {})

    @property
    def seed_pruning(self) -> Dict[str, Any]:
        """Get seed pruning configurations."""
        return self.get('seed_pruning', {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configurations."""
        return self.get('evaluation', {})
