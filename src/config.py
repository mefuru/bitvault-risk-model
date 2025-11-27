"""
Configuration loading utilities.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the project root directory (where src/ lives)."""
    # This file is at src/config.py, so parent is the project root
    return Path(__file__).parent.parent


def load_config() -> dict[str, Any]:
    """
    Load configuration from config.yaml and environment variables.
    
    Environment variables take precedence over config file values for secrets.
    """
    project_root = get_project_root()
    
    # Load .env file if it exists
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Load config.yaml
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.exists():
        # Fall back to template
        config_path = project_root / "config" / "config.template.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Expected at {config_path}\n"
            f"Run: cp config/config.template.yaml config/config.yaml"
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv("CRYPTOQUANT_API_KEY"):
        config["cryptoquant"]["api_key"] = os.getenv("CRYPTOQUANT_API_KEY")
    
    if os.getenv("FRED_API_KEY"):
        config["fred"]["api_key"] = os.getenv("FRED_API_KEY")
    
    return config


def get_api_key(service: str) -> str:
    """
    Get API key for a service.
    
    Args:
        service: One of 'cryptoquant', 'fred'
    
    Returns:
        API key string
    
    Raises:
        ValueError: If API key not configured
    """
    config = load_config()
    
    if service == "cryptoquant":
        key = config.get("cryptoquant", {}).get("api_key", "")
    elif service == "fred":
        key = config.get("fred", {}).get("api_key", "")
    else:
        raise ValueError(f"Unknown service: {service}")
    
    if not key or key.startswith("your_"):
        raise ValueError(
            f"API key for {service} not configured. "
            f"Set {service.upper()}_API_KEY in .env or config/config.yaml"
        )
    
    return key
