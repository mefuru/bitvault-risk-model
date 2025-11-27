"""
Configuration loading utilities.

Supports:
- Local .env file
- config.yaml
- Streamlit Cloud secrets (for cloud deployment)
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the project root directory (where src/ lives)."""
    # This file is at src/config.py, so parent is the project root
    return Path(__file__).parent.parent


def _get_streamlit_secret(key: str) -> Optional[str]:
    """
    Try to get a secret from Streamlit Cloud.
    Returns None if not running in Streamlit or secret not found.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None


def load_config() -> dict[str, Any]:
    """
    Load configuration from config.yaml and environment variables.
    
    Priority (highest to lowest):
    1. Streamlit Cloud secrets
    2. Environment variables
    3. config.yaml values
    """
    # Load .env file if it exists
    env_path = get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Load config.yaml (or use defaults if not found)
    config_path = get_project_root() / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        # Default configuration
        config = {
            "cryptoquant": {"api_key": ""},
            "fred": {"api_key": ""},
            "data": {"btc_source": "yahoo", "history_years": 3},
            "model": {"n_paths": 100000, "horizon_days": 30},
            "risk": {"margin_call_ltv": 0.85, "liquidation_ltv": 0.95},
        }
    
    # Ensure nested dicts exist
    config.setdefault("cryptoquant", {})
    config.setdefault("fred", {})
    
    # Override with Streamlit secrets (highest priority)
    streamlit_cryptoquant = _get_streamlit_secret("CRYPTOQUANT_API_KEY")
    if streamlit_cryptoquant:
        config["cryptoquant"]["api_key"] = streamlit_cryptoquant
    
    streamlit_fred = _get_streamlit_secret("FRED_API_KEY")
    if streamlit_fred:
        config["fred"]["api_key"] = streamlit_fred
    
    # Override with environment variables (second priority)
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
            f"Set {service.upper()}_API_KEY in environment, .env, or Streamlit secrets"
        )
    
    return key