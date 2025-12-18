"""LLM Configuration Loader

This module provides functionality to load and manage LLM client configurations
from YAML files with environment variable substitution support.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def _expand_env_vars(value: Any, allow_missing: bool = False) -> Any:
    """
    Recursively expand environment variables in configuration values.

    Supports ${VAR_NAME} syntax for environment variable substitution.

    Args:
        value: Configuration value (can be dict, list, str, or other types)
        allow_missing: If True, keep ${VAR_NAME} as-is when variable is not set

    Returns:
        Value with environment variables expanded

    Raises:
        ValueError: If a referenced environment variable is not set and allow_missing=False
    """
    if isinstance(value, dict):
        return {k: _expand_env_vars(v, allow_missing) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item, allow_missing) for item in value]
    elif isinstance(value, str):
        # Find all ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        result = value
        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value is None:
                if not allow_missing:
                    raise ValueError(
                        f"Environment variable '{var_name}' is not set.\n"
                        f"Please set it in your shell:\n"
                        f"  export {var_name}=\"your-value\"\n"
                        f"Or add it to your ~/.bashrc or ~/.zshrc for persistence."
                    )
                # Keep the placeholder if missing variables are allowed
                continue
            result = result.replace(f"${{{var_name}}}", env_value)

        return result
    else:
        return value


def load_llm_config(config_path: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Load LLM configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, searches for:
                    1. configs/llm_config.yaml (in project root)
                    2. configs/llm_config.example.yaml (fallback)
        verbose: If True, print debug information about config loading

    Returns:
        Dictionary containing the configuration with environment variables expanded

    Raises:
        FileNotFoundError: If no configuration file is found
        ValueError: If configuration is invalid or environment variables are missing
    """
    if config_path is None:
        # Find project root (assuming this file is in src/finegrained_conf/config/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        if verbose:
            print(f"[LLM Config] Current file: {current_file}")
            print(f"[LLM Config] Project root: {project_root}")

        # Try to find config file
        config_path = project_root / "configs" / "llm_config.yaml"

        if verbose:
            print(f"[LLM Config] Looking for: {config_path}")
            print(f"[LLM Config] Exists: {config_path.exists()}")

        if not config_path.exists():
            # Fall back to example config
            config_path = project_root / "configs" / "llm_config.example.yaml"
            if verbose:
                print(f"[LLM Config] Falling back to: {config_path}")
                print(f"[LLM Config] Exists: {config_path.exists()}")

            if not config_path.exists():
                raise FileNotFoundError(
                    "No LLM configuration file found. "
                    "Please create configs/llm_config.yaml based on configs/llm_config.example.yaml"
                )
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if verbose:
        print(f"[LLM Config] Loading config from: {config_path}")

    # Load YAML file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Expand environment variables
    try:
        config = _expand_env_vars(config)
    except ValueError as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")

    return config


def get_model_config(model_name: Optional[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model (e.g., "gpt-4.1-nano-2025-04-14")
        config: Pre-loaded configuration dictionary. If None, loads from default location.

    Returns:
        Dictionary containing model-specific configuration merged with defaults
    """
    if config is None:
        config = load_llm_config()

    # Start with default configuration
    model_config = config.get('default', {}).copy()

    # Apply proxy settings if present
    if 'proxy' in config:
        model_config['proxy'] = config['proxy'].copy()

    if model_name is None:
        return model_config

    # Check for exact model match
    models_config = config.get('models', {})
    if model_name in models_config:
        model_config.update(models_config[model_name])
        return model_config

    # Check for pattern matches (e.g., "llama" matches any model containing "llama")
    for pattern, pattern_config in models_config.items():
        if pattern.lower() in model_name.lower():
            model_config.update(pattern_config)
            return model_config

    # No specific configuration found, return defaults
    return model_config


def apply_proxy_settings(config: Dict[str, Any]) -> None:
    """
    Apply proxy settings from configuration to environment variables.

    Args:
        config: Configuration dictionary containing optional 'proxy' section
    """
    if 'proxy' not in config:
        return

    proxy_config = config['proxy']

    if 'http_proxy' in proxy_config and proxy_config['http_proxy']:
        os.environ['HTTP_PROXY'] = proxy_config['http_proxy']

    if 'https_proxy' in proxy_config and proxy_config['https_proxy']:
        os.environ['HTTPS_PROXY'] = proxy_config['https_proxy']

    if 'no_proxy' in proxy_config and proxy_config['no_proxy']:
        os.environ['NO_PROXY'] = proxy_config['no_proxy']
