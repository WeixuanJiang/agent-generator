"""
Config Parser Module

Loads and validates configuration from config.json.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


PROJECT_ROOT = Path.cwd()

class ConfigParser:
    """
    Load and validate system configuration from config.json.

    Supports multiple AI frameworks and model providers with validation.
    """

    SUPPORTED_FRAMEWORKS = ["langchain", "langgraph", "strands", "crewai"]
    SUPPORTED_PROVIDERS = ["aws", "openai", "gemini", "anthropic"]
    SUPPORTED_LLMS = ["claude", "codex"]

    def __init__(self, config_file: str = str(PROJECT_ROOT / "data" / "config.json")):
        """
        Initialize ConfigParser.

        Args:
            config_file: Path to config.json file
        """
        self.config_file = config_file
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate configuration from config.json.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config.json doesn't exist
            ValueError: If configuration is invalid
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}\n"
                f"Please create a config.json file with your settings."
            )

        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.validate_config()

        # Apply defaults for optional keys
        self.config.setdefault("llm_runner", "claude")

        return self.config

    def validate_config(self):
        """
        Validate configuration structure and values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        # Check required top-level keys
        required_keys = ["framework", "model_provider"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate framework
        framework = self.config["framework"].lower()
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework: {framework}. "
                f"Supported frameworks: {', '.join(self.SUPPORTED_FRAMEWORKS)}"
            )

        # Validate LLM runner
        llm_runner = self.config.get("llm_runner", "claude").lower()
        if llm_runner not in self.SUPPORTED_LLMS:
            raise ValueError(
                f"Unsupported llm_runner: {llm_runner}. "
                f"Supported runtimes: {', '.join(self.SUPPORTED_LLMS)}"
            )

        # Validate model provider
        if not isinstance(self.config["model_provider"], dict):
            raise ValueError("model_provider must be a dictionary")

        provider_config = self.config["model_provider"]
        if "name" not in provider_config:
            raise ValueError("model_provider must have a 'name' field")

        provider_name = provider_config["name"].lower()
        if provider_name not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported model provider: {provider_name}. "
                f"Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

    def get_framework(self) -> str:
        """
        Get selected AI framework.

        Returns:
            Framework name (langchain, langgraph, strands, crewai)
        """
        if self.config is None:
            self.load_config()

        return self.config["framework"].lower()

    def get_model_provider(self) -> Dict[str, Any]:
        """
        Get model provider configuration.

        Returns:
            Model provider configuration dictionary
        """
        if self.config is None:
            self.load_config()

        return self.config["model_provider"]

    def get_provider_name(self) -> str:
        """
        Get model provider name.

        Returns:
            Provider name (aws, openai, gemini, anthropic)
        """
        provider_config = self.get_model_provider()
        return provider_config["name"].lower()

    def get_llm_runner(self, override: Optional[str] = None) -> str:
        """
        Get the selected LLM runtime (claude or codex).

        Args:
            override: Optional runtime override from CLI/env

        Returns:
            LLM runner identifier
        """
        if override:
            return override.lower()

        if self.config is None:
            self.load_config()

        return self.config.get("llm_runner", "claude").lower()

    def get_model_name(self) -> str:
        """
        Get model name/identifier.

        Returns:
            Model name
        """
        provider_config = self.get_model_provider()
        return provider_config.get("model", self._get_default_model())

    def _get_default_model(self) -> str:
        """
        Get default model based on provider.

        Returns:
            Default model name
        """
        provider = self.get_provider_name()

        defaults = {
            "aws": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "openai": "gpt-4-turbo-preview",
            "gemini": "gemini-pro",
            "anthropic": "claude-3-5-sonnet-20241022"
        }

        return defaults.get(provider, "unknown-model")

    def get_credentials(self) -> Dict[str, str]:
        """
        Get provider credentials.

        Resolves environment variable references (env:VAR_NAME format).

        Returns:
            Credentials dictionary
        """
        provider_config = self.get_model_provider()
        credentials = provider_config.get("credentials", {})

        # Resolve environment variables
        # Note: Since we're using Claude Code CLI for execution, it handles
        # authentication via its own .env file. We use placeholder values here.
        resolved_credentials = {}
        for key, value in credentials.items():
            if isinstance(value, str) and value.startswith("env:"):
                env_var = value[4:]  # Remove "env:" prefix
                resolved_value = os.environ.get(env_var)
                if resolved_value is None:
                    # Use placeholder - Claude Code CLI will handle authentication
                    resolved_credentials[key] = f"CLAUDE_CODE_WILL_HANDLE_{env_var}"
                    # Note: Claude Code CLI handles auth via .env, so this is expected
                else:
                    resolved_credentials[key] = resolved_value
            else:
                resolved_credentials[key] = value

        return resolved_credentials

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters (temperature, max_tokens, etc.).

        Returns:
            Model parameters dictionary
        """
        provider_config = self.get_model_provider()
        return provider_config.get("parameters", self._get_default_params())

    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default model parameters.

        Returns:
            Default parameters dictionary
        """
        return {
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0
        }

    def get_generation_settings(self) -> Dict[str, Any]:
        """
        Get generation settings.

        Returns:
            Generation settings dictionary
        """
        if self.config is None:
            self.load_config()

        return self.config.get("generation_settings", self._get_default_generation_settings())

    def get_fixed_mcps(self) -> Optional[list]:
        """
        Get a fixed list of MCP servers to use instead of auto-detection.

        Returns:
            List of MCP names or None if not provided
        """
        if self.config is None:
            self.load_config()

        mcps = self.config.get("mcp_servers") or self.config.get("mcps")
        if mcps is None:
            return None
        if not isinstance(mcps, list):
            raise ValueError("mcp_servers must be a list when provided")
        return mcps

    def _get_default_generation_settings(self) -> Dict[str, Any]:
        """
        Get default generation settings.

        Returns:
            Default generation settings
        """
        return {
            "max_iterations": 10,
            "enable_testing": True,
            "output_directory": str(PROJECT_ROOT / "outputs"),
            "verbose": True,
            "save_progress": True
        }

    def get_max_iterations(self) -> int:
        """
        Get maximum iteration limit for autonomous generation.

        Returns:
            Maximum iterations
        """
        settings = self.get_generation_settings()
        return settings.get("max_iterations", 10)

    def get_output_directory(self) -> str:
        """
        Get output directory for generated agents.

        Returns:
            Output directory path
        """
        settings = self.get_generation_settings()
        return settings.get("output_directory", str(PROJECT_ROOT / "outputs"))

    def is_testing_enabled(self) -> bool:
        """
        Check if testing is enabled.

        Returns:
            True if testing enabled, False otherwise
        """
        settings = self.get_generation_settings()
        return settings.get("enable_testing", True)

    def is_verbose(self) -> bool:
        """
        Check if verbose output is enabled.

        Returns:
            True if verbose, False otherwise
        """
        settings = self.get_generation_settings()
        return settings.get("verbose", True)

    def get_region(self) -> Optional[str]:
        """
        Get AWS region (for AWS Bedrock).

        Returns:
            AWS region or None
        """
        provider_config = self.get_model_provider()
        return provider_config.get("region")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        if self.config is None:
            self.load_config()

        return self.config

    def save_config(self, output_file: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            output_file: Output file path (defaults to original config_file)
        """
        if self.config is None:
            raise ValueError("No configuration to save")

        output_path = output_file or self.config_file

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

    def create_llm_config(self) -> Dict[str, Any]:
        """
        Create LLM configuration dictionary for framework initialization.

        Returns:
            LLM configuration suitable for framework use
        """
        provider = self.get_provider_name()
        model = self.get_model_name()
        params = self.get_model_params()
        credentials = self.get_credentials()

        config = {
            "provider": provider,
            "model": model,
            "parameters": params
        }

        # Add provider-specific configuration
        if provider == "aws":
            config["region"] = self.get_region() or "us-east-1"
            config["credentials"] = credentials

        elif provider == "openai":
            config["api_key"] = credentials.get("api_key")

        elif provider == "gemini":
            config["api_key"] = credentials.get("api_key")

        elif provider == "anthropic":
            config["api_key"] = credentials.get("api_key")

        return config


# Example usage
if __name__ == "__main__":
    parser = ConfigParser("data/config.json")

    try:
        config = parser.load_config()

        print("Configuration loaded successfully!")
        print(f"\nFramework: {parser.get_framework()}")
        print(f"Provider: {parser.get_provider_name()}")
        print(f"Model: {parser.get_model_name()}")
        print(f"Max Iterations: {parser.get_max_iterations()}")
        print(f"Output Directory: {parser.get_output_directory()}")
        print(f"Testing Enabled: {parser.is_testing_enabled()}")

        print("\nLLM Config:")
        print(json.dumps(parser.create_llm_config(), indent=2))

    except Exception as e:
        print(f"Error loading configuration: {e}")
