"""
Configuration management for the ABSA Engine.

We use Pydantic's BaseSettings to load configuration from:
1. Environment variables (highest priority)
2. The .env file (if present)
3. Default values defined below

This keeps sensitive data (API keys, URLs) out of the source code
and makes it easy to change behaviour per environment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application-level configuration loaded from environment / .env file.

    Attributes:
        ENVIRONMENT:  Runtime environment (development, staging, production).
        OLLAMA_BASE_URL:  Base URL for the local Ollama inference server.
        LOG_LEVEL:  Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    """

    # ── Runtime environment ──────────────────────────────────────────────
    ENVIRONMENT: str = "development"

    # ── External service URLs ────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"

    # Load values from .env file (placed next to this module).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Prevent setting fields that are not defined above.
        extra="forbid",
    )


# Single re-usable instance so config is parsed once at startup.
settings = AppConfig()
