"""
LLM Configuration Models

Defines configuration for different LLM providers (OpenAI, Anthropic).
"""

from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Future: COHERE, GEMINI, etc.


class LLMConfig(SQLModel, table=True):
    """
    LLM provider configuration.
    
    Stores API credentials, model preferences, and provider-specific settings.
    Each configuration can be associated with specific users or be system-wide.
    """
    __tablename__ = "llm_configs"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Provider identification
    provider: LLMProvider = Field(
        description="LLM provider (openai, anthropic, etc.)"
    )
    
    # Model configuration
    model_name: str = Field(
        description="Specific model to use (e.g., 'gpt-4', 'claude-3-opus-20240229')",
        max_length=100
    )
    
    # API credentials (encrypted in production)
    api_key: str = Field(
        description="API key for the provider (should be encrypted)",
        max_length=500
    )
    
    api_base_url: Optional[str] = Field(
        default=None,
        description="Optional custom API endpoint",
        max_length=500
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate (None = provider default)",
        ge=1
    )
    
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # Feature flags
    supports_streaming: bool = Field(
        default=True,
        description="Whether this provider/model supports streaming responses"
    )
    
    supports_function_calling: bool = Field(
        default=False,
        description="Whether this provider/model supports function calling"
    )
    
    # Rate limiting
    max_requests_per_minute: Optional[int] = Field(
        default=None,
        description="Rate limit for this configuration"
    )
    
    # Priority and fallback
    priority: int = Field(
        default=0,
        description="Priority when multiple configs exist (higher = preferred)"
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether this configuration is currently active"
    )
    
    # Metadata
    name: str = Field(
        description="Friendly name for this configuration",
        max_length=100
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Optional description of this configuration"
    )
    
    # Association (nullable for system-wide configs)
    # TODO: Re-enable foreign key when users table is created
    # foreign_key="users.id",
    user_id: Optional[UUID] = Field(
        default=None,
        description="User this config belongs to (None = system-wide)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "openai",
                "model_name": "gpt-4-turbo-preview",
                "temperature": 0.7,
                "max_tokens": 4096,
                "supports_streaming": True,
                "name": "Default GPT-4 Config",
                "priority": 10,
                "is_active": True
            }
        }
    )


class LLMModelInfo(SQLModel):
    """
    Information about a specific LLM model (not a database table).
    
    Used for runtime model selection and validation.
    """
    provider: LLMProvider
    model_name: str
    context_window: int = Field(description="Maximum context window in tokens")
    cost_per_1k_input_tokens: float = Field(description="Cost in USD per 1K input tokens")
    cost_per_1k_output_tokens: float = Field(description="Cost in USD per 1K output tokens")
    supports_streaming: bool = True
    supports_function_calling: bool = False
    max_output_tokens: Optional[int] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "openai",
                "model_name": "gpt-4-turbo-preview",
                "context_window": 128000,
                "cost_per_1k_input_tokens": 0.01,
                "cost_per_1k_output_tokens": 0.03,
                "supports_streaming": True,
                "supports_function_calling": True,
                "max_output_tokens": 4096
            }
        }
    )


# Model registry (hardcoded for now, could be database-driven)
MODEL_REGISTRY: dict[str, LLMModelInfo] = {
    # OpenAI Models
    "gpt-4-turbo-preview": LLMModelInfo(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4-turbo-preview",
        context_window=128000,
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        supports_streaming=True,
        supports_function_calling=True,
        max_output_tokens=4096
    ),
    "gpt-4": LLMModelInfo(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        context_window=8192,
        cost_per_1k_input_tokens=0.03,
        cost_per_1k_output_tokens=0.06,
        supports_streaming=True,
        supports_function_calling=True,
        max_output_tokens=4096
    ),
    "gpt-3.5-turbo": LLMModelInfo(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        context_window=16385,
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0015,
        supports_streaming=True,
        supports_function_calling=True,
        max_output_tokens=4096
    ),
    
    # Anthropic Models
    "claude-3-opus-20240229": LLMModelInfo(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        context_window=200000,
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        supports_streaming=True,
        supports_function_calling=True,
        max_output_tokens=4096
    ),
    "claude-3-sonnet-20240229": LLMModelInfo(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        context_window=200000,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        supports_streaming=True,
        supports_function_calling=True,
        max_output_tokens=4096
    ),
    "claude-3-haiku-20240307": LLMModelInfo(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        context_window=200000,
        cost_per_1k_input_tokens=0.00025,
        cost_per_1k_output_tokens=0.00125,
        supports_streaming=True,
        supports_function_calling=False,
        max_output_tokens=4096
    ),
}


def get_model_info(model_name: str) -> Optional[LLMModelInfo]:
    """Get model information from registry."""
    return MODEL_REGISTRY.get(model_name)


def validate_model_config(config: LLMConfig) -> tuple[bool, Optional[str]]:
    """
    Validate an LLM configuration.
    
    Returns:
        (is_valid, error_message)
    """
    # Check if model exists in registry
    model_info = get_model_info(config.model_name)
    if not model_info:
        return False, f"Unknown model: {config.model_name}"
    
    # Check provider matches
    if model_info.provider != config.provider:
        return False, f"Model {config.model_name} belongs to {model_info.provider}, not {config.provider}"
    
    # Check max_tokens doesn't exceed model limit
    if config.max_tokens and model_info.max_output_tokens:
        if config.max_tokens > model_info.max_output_tokens:
            return False, f"max_tokens ({config.max_tokens}) exceeds model limit ({model_info.max_output_tokens})"
    
    # Check streaming support
    if config.supports_streaming and not model_info.supports_streaming:
        return False, f"Model {config.model_name} does not support streaming"
    
    return True, None
