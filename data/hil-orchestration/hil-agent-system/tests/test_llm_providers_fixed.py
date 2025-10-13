"""
Fixed tests for LLM providers that actually test the real interfaces.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.llm_providers import (
    AnthropicProvider,
    LLMProvider,
    LLMProviderFactory,
    MockLLMProvider,
    OpenAIProvider,
)


class TestLLMProviderFactory:
    """Test cases for LLM Provider Factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        provider = LLMProviderFactory.create("mock", "test-model")
        assert isinstance(provider, MockLLMProvider)
        assert provider.model == "test-model"

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = LLMProviderFactory.create("openai", "gpt-4")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = LLMProviderFactory.create("anthropic", "claude-3")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3"

    def test_create_invalid_provider(self):
        """Test creating invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create("invalid", "model")

    def test_list_providers(self):
        """Test listing available providers."""
        providers = LLMProviderFactory.list_providers()
        assert "mock" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_register_provider(self):
        """Test registering new provider."""

        class CustomProvider(LLMProvider):
            async def complete(self, messages, **kwargs):
                return "custom"

            async def complete_structured(self, messages, schema=None, **kwargs):
                return {"custom": True}

        LLMProviderFactory.register_provider("custom", CustomProvider)
        providers = LLMProviderFactory.list_providers()
        assert "custom" in providers

        # Test creating the custom provider
        provider = LLMProviderFactory.create("custom", "test-model")
        assert isinstance(provider, CustomProvider)


class TestMockLLMProvider:
    """Test cases for Mock LLM Provider."""

    @pytest.mark.asyncio
    async def test_complete_basic(self):
        """Test basic completion."""
        provider = MockLLMProvider("test-model")
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.complete(messages)
        assert result == "Mock LLM response for testing"

    @pytest.mark.asyncio
    async def test_complete_structured_basic(self):
        """Test structured completion without schema."""
        provider = MockLLMProvider("test-model")
        messages = [{"role": "user", "content": "What's your intent?"}]

        result = await provider.complete_structured(messages)
        expected = {
            "intent": "mock_intent",
            "confidence": 0.95,
            "entities": {"test": "value"},
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_complete_structured_with_schema(self):
        """Test structured completion with schema."""
        provider = MockLLMProvider("test-model")
        messages = [{"role": "user", "content": "Analyze this"}]
        schema = {
            "properties": {
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
                "valid": {"type": "boolean"},
                "count": {"type": "integer"},
                "metadata": {"type": "object"},
                "tags": {"type": "array"},
            }
        }

        result = await provider.complete_structured(messages, schema=schema)
        assert result["intent"] == "mock_intent"
        assert result["confidence"] == 0.95
        assert result["valid"] is True
        assert result["count"] == 1
        assert result["metadata"] == {"test": "value"}
        assert result["tags"] == ["test"]


class TestOpenAIProvider:
    """Test cases for OpenAI Provider."""

    @pytest.mark.asyncio
    async def test_complete_basic(self):
        """Test basic completion."""
        provider = OpenAIProvider("gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.complete(messages)
        assert result == "OpenAI response (not implemented)"

    @pytest.mark.asyncio
    async def test_complete_structured_basic(self):
        """Test structured completion."""
        provider = OpenAIProvider("gpt-4")
        messages = [{"role": "user", "content": "What's your intent?"}]

        result = await provider.complete_structured(messages)
        expected = {"intent": "openai_mock_intent", "confidence": 0.92, "entities": {}}
        assert result == expected


class TestAnthropicProvider:
    """Test cases for Anthropic Provider."""

    @pytest.mark.asyncio
    async def test_complete_basic(self):
        """Test basic completion."""
        provider = AnthropicProvider("claude-3")
        messages = [{"role": "user", "content": "Hello"}]

        result = await provider.complete(messages)
        assert result == "Anthropic response (not implemented)"

    @pytest.mark.asyncio
    async def test_complete_structured_basic(self):
        """Test structured completion."""
        provider = AnthropicProvider("claude-3")
        messages = [{"role": "user", "content": "What's your intent?"}]

        result = await provider.complete_structured(messages)
        expected = {
            "intent": "anthropic_mock_intent",
            "confidence": 0.88,
            "entities": {},
        }
        assert result == expected


class TestLLMProviderBase:
    """Test cases for base LLM Provider functionality."""

    def test_provider_initialization(self):
        """Test provider initialization with config."""
        provider = MockLLMProvider("test-model", api_key="test", timeout=30)
        assert provider.model == "test-model"
        assert provider.config["api_key"] == "test"
        assert provider.config["timeout"] == 30
