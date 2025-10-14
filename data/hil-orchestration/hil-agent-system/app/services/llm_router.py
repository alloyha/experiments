"""
LLM Router Service

Routes requests to appropriate LLM providers (OpenAI, Anthropic).
Handles streaming, retries, and fallback logic.
"""

import asyncio
from typing import AsyncGenerator, Optional
from uuid import UUID

import anthropic
import openai
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_config import (
    LLMConfig,
    LLMProvider,
    get_model_info,
    validate_model_config,
)

logger = structlog.get_logger()


class LLMRouterError(Exception):
    """Base exception for LLM routing errors."""
    pass


class ProviderUnavailableError(LLMRouterError):
    """Raised when no providers are available."""
    pass


class ModelNotFoundError(LLMRouterError):
    """Raised when specified model is not found."""
    pass


class LLMRouterService:
    """
    Service for routing LLM requests to appropriate providers.
    
    Features:
    - Multi-provider support (OpenAI, Anthropic)
    - Streaming responses
    - Automatic retries with exponential backoff
    - Provider fallback
    - Token counting and validation
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize LLM Router.
        
        Args:
            session: Database session for fetching configurations
        """
        self.session = session
        self.logger = logger.bind(service="llm_router")
    
    async def get_config(
        self,
        config_id: Optional[UUID] = None,
        provider: Optional[LLMProvider] = None,
        user_id: Optional[UUID] = None
    ) -> LLMConfig:
        """
        Get LLM configuration.
        
        Args:
            config_id: Specific configuration ID
            provider: Filter by provider
            user_id: Filter by user (None = system-wide)
        
        Returns:
            LLMConfig instance
        
        Raises:
            ModelNotFoundError: If no suitable config found
        """
        query = select(LLMConfig).where(LLMConfig.is_active == True)
        
        if config_id:
            query = query.where(LLMConfig.id == config_id)
        
        if provider:
            query = query.where(LLMConfig.provider == provider)
        
        if user_id is not None:
            # Prefer user-specific, fallback to system-wide
            query = query.where(
                (LLMConfig.user_id == user_id) | (LLMConfig.user_id == None)
            )
        else:
            # System-wide only
            query = query.where(LLMConfig.user_id == None)
        
        # Order by priority (higher first)
        query = query.order_by(LLMConfig.priority.desc())
        
        result = await self.session.execute(query)
        config = result.scalars().first()
        
        if not config:
            self.logger.error(
                "no_llm_config_found",
                config_id=str(config_id) if config_id else None,
                provider=provider,
                user_id=str(user_id) if user_id else None
            )
            raise ModelNotFoundError("No active LLM configuration found")
        
        return config
    
    async def generate_response(
        self,
        messages: list[dict],
        config_id: Optional[UUID] = None,
        provider: Optional[LLMProvider] = None,
        user_id: Optional[UUID] = None,
        stream: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate LLM response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            config_id: Specific configuration to use
            provider: Provider preference
            user_id: User making the request
            stream: Whether to stream response
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        
        Returns:
            Complete response string or async generator for streaming
        
        Raises:
            ProviderUnavailableError: If all providers fail
            LLMRouterError: For other errors
        """
        config = await self.get_config(config_id, provider, user_id)
        
        # Validate configuration
        is_valid, error = validate_model_config(config)
        if not is_valid:
            self.logger.error("invalid_llm_config", config_id=str(config.id), error=error)
            raise LLMRouterError(f"Invalid configuration: {error}")
        
        # Check streaming support
        if stream and not config.supports_streaming:
            self.logger.warning(
                "streaming_not_supported",
                config_id=str(config.id),
                model=config.model_name
            )
            stream = False
        
        self.logger.info(
            "generating_llm_response",
            provider=config.provider,
            model=config.model_name,
            stream=stream,
            message_count=len(messages)
        )
        
        # Route to appropriate provider
        try:
            if config.provider == LLMProvider.OPENAI:
                return await self._generate_openai(
                    config, messages, stream, max_retries, timeout
                )
            elif config.provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic(
                    config, messages, stream, max_retries, timeout
                )
            else:
                raise LLMRouterError(f"Unsupported provider: {config.provider}")
        
        except Exception as e:
            self.logger.error(
                "llm_generation_failed",
                provider=config.provider,
                model=config.model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _generate_openai(
        self,
        config: LLMConfig,
        messages: list[dict],
        stream: bool,
        max_retries: int,
        timeout: float
    ) -> str | AsyncGenerator[str, None]:
        """Generate response using OpenAI."""
        client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
            timeout=timeout
        )
        
        params = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "stream": stream
        }
        
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
        
        if config.top_p:
            params["top_p"] = config.top_p
        
        # Retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                if stream:
                    return self._stream_openai(client, params)
                else:
                    response = await client.chat.completions.create(**params)
                    return response.choices[0].message.content
            
            except openai.RateLimitError as e:
                last_error = e
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                self.logger.warning(
                    "openai_rate_limit",
                    attempt=attempt + 1,
                    wait_time=wait_time,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                continue
            
            except openai.APITimeoutError as e:
                last_error = e
                self.logger.warning(
                    "openai_timeout",
                    attempt=attempt + 1,
                    timeout=timeout,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                continue
            
            except openai.APIError as e:
                # Don't retry on client errors (4xx)
                if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                    self.logger.error("openai_client_error", error=str(e))
                    raise LLMRouterError(f"OpenAI API error: {str(e)}")
                
                last_error = e
                self.logger.warning(
                    "openai_api_error",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        # All retries failed
        raise ProviderUnavailableError(f"OpenAI unavailable after {max_retries} attempts: {last_error}")
    
    async def _stream_openai(
        self,
        client: openai.AsyncOpenAI,
        params: dict
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI."""
        async def _generator():
            try:
                stream = await client.chat.completions.create(**params)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                self.logger.error("openai_streaming_error", error=str(e))
                raise
        
        return _generator()
    
    async def _generate_anthropic(
        self,
        config: LLMConfig,
        messages: list[dict],
        stream: bool,
        max_retries: int,
        timeout: float
    ) -> str | AsyncGenerator[str, None]:
        """Generate response using Anthropic."""
        client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.api_base_url,
            timeout=timeout
        )
        
        # Convert messages to Anthropic format
        # Anthropic requires system message separate from messages
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        params = {
            "model": config.model_name,
            "messages": anthropic_messages,
            "temperature": config.temperature,
            "stream": stream
        }
        
        if system_message:
            params["system"] = system_message
        
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
        else:
            # Anthropic requires max_tokens
            params["max_tokens"] = 4096
        
        if config.top_p:
            params["top_p"] = config.top_p
        
        # Retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                if stream:
                    return self._stream_anthropic(client, params)
                else:
                    response = await client.messages.create(**params)
                    return response.content[0].text
            
            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = min(2 ** attempt, 60)
                self.logger.warning(
                    "anthropic_rate_limit",
                    attempt=attempt + 1,
                    wait_time=wait_time,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                continue
            
            except anthropic.APITimeoutError as e:
                last_error = e
                self.logger.warning(
                    "anthropic_timeout",
                    attempt=attempt + 1,
                    timeout=timeout,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                continue
            
            except anthropic.APIError as e:
                # Don't retry on client errors
                if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                    self.logger.error("anthropic_client_error", error=str(e))
                    raise LLMRouterError(f"Anthropic API error: {str(e)}")
                
                last_error = e
                self.logger.warning(
                    "anthropic_api_error",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        # All retries failed
        raise ProviderUnavailableError(f"Anthropic unavailable after {max_retries} attempts: {last_error}")
    
    async def _stream_anthropic(
        self,
        client: anthropic.AsyncAnthropic,
        params: dict
    ) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic."""
        async def _generator():
            try:
                async with client.messages.stream(**params) as stream:
                    async for text in stream.text_stream:
                        yield text
            except Exception as e:
                self.logger.error("anthropic_streaming_error", error=str(e))
                raise
        
        return _generator()
    
    async def health_check(self, config_id: UUID) -> dict:
        """
        Check health of LLM provider connection.
        
        Args:
            config_id: Configuration to test
        
        Returns:
            Health status dict
        """
        try:
            config = await self.get_config(config_id=config_id)
            
            # Simple ping test with minimal message
            test_messages = [
                {"role": "user", "content": "Hi"}
            ]
            
            response = await self.generate_response(
                messages=test_messages,
                config_id=config_id,
                stream=False,
                max_retries=1,
                timeout=10.0
            )
            
            return {
                "status": "healthy",
                "provider": config.provider,
                "model": config.model_name,
                "response_received": bool(response)
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
