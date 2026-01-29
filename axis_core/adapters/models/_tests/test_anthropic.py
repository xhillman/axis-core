"""Tests for AnthropicModel adapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from axis_core.protocols.model import ModelResponse

# Test that import provides helpful error message if anthropic not installed
try:
    from axis_core.adapters.models.anthropic import MODEL_PRICING, AnthropicModel
except ImportError as e:
    if "anthropic package" in str(e):
        pytest.skip("anthropic package not installed", allow_module_level=True)
    raise


@pytest.mark.unit
class TestAnthropicModel:
    """Test suite for AnthropicModel adapter."""

    def test_model_id_property(self) -> None:
        """Test that model_id property returns the configured model."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")
        assert model.model_id == "claude-sonnet-4-20250514"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom temperature and max_tokens."""
        model = AnthropicModel(
            model_id="claude-opus-4-20250514",
            api_key="test_key",
            temperature=0.5,
            max_tokens=2000,
        )
        assert model.model_id == "claude-opus-4-20250514"
        assert model._temperature == 0.5
        assert model._max_tokens == 2000

    def test_init_api_key_from_env(self) -> None:
        """Test that API key can be loaded from environment."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env_key"}):
            model = AnthropicModel(model_id="claude-sonnet-4-20250514")
            assert model._api_key == "env_key"

    def test_init_missing_api_key(self) -> None:
        """Test that missing API key raises helpful error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicModel(model_id="claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    async def test_complete_basic(self) -> None:
        """Test basic completion without tools."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hello! How can I help?")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)):  # noqa: E501
            response = await model.complete(
                messages=[{"role": "user", "content": "Hello"}],
                system="You are a helpful assistant.",
            )

        assert isinstance(response, ModelResponse)
        assert response.content == "Hello! How can I help?"
        assert response.tool_calls is None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self) -> None:
        """Test completion that returns tool calls."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "toolu_123"
        mock_tool_use.name = "search"
        mock_tool_use.input = {"query": "test"}

        mock_response = Mock()
        mock_response.content = [
            Mock(type="text", text="I'll search for that."),
            mock_tool_use,
        ]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = Mock(input_tokens=50, output_tokens=30)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)):  # noqa: E501
            response = await model.complete(
                messages=[{"role": "user", "content": "Search for test"}],
                tools=[{"name": "search", "description": "Search tool", "input_schema": {}}],
            )

        assert response.content == "I'll search for that."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "toolu_123"
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_complete_with_temperature_override(self) -> None:
        """Test that temperature can be overridden per request."""
        model = AnthropicModel(
            model_id="claude-sonnet-4-20250514",
            api_key="test_key",
            temperature=0.7,
        )

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=10)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:  # noqa: E501
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.2,
            )

            # Verify temperature was overridden
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens_override(self) -> None:
        """Test that max_tokens can be overridden per request."""
        model = AnthropicModel(
            model_id="claude-sonnet-4-20250514",
            api_key="test_key",
            max_tokens=1000,
        )

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=10)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:  # noqa: E501
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=500,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_stream_basic(self) -> None:
        """Test streaming completion."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        # Mock stream chunks
        mock_chunks = [
            Mock(type="content_block_delta", delta=Mock(type="text_delta", text="Hello")),
            Mock(type="content_block_delta", delta=Mock(type="text_delta", text=" world")),
            Mock(type="message_stop"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk

        with patch.object(model._client.messages, "stream") as mock_stream_ctx:
            mock_stream_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_stream(*[], **{}))
            mock_stream_ctx.return_value.__aexit__ = AsyncMock()

            chunks = []
            async for chunk in model.stream(messages=[{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

        # Verify we got content chunks
        assert len(chunks) >= 2
        assert any(c.content == "Hello" for c in chunks)
        assert any(c.content == " world" for c in chunks)
        assert chunks[-1].is_final

    def test_estimate_tokens_basic(self) -> None:
        """Test token estimation for text."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        # Test simple text
        tokens = model.estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 100  # Should be small for short text

        # Test longer text has more tokens
        long_text = "Hello " * 100
        long_tokens = model.estimate_tokens(long_text)
        assert long_tokens > tokens

    def test_estimate_tokens_empty(self) -> None:
        """Test token estimation for empty string."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")
        tokens = model.estimate_tokens("")
        assert tokens == 0

    def test_estimate_cost_sonnet(self) -> None:
        """Test cost estimation for Claude Sonnet."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)

        # Sonnet: $3/MTok input, $15/MTok output
        expected = (1000 / 1_000_000) * 3.00 + (500 / 1_000_000) * 15.00
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_estimate_cost_opus(self) -> None:
        """Test cost estimation for Claude Opus."""
        model = AnthropicModel(model_id="claude-opus-4-20250514", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)

        # Opus: $15/MTok input, $75/MTok output
        expected = (1000 / 1_000_000) * 15.00 + (500 / 1_000_000) * 75.00
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_estimate_cost_unknown_model(self) -> None:
        """Test cost estimation for unknown model returns 0."""
        model = AnthropicModel(model_id="unknown-model", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_error_classification_rate_limit(self) -> None:
        """Test that rate limit errors are classified as recoverable."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        # Mock rate limit error from Anthropic SDK
        from anthropic import RateLimitError

        # Create properly-formed error
        mock_response = Mock()
        mock_response.status_code = 429
        error = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

        with patch.object(
            model._client.messages,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            from axis_core.errors import ModelError

            with pytest.raises(ModelError) as exc_info:
                await model.complete(messages=[{"role": "user", "content": "Test"}])

            # Should be classified as recoverable
            assert exc_info.value.recoverable is True

    @pytest.mark.asyncio
    async def test_error_classification_auth_error(self) -> None:
        """Test that auth errors are classified as non-recoverable."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        from anthropic import AuthenticationError

        # Create properly-formed error
        mock_response = Mock()
        mock_response.status_code = 401
        error = AuthenticationError("Invalid API key", response=mock_response, body=None)

        with patch.object(
            model._client.messages,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            from axis_core.errors import ModelError

            with pytest.raises(ModelError) as exc_info:
                await model.complete(messages=[{"role": "user", "content": "Test"}])

            # Should be classified as non-recoverable
            assert exc_info.value.recoverable is False

    @pytest.mark.asyncio
    async def test_error_classification_bad_request(self) -> None:
        """Test that bad request errors are classified as non-recoverable."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        from anthropic import BadRequestError

        # Create properly-formed error
        mock_response = Mock()
        mock_response.status_code = 400
        error = BadRequestError("Invalid request", response=mock_response, body=None)

        with patch.object(
            model._client.messages,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            from axis_core.errors import ModelError

            with pytest.raises(ModelError) as exc_info:
                await model.complete(messages=[{"role": "user", "content": "Test"}])

            # Should be classified as non-recoverable
            assert exc_info.value.recoverable is False

    def test_model_pricing_table(self) -> None:
        """Test that MODEL_PRICING table has expected models."""
        assert "claude-opus-4-20250514" in MODEL_PRICING
        assert "claude-sonnet-4-20250514" in MODEL_PRICING
        assert "claude-haiku" in MODEL_PRICING

        # Verify pricing structure
        opus_pricing = MODEL_PRICING["claude-opus-4-20250514"]
        assert "input_per_mtok" in opus_pricing
        assert "output_per_mtok" in opus_pricing
        assert opus_pricing["input_per_mtok"] > 0
        assert opus_pricing["output_per_mtok"] > 0

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequences(self) -> None:
        """Test that stop sequences are passed through."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=10)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:  # noqa: E501
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                stop_sequences=["STOP", "END"],
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stop_sequences"] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_complete_extracts_exact_token_counts(self) -> None:
        """Test that exact token counts are extracted from response (AD-029)."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        # Mock response with specific token counts
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=123, output_tokens=456)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)):  # noqa: E501
            response = await model.complete(messages=[{"role": "user", "content": "Test"}])

        # Verify exact token counts (not estimated)
        assert response.usage.input_tokens == 123
        assert response.usage.output_tokens == 456
        assert response.usage.total_tokens == 579

    @pytest.mark.asyncio
    async def test_complete_with_metadata(self) -> None:
        """Test that metadata is passed through to the API."""
        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=10)

        with patch.object(model._client.messages, "create", new=AsyncMock(return_value=mock_response)) as mock_create:  # noqa: E501
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                metadata={"user_id": "test_user"},
            )

            call_kwargs = mock_create.call_args.kwargs
            assert "metadata" in call_kwargs
            assert call_kwargs["metadata"] == {"user_id": "test_user"}
