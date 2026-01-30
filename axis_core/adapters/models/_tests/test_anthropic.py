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

    def test_convert_tool_manifest_to_anthropic(self) -> None:
        """Test conversion of ToolManifest to Anthropic format."""
        from axis_core.tool import ToolManifest

        manifest = ToolManifest(
            name="get_weather",
            description="Get the weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            output_schema={"type": "string"},
            capabilities=(),
        )

        result = AnthropicModel._convert_tool_manifest_to_anthropic(manifest)

        assert isinstance(result, dict)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get the weather for a city"
        assert result["input_schema"]["type"] == "object"
        assert "city" in result["input_schema"]["properties"]
        assert "city" in result["input_schema"]["required"]

    def test_convert_tools_with_manifests(self) -> None:
        """Test _convert_tools_to_anthropic with ToolManifest objects."""
        from axis_core.tool import ToolManifest

        manifests = [
            ToolManifest(
                name="tool1",
                description="First tool",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "string"},
                capabilities=(),
            ),
            ToolManifest(
                name="tool2",
                description="Second tool",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "string"},
                capabilities=(),
            ),
        ]

        result = AnthropicModel._convert_tools_to_anthropic(manifests)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"
        # Should be dicts now, not ToolManifest objects
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)

    def test_convert_tools_with_dicts(self) -> None:
        """Test _convert_tools_to_anthropic with raw dicts (backward compat)."""
        tool_dicts = [
            {"name": "tool1", "description": "First", "input_schema": {}},
            {"name": "tool2", "description": "Second", "input_schema": {}},
        ]

        result = AnthropicModel._convert_tools_to_anthropic(tool_dicts)

        # Should pass through unchanged
        assert result == tool_dicts

    def test_convert_tools_with_none(self) -> None:
        """Test _convert_tools_to_anthropic with None."""
        result = AnthropicModel._convert_tools_to_anthropic(None)
        assert result is None

    def test_convert_tools_with_empty_list(self) -> None:
        """Test _convert_tools_to_anthropic with empty list."""
        result = AnthropicModel._convert_tools_to_anthropic([])
        assert result is None

    @pytest.mark.asyncio
    async def test_complete_with_tool_manifests(self) -> None:
        """Test that complete() properly converts ToolManifest objects."""
        from axis_core.tool import ToolManifest

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        model = AnthropicModel(model_id="claude-sonnet-4-20250514", api_key="test_key")

        manifests = [
            ToolManifest(
                name="get_weather",
                description="Get weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
                output_schema={"type": "string"},
                capabilities=(),
            )
        ]

        with patch.object(
            model._client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            await model.complete(
                messages=[{"role": "user", "content": "Hello"}],
                tools=manifests,
            )

            # Verify tools were converted and passed as dicts
            model._client.messages.create.assert_called_once()
            call_kwargs = model._client.messages.create.call_args[1]
            assert "tools" in call_kwargs
            tools_arg = call_kwargs["tools"]
            assert isinstance(tools_arg, list)
            assert len(tools_arg) == 1
            # Should be converted to dict, not ToolManifest
            assert isinstance(tools_arg[0], dict)
            assert tools_arg[0]["name"] == "get_weather"

    def test_convert_messages_simple_user_message(self) -> None:
        """Test that simple user messages pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_messages_simple_assistant_message(self) -> None:
        """Test that simple assistant messages pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_convert_messages_tool_role_to_user_with_tool_result(self) -> None:
        """Test that role='tool' messages are converted to user with tool_result."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "tool", "tool_call_id": "tc_123", "content": "Sunny, 72°F"},
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"  # Original user message

        # Tool result should be converted to user with tool_result content block
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "tc_123"
        assert result[1]["content"][0]["content"] == "Sunny, 72°F"

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        """Test that assistant messages with tool_calls are converted to content blocks."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_123", "name": "get_weather", "arguments": {"city": "SF"}}
                ],
            },
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 2

        # Assistant message should have content as array with text and tool_use blocks
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert len(assistant_msg["content"]) == 2

        # First block should be text
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][0]["text"] == "Let me check."

        # Second block should be tool_use
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["id"] == "tc_123"
        assert assistant_msg["content"][1]["name"] == "get_weather"
        assert assistant_msg["content"][1]["input"] == {"city": "SF"}

    def test_convert_messages_full_tool_conversation(self) -> None:
        """Test conversion of a full tool use conversation."""
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "content": "I'll check the weather.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "arguments": {"city": "San Francisco"}}
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "Sunny, 72°F"},
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 3

        # User message unchanged
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What's the weather in SF?"

        # Assistant with tool_use blocks
        assert result[1]["role"] == "assistant"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][1]["type"] == "tool_use"
        assert result[1]["content"][1]["id"] == "tc_1"

        # Tool result as user message with tool_result block
        assert result[2]["role"] == "user"
        assert isinstance(result[2]["content"], list)
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["tool_use_id"] == "tc_1"

    def test_convert_messages_multiple_tool_results_batched(self) -> None:
        """Test that multiple consecutive tool results are batched into one user message."""
        messages = [
            {"role": "user", "content": "Get weather for two cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "arguments": {"city": "SF"}},
                    {"id": "tc_2", "name": "get_weather", "arguments": {"city": "NYC"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "Sunny, 72°F"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "Cloudy, 55°F"},
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        assert len(result) == 3  # user, assistant, user (with batched tool results)

        # Both tool results should be in a single user message
        tool_results_msg = result[2]
        assert tool_results_msg["role"] == "user"
        assert isinstance(tool_results_msg["content"], list)
        assert len(tool_results_msg["content"]) == 2

        assert tool_results_msg["content"][0]["type"] == "tool_result"
        assert tool_results_msg["content"][0]["tool_use_id"] == "tc_1"
        assert tool_results_msg["content"][1]["type"] == "tool_result"
        assert tool_results_msg["content"][1]["tool_use_id"] == "tc_2"

    def test_convert_messages_assistant_tool_calls_no_text(self) -> None:
        """Test assistant message with tool calls but no text content."""
        messages = [
            {"role": "user", "content": "Get the weather"},
            {
                "role": "assistant",
                "content": "",  # Empty text
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "arguments": {"city": "SF"}}
                ],
            },
        ]
        result = AnthropicModel._convert_messages_to_anthropic(messages)

        # Assistant message should only have tool_use block, no empty text block
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert len(assistant_msg["content"]) == 1  # Only tool_use, no text
        assert assistant_msg["content"][0]["type"] == "tool_use"
