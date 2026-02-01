"""Tests for OpenAIModel adapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from axis_core.protocols.model import ModelResponse

# Test that import provides helpful error message if openai not installed
try:
    from axis_core.adapters.models.openai import MODEL_PRICING, OpenAIModel
except ImportError as e:
    if "openai package" in str(e):
        pytest.skip("openai package not installed", allow_module_level=True)
    raise


@pytest.mark.unit
class TestOpenAIModel:
    """Test suite for OpenAIModel adapter."""

    def test_model_id_property(self) -> None:
        """Test that model_id property returns the configured model."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")
        assert model.model_id == "gpt-4"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom temperature and max_tokens."""
        model = OpenAIModel(
            model_id="gpt-4o",
            api_key="test_key",
            temperature=0.5,
            max_tokens=2000,
        )
        assert model.model_id == "gpt-4o"
        assert model._temperature == 0.5
        assert model._max_tokens == 2000

    def test_init_api_key_from_env(self) -> None:
        """Test that API key can be loaded from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"}):
            model = OpenAIModel(model_id="gpt-4")
            assert model._api_key == "env_key"

    def test_init_missing_api_key(self) -> None:
        """Test that missing API key raises helpful error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIModel(model_id="gpt-4")

    @pytest.mark.asyncio
    async def test_complete_basic(self) -> None:
        """Test basic completion without tools."""
        model = OpenAIModel(model_id="gpt-4o", api_key="test_key")

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="Hello! How can I help?",
                    tool_calls=None,
                )
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
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
        """Test completion that returns tool calls (function calling)."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        # Mock tool call response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_function = Mock()
        mock_function.name = "search"
        mock_function.arguments = '{"query": "test"}'
        mock_tool_call.function = mock_function

        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I'll search for that.",
                    tool_calls=[mock_tool_call],
                )
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
        )

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            response = await model.complete(
                messages=[{"role": "user", "content": "Search for test"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search tool",
                            "parameters": {},
                        },
                    }
                ],
            )

        assert response.content == "I'll search for that."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_complete_with_temperature_override(self) -> None:
        """Test that temperature can be overridden per request."""
        model = OpenAIModel(
            model_id="gpt-4",
            api_key="test_key",
            temperature=0.7,
        )

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
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
        model = OpenAIModel(
            model_id="gpt-4o",
            api_key="test_key",
            max_tokens=1000,
        )

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=500,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_complete_uses_max_completion_tokens_for_gpt5(self) -> None:
        """Test that GPT-5 models use max_completion_tokens parameter."""
        model = OpenAIModel(
            model_id="gpt-5",
            api_key="test_key",
            max_tokens=1000,
        )

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=500,
            )

            call_kwargs = mock_create.call_args.kwargs
            # GPT-5 should use max_completion_tokens, not max_tokens
            assert "max_completion_tokens" in call_kwargs
            assert call_kwargs["max_completion_tokens"] == 500
            assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_basic(self) -> None:
        """Test streaming completion."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        # Mock stream chunks
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello", tool_calls=None))]),
            Mock(choices=[Mock(delta=Mock(content=" world", tool_calls=None))]),
            Mock(choices=[Mock(delta=Mock(content=None, tool_calls=None))]),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            model._client.chat.completions, "create", new=mock_stream
        ):
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
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

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
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")
        tokens = model.estimate_tokens("")
        assert tokens == 0

    def test_estimate_cost_gpt5(self) -> None:
        """Test cost estimation for GPT-5."""
        model = OpenAIModel(model_id="gpt-5", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)

        # GPT-5: $1.25/MTok input, $10/MTok output
        expected = (1000 / 1_000_000) * 1.25 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_estimate_cost_gpt4o(self) -> None:
        """Test cost estimation for GPT-4o."""
        model = OpenAIModel(model_id="gpt-4o", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)

        # GPT-4o: $2.50/MTok input, $10/MTok output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_estimate_cost_o1(self) -> None:
        """Test cost estimation for O1."""
        model = OpenAIModel(model_id="o1", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)

        # O1: $15/MTok input, $60/MTok output
        expected = (1000 / 1_000_000) * 15.00 + (500 / 1_000_000) * 60.00
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_estimate_cost_unknown_model(self) -> None:
        """Test cost estimation for unknown model returns 0."""
        model = OpenAIModel(model_id="unknown-model", api_key="test_key")

        cost = model.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_error_classification_rate_limit(self) -> None:
        """Test that rate limit errors are classified as recoverable."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        # Mock rate limit error from OpenAI SDK
        from openai import RateLimitError

        error = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body=None,
        )

        with patch.object(
            model._client.chat.completions,
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
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        from openai import AuthenticationError

        error = AuthenticationError(
            "Invalid API key",
            response=Mock(status_code=401),
            body=None,
        )

        with patch.object(
            model._client.chat.completions,
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
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        from openai import BadRequestError

        error = BadRequestError(
            "Invalid request",
            response=Mock(status_code=400),
            body=None,
        )

        with patch.object(
            model._client.chat.completions,
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
        assert "gpt-5" in MODEL_PRICING
        assert "gpt-4o" in MODEL_PRICING
        assert "o1" in MODEL_PRICING

        # Verify pricing structure
        gpt5_pricing = MODEL_PRICING["gpt-5"]
        assert "input_per_mtok" in gpt5_pricing
        assert "output_per_mtok" in gpt5_pricing
        assert gpt5_pricing["input_per_mtok"] > 0
        assert gpt5_pricing["output_per_mtok"] > 0

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequences(self) -> None:
        """Test that stop sequences are passed through."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                stop_sequences=["STOP", "END"],
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stop"] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_complete_extracts_exact_token_counts(self) -> None:
        """Test that exact token counts are extracted from response."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        # Mock response with specific token counts
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(
            prompt_tokens=123,
            completion_tokens=456,
            total_tokens=579,
        )

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            response = await model.complete(messages=[{"role": "user", "content": "Test"}])

        # Verify exact token counts (not estimated)
        assert response.usage.input_tokens == 123
        assert response.usage.output_tokens == 456
        assert response.usage.total_tokens == 579

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self) -> None:
        """Test that system parameter is converted to system message."""
        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

        with patch.object(
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            await model.complete(
                messages=[{"role": "user", "content": "Test"}],
                system="You are a helpful assistant.",
            )

            call_kwargs = mock_create.call_args.kwargs
            messages = call_kwargs["messages"]
            # First message should be system message
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."
            # Second message should be the user message
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Test"

    def test_convert_tool_manifest_to_openai(self) -> None:
        """Test conversion of ToolManifest to OpenAI format."""
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

        result = OpenAIModel._convert_tool_manifest_to_openai(manifest)

        assert isinstance(result, dict)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get the weather for a city"
        assert result["function"]["parameters"]["type"] == "object"
        assert "city" in result["function"]["parameters"]["properties"]

    def test_convert_tools_with_manifests(self) -> None:
        """Test _convert_tools_to_openai with ToolManifest objects."""
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

        result = OpenAIModel._convert_tools_to_openai(manifests)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["type"] == "function"
        assert result[1]["function"]["name"] == "tool2"

    def test_convert_tools_with_dicts(self) -> None:
        """Test _convert_tools_to_openai with raw dicts (backward compat)."""
        tool_dicts = [
            {
                "type": "function",
                "function": {"name": "tool1", "description": "First", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "tool2", "description": "Second", "parameters": {}},
            },
        ]

        result = OpenAIModel._convert_tools_to_openai(tool_dicts)

        # Should pass through unchanged
        assert result == tool_dicts

    def test_convert_tools_with_none(self) -> None:
        """Test _convert_tools_to_openai with None."""
        result = OpenAIModel._convert_tools_to_openai(None)
        assert result is None

    def test_convert_tools_with_empty_list(self) -> None:
        """Test _convert_tools_to_openai with empty list."""
        result = OpenAIModel._convert_tools_to_openai([])
        assert result is None

    @pytest.mark.asyncio
    async def test_complete_with_tool_manifests(self) -> None:
        """Test that complete() properly converts ToolManifest objects."""
        from axis_core.tool import ToolManifest

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response", tool_calls=None))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        model = OpenAIModel(model_id="gpt-4", api_key="test_key")

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
            model._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            await model.complete(
                messages=[{"role": "user", "content": "Hello"}],
                tools=manifests,
            )

            # Verify tools were converted and passed
            model._client.chat.completions.create.assert_called_once()
            call_kwargs = model._client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            tools_arg = call_kwargs["tools"]
            assert isinstance(tools_arg, list)
            assert len(tools_arg) == 1
            # Should be converted to OpenAI format dict
            assert isinstance(tools_arg[0], dict)
            assert tools_arg[0]["type"] == "function"
            assert tools_arg[0]["function"]["name"] == "get_weather"
