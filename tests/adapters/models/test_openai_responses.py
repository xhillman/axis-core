"""Tests for OpenAIResponsesModel adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from axis_core.protocols.model import ModelResponse

try:
    from axis_core.adapters.models.openai_responses import OpenAIResponsesModel
except ImportError as e:
    if "openai package" in str(e):
        pytest.skip("openai package not installed", allow_module_level=True)
    raise


@pytest.mark.unit
class TestOpenAIResponsesModel:
    """Test suite for OpenAIResponsesModel adapter."""

    def test_model_id_property(self) -> None:
        """Model adapter should expose configured model id."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")
        assert model.model_id == "gpt-5-codex"

    @pytest.mark.asyncio
    async def test_complete_maps_text_and_usage(self) -> None:
        """Non-streaming completion should map Responses payload to ModelResponse."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")

        mock_usage = Mock(input_tokens=12, output_tokens=8, total_tokens=20)
        mock_response = Mock(output_text="hello from responses", output=[], usage=mock_usage)

        with patch.object(
            model._client.responses,
            "create",
            new=AsyncMock(return_value=mock_response),
        ) as mock_create:
            response = await model.complete(
                messages=[{"role": "user", "content": "hello"}],
                system="respond briefly",
            )

        assert isinstance(response, ModelResponse)
        assert response.content == "hello from responses"
        assert response.tool_calls is None
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 8
        assert response.usage.total_tokens == 20

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5-codex"
        assert call_kwargs["instructions"] == "respond briefly"
        assert isinstance(call_kwargs["input"], list)

    @pytest.mark.asyncio
    async def test_complete_maps_function_call_items(self) -> None:
        """Function-call output items should map back to ToolCall values."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")

        tool_item = SimpleNamespace(
            type="function_call",
            call_id="call_abc",
            name="search_docs",
            arguments='{"query":"axis"}',
            id="tool_item_1",
        )
        mock_usage = Mock(input_tokens=40, output_tokens=10, total_tokens=50)
        mock_response = Mock(output_text="", output=[tool_item], usage=mock_usage)

        with patch.object(
            model._client.responses,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await model.complete(
                messages=[{"role": "user", "content": "search for axis docs"}],
            )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc"
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "axis"}

    @pytest.mark.asyncio
    async def test_complete_handles_malformed_function_call_arguments(self) -> None:
        """Malformed JSON tool arguments should degrade to an empty arguments dict."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")

        tool_item = SimpleNamespace(
            type="function_call",
            call_id="call_abc",
            name="search_docs",
            arguments="{not valid json",
            id="tool_item_1",
        )
        mock_usage = Mock(input_tokens=40, output_tokens=10, total_tokens=50)
        mock_response = Mock(output_text="", output=[tool_item], usage=mock_usage)

        with patch.object(
            model._client.responses,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await model.complete(
                messages=[{"role": "user", "content": "search for axis docs"}],
            )

        assert response.tool_calls is not None
        assert response.tool_calls[0].arguments == {}

    def test_convert_messages_preserves_tool_call_and_tool_result_context(self) -> None:
        """Assistant tool calls and tool outputs should map to Responses input items."""
        messages = [
            {"role": "user", "content": "find docs"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "search_docs", "arguments": {"query": "axis core"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found 3 matches"},
        ]

        converted = OpenAIResponsesModel._convert_messages_to_responses_input(messages)
        assert converted[0]["type"] == "message"
        assert converted[1]["type"] == "function_call"
        assert converted[1]["call_id"] == "call_1"
        assert converted[2]["type"] == "function_call_output"
        assert converted[2]["call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_stream_emits_text_deltas_and_final_chunk(self) -> None:
        """Streaming should surface text deltas and emit a final chunk."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")

        async def mock_stream(*args, **kwargs):
            yield Mock(type="response.output_text.delta", delta="hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed")

        with patch.object(model._client.responses, "create", new=mock_stream):
            chunks = [
                chunk
                async for chunk in model.stream(messages=[{"role": "user", "content": "hi"}])
            ]

        assert len(chunks) == 3
        assert chunks[0].content == "hello"
        assert chunks[1].content == " world"
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_emits_partial_json_delta_for_function_calls(self) -> None:
        """Function-call argument deltas should set partial_json for lifecycle fallback."""
        model = OpenAIResponsesModel(model_id="gpt-5-codex", api_key="test_key")

        async def mock_stream(*args, **kwargs):
            yield Mock(type="response.function_call_arguments.delta", delta='{"query":"a')
            yield Mock(type="response.completed")

        with patch.object(model._client.responses, "create", new=mock_stream):
            chunks = [
                chunk
                async for chunk in model.stream(messages=[{"role": "user", "content": "hi"}])
            ]

        assert chunks[0].tool_call_delta == {"partial_json": '{"query":"a'}
        assert chunks[-1].is_final is True
