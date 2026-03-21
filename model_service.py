import json
import logging
from typing import AsyncIterator
from uuid import UUID

from langchain_core.messages import AIMessage

from app.providers.base import get_provider
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceMessage,
    ChunkChoice,
    DeltaMessage,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)

logger = logging.getLogger(__name__)


def _extract_usage(message: AIMessage) -> UsageInfo:
    """Extract token usage from an AIMessage, normalizing across provider formats."""
    metadata = getattr(message, "response_metadata", {}) or {}
    usage_metadata = getattr(message, "usage_metadata", {}) or {}

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # OpenAI / Groq / Azure format: response_metadata.token_usage
    token_usage = metadata.get("token_usage", {})
    if token_usage:
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)

    # Anthropic format: response_metadata.usage
    elif "usage" in metadata:
        usage = metadata["usage"]
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

    # Google Gemini format: response_metadata.usage_metadata
    elif "usage_metadata" in metadata:
        usage = metadata["usage_metadata"]
        prompt_tokens = usage.get("prompt_token_count", 0)
        completion_tokens = usage.get("candidates_token_count", 0)
        total_tokens = usage.get("total_token_count", prompt_tokens + completion_tokens)

    # LangChain built-in usage_metadata on the message itself
    elif usage_metadata:
        prompt_tokens = usage_metadata.get("input_tokens", 0)
        completion_tokens = usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", prompt_tokens + completion_tokens)

    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _extract_finish_reason(message: AIMessage) -> str:
    """Extract and normalize the finish reason from an AIMessage."""
    metadata = getattr(message, "response_metadata", {}) or {}

    reason = metadata.get("finish_reason") or metadata.get("stop_reason") or "stop"

    # Normalize common variants
    reason_lower = str(reason).lower()
    if reason_lower in ("stop", "end_turn", "eos", "finish"):
        return "stop"
    if reason_lower in ("length", "max_tokens"):
        return "length"
    return reason_lower


async def _resolve_registry_config(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """If the request references a registry model, resolve the full config from DB."""
    registry_model_id = request.provider_config.get("registry_model_id")
    if not registry_model_id:
        return request

    from app.database import get_session
    from app.services.registry_service import get_decrypted_config

    async for session in get_session():
        config = await get_decrypted_config(session, UUID(str(registry_model_id)))

    if config is None:
        msg = f"Registry model {registry_model_id} not found"
        raise ValueError(msg)

    # Build merged provider_config from registry data
    provider_config = dict(config.get("provider_config", {}))
    provider_config["api_key"] = config["api_key"]
    if config.get("base_url"):
        provider_config["base_url"] = config["base_url"]
        # Azure needs azure_endpoint too
        if config["provider"] == "azure":
            provider_config.setdefault("azure_endpoint", config["base_url"])

    # Apply default params from registry (request params override)
    defaults = config.get("default_params", {})

    return ChatCompletionRequest(
        provider=config["provider"],
        model=config["model_name"],
        messages=request.messages,
        provider_config=provider_config,
        temperature=request.temperature if request.temperature is not None else defaults.get("temperature"),
        max_tokens=request.max_tokens if request.max_tokens is not None else defaults.get("max_tokens"),
        top_p=request.top_p if request.top_p is not None else defaults.get("top_p"),
        top_k=request.top_k if request.top_k is not None else defaults.get("top_k"),
        n=request.n,
        stream=request.stream,
        seed=request.seed,
        json_mode=request.json_mode,
        model_kwargs=request.model_kwargs or defaults.get("model_kwargs"),
        tools=request.tools,
    )


async def chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Process a non-streaming chat completion request."""
    request = await _resolve_registry_config(request)
    provider = get_provider(request.provider.value)

    model = provider.build_model(
        model=request.model,
        provider_config=request.provider_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        n=request.n,
        streaming=False,
        seed=request.seed,
        json_mode=request.json_mode,
        model_kwargs=request.model_kwargs,
    )

    # Bind tools if provided
    if request.tools:
        model = model.bind_tools(request.tools)

    messages = provider.build_messages([m.model_dump() for m in request.messages])

    ai_message = await provider.invoke(model, messages)

    content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
    usage = _extract_usage(ai_message)
    finish_reason = _extract_finish_reason(ai_message)

    # Extract tool calls from the AIMessage
    tool_calls_out: list[ToolCall] | None = None
    lc_tool_calls = getattr(ai_message, "tool_calls", None)
    if lc_tool_calls:
        tool_calls_out = []
        for tc in lc_tool_calls:
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            tool_calls_out.append(
                ToolCall(
                    id=tc_id,
                    function=ToolCallFunction(
                        name=tc_name,
                        arguments=json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args),
                    ),
                )
            )
        finish_reason = "tool_calls"

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=content or "",
                    tool_calls=tool_calls_out,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


async def chat_completion_stream(request: ChatCompletionRequest) -> AsyncIterator[str]:
    """Process a streaming chat completion request, yielding SSE-formatted strings."""
    request = await _resolve_registry_config(request)
    provider = get_provider(request.provider.value)

    model = provider.build_model(
        model=request.model,
        provider_config=request.provider_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        n=request.n,
        streaming=True,
        seed=request.seed,
        json_mode=request.json_mode,
        model_kwargs=request.model_kwargs,
    )

    # Bind tools if provided (mirrors the non-streaming path)
    if request.tools:
        model = model.bind_tools(request.tools)

    messages = provider.build_messages([m.model_dump() for m in request.messages])

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        model=request.model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream content chunks and accumulate for usage extraction
    accumulated = None
    finish_reason = "stop"
    async for chunk in provider.stream(model, messages):
        # Accumulate chunks so the final message carries usage metadata
        try:
            accumulated = chunk if accumulated is None else accumulated + chunk
        except TypeError:
            pass

        content = ""
        if hasattr(chunk, "content") and chunk.content:
            content = chunk.content
        elif isinstance(chunk, str):
            content = chunk

        if content:
            stream_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=content),
                    )
                ],
            )
            yield f"data: {stream_chunk.model_dump_json()}\n\n"

        # Stream tool call deltas so the client can reconstruct tool calls
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        if tool_call_chunks:
            tc_deltas = []
            for tc_chunk in tool_call_chunks:
                tc_dict: dict = {}
                if isinstance(tc_chunk, dict):
                    tc_dict = {
                        "index": tc_chunk.get("index", 0),
                        "id": tc_chunk.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc_chunk.get("name", ""),
                            "arguments": tc_chunk.get("args", ""),
                        },
                    }
                else:
                    tc_dict = {
                        "index": getattr(tc_chunk, "index", 0),
                        "id": getattr(tc_chunk, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc_chunk, "name", ""),
                            "arguments": getattr(tc_chunk, "args", ""),
                        },
                    }
                tc_deltas.append(tc_dict)

            tc_stream_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(tool_calls=tc_deltas),
                    )
                ],
            )
            yield f"data: {tc_stream_chunk.model_dump_json()}\n\n"
            finish_reason = "tool_calls"

    # If the accumulated message has tool_calls but none were streamed
    # (some providers only expose them on the final accumulated message),
    # send them now so the client can reconstruct them.
    if accumulated is not None:
        lc_tool_calls = getattr(accumulated, "tool_calls", None)
        if lc_tool_calls and finish_reason != "tool_calls":
            tc_list = []
            for tc in lc_tool_calls:
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tc_list.append({
                    "index": 0,
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args),
                    },
                })
            tc_final_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(tool_calls=tc_list),
                    )
                ],
            )
            yield f"data: {tc_final_chunk.model_dump_json()}\n\n"
            finish_reason = "tool_calls"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        model=request.model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=finish_reason,
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    # Extract and send usage from accumulated message
    if accumulated is not None:
        usage = _extract_usage(accumulated)
        if usage.prompt_tokens or usage.completion_tokens or usage.total_tokens:
            usage_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[],
                usage=usage,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"
