"""HTTP client for the Model microservice.

Bridges agentcore backend to the standalone Model microservice by
proxying registry CRUD, chat completions, and embedding requests.

Also provides ``MicroserviceChatModel`` and ``MicroserviceEmbeddings`` —
LangChain-compatible proxy objects that the flow engine can use exactly
like any other ``BaseChatModel`` / ``Embeddings`` instance.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List
from uuid import UUID

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from agentcore.base.models.model import LCModelNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _get_model_service_settings() -> tuple[str, str]:
    """Get Model service URL and API key from agentcore settings."""
    from agentcore.services.deps import get_settings_service

    settings = get_settings_service().settings
    url = getattr(settings, "model_service_url", "")
    api_key = getattr(settings, "model_service_api_key", "")

    if not url:
        msg = "MODEL_SERVICE_URL is not configured. Set it in your environment or .env file."
        raise ValueError(msg)

    return url.rstrip("/"), api_key or ""


def _headers(api_key: str) -> dict[str, str]:
    """Build standard headers for Model service requests."""
    h = {"Content-Type": "application/json"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def is_service_configured() -> bool:
    """Check whether the Model service URL is configured (non-empty)."""
    try:
        _get_model_service_settings()
        return True
    except (ValueError, Exception):
        return False


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _safe_str(value) -> str | None:
    """Convert a value to a plain string, handling SecretStr and None."""
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    s = str(value)
    return s if s else None


def _safe_int(value) -> int | None:
    if value is None or value == "" or value == 0:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain BaseMessage list to OpenAI-format dicts.

    Preserves multimodal content (text + image_url) for user messages
    so that vision-capable models can process images.
    """
    result = []
    for msg in messages:
        # Preserve list content (multimodal: text + images) as-is for the API.
        # Only stringify if it's not already a string or list.
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = msg.content  # Keep structured content for multimodal
        else:
            content = str(msg.content)

        if isinstance(msg, SystemMessage):
            # System messages must be string content
            result.append({"role": "system", "content": msg.content if isinstance(msg.content, str) else str(msg.content)})
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                "tool_call_id": msg.tool_call_id,
            })
        elif isinstance(msg, AIMessage):
            entry: dict = {"role": "assistant", "content": msg.content if isinstance(msg.content, str) else str(msg.content)}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            "arguments": json.dumps(
                                tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        else:
            # User messages: preserve multimodal content (text + image_url)
            result.append({"role": "user", "content": content})
    return result


# ---------------------------------------------------------------------------
# Provider detection for component-based invocations
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, str] = {
    "AzureChatOpenAIComponent": "azure",
    "AzureOpenAIModel": "azure",
    "GroqModel": "groq",
    "GoogleGenerativeAI": "google",
    "GoogleGenerativeAIComponent": "google",
    "GoogleGenerativeAIModel": "google",
}


def _detect_provider(component: LCModelNode) -> str:
    """Infer the provider name from a component's class or name attribute."""
    class_name = type(component).__name__

    if class_name in ("RegistryModelComponent", "LanguageModelComponent"):
        return _parse_registry_provider(component)

    if class_name in _PROVIDER_MAP:
        return _PROVIDER_MAP[class_name]

    component_name = getattr(component, "name", "")
    if component_name in _PROVIDER_MAP:
        return _PROVIDER_MAP[component_name]

    class_lower = class_name.lower()
    if "azure" in class_lower:
        return "azure"
    if "groq" in class_lower:
        return "groq"
    if "google" in class_lower or "gemini" in class_lower:
        return "google"

    msg = f"Cannot detect provider for component class '{class_name}'"
    raise ValueError(msg)


def _parse_registry_provider(component: LCModelNode) -> str:
    """Get the provider string for a registry-based component."""
    provider = getattr(component, "provider", "")
    if provider and provider not in ("All", ""):
        # Map UI labels to DB keys
        from agentcore.components.models.registry_model import PROVIDER_LABEL_TO_KEY
        return PROVIDER_LABEL_TO_KEY.get(provider, provider)
    return "openai"  # placeholder — overridden by _resolve_registry_config in microservice


def _parse_registered_model_id(component: LCModelNode) -> str | None:
    """Extract the registry model UUID from the component's selected value."""
    selected = getattr(component, "registry_model", "") or ""
    if not selected:
        return None
    parts = [p.strip() for p in selected.split("|")]
    if len(parts) >= 3:
        return parts[2]
    return None


def _build_request_payload(
    component: LCModelNode,
    messages: list[BaseMessage],
    stream: bool = False,
) -> dict:
    """Build the full request payload for the Model microservice."""
    provider = _detect_provider(component)

    # Check if this is a registry-based component
    registry_model_id = _parse_registered_model_id(component)

    provider_config: dict = {}
    model_name = ""

    if registry_model_id:
        provider_config["registry_model_id"] = registry_model_id
        # Extract model name from selection
        selected = getattr(component, "registry_model", "") or ""
        parts = [p.strip() for p in selected.split("|")]
        model_name = parts[1] if len(parts) >= 2 else ""
    else:
        api_key = _safe_str(getattr(component, "api_key", None))
        if api_key:
            provider_config["api_key"] = api_key
        model_name = _safe_str(getattr(component, "model_name", None)) or ""

    payload: dict = {
        "provider": provider,
        "model": model_name,
        "messages": _messages_to_dicts(messages),
        "provider_config": provider_config,
        "stream": stream,
    }

    temperature = _safe_float(getattr(component, "temperature", None))
    if temperature is not None:
        payload["temperature"] = temperature

    max_tokens = _safe_int(getattr(component, "max_tokens", None))
    if max_tokens:
        payload["max_tokens"] = max_tokens

    return payload


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


async def invoke_via_service(
    component: LCModelNode,
    messages: list[BaseMessage],
) -> AIMessage:
    """Invoke the Model microservice and return an AIMessage with response_metadata."""
    url, api_key = _get_model_service_settings()
    payload = _build_request_payload(component, messages, stream=False)

    logger.debug("Model service request: provider=%s model=%s", payload.get("provider"), payload.get("model"))

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{url}/v1/chat/completions",
            headers=_headers(api_key),
            json=payload,
        )
        response.raise_for_status()

    data = response.json()

    content = ""
    finish_reason = "stop"
    if data.get("choices"):
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

    usage = data.get("usage", {})
    response_metadata = {
        "token_usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "model_name": data.get("model", payload.get("model", "")),
        "finish_reason": finish_reason,
    }

    return AIMessage(content=content, response_metadata=response_metadata)


async def stream_via_service(
    component: LCModelNode,
    messages: list[BaseMessage],
) -> AsyncIterator[str]:
    """Stream from the Model microservice, yielding content strings."""
    url, api_key = _get_model_service_settings()
    payload = _build_request_payload(component, messages, stream=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            headers=_headers(api_key),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def embed_via_service(
    provider: str,
    model: str,
    texts: list[str],
    provider_config: dict | None = None,
    dimensions: int | None = None,
    registry_model_id: str | None = None,
) -> list[list[float]]:
    """Generate embeddings via the Model microservice."""
    url, api_key = _get_model_service_settings()

    config = dict(provider_config or {})
    if registry_model_id:
        config["registry_model_id"] = registry_model_id

    payload: dict = {
        "provider": provider,
        "model": model,
        "input": texts,
        "provider_config": config,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{url}/v1/embeddings",
            headers=_headers(api_key),
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    return [item["embedding"] for item in data.get("data", [])]


# ---------------------------------------------------------------------------
# Registry proxy functions (for backend API to forward to microservice)
# ---------------------------------------------------------------------------


def fetch_registry_models(
    provider: str | None = None,
    environment: str | None = None,
    model_type: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Fetch registry models from the Model microservice (sync)."""
    try:
        url, api_key = _get_model_service_settings()
    except ValueError:
        return []

    params: dict = {}
    if provider:
        params["provider"] = provider
    if environment:
        params["environment"] = environment
    if model_type:
        params["model_type"] = model_type
    if not active_only:
        params["active_only"] = "false"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{url}/v1/registry/models",
                headers=_headers(api_key),
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch registry models from Model service: %s", e)
        return []


async def fetch_registry_models_async(
    provider: str | None = None,
    environment: str | None = None,
    model_type: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Fetch registry models from the Model microservice (async)."""
    try:
        url, api_key = _get_model_service_settings()
    except ValueError:
        return []

    params: dict = {}
    if provider:
        params["provider"] = provider
    if environment:
        params["environment"] = environment
    if model_type:
        params["model_type"] = model_type
    if not active_only:
        params["active_only"] = "false"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{url}/v1/registry/models",
                headers=_headers(api_key),
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch registry models from Model service: %s", e)
        return []


async def create_registry_model_via_service(body: dict) -> dict:
    """Create a registry model via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{url}/v1/registry/models",
            headers=_headers(api_key),
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_decrypted_model_config(model_id: str) -> dict | None:
    """Fetch decrypted model config (with API key) from the Model microservice."""
    try:
        url, api_key = _get_model_service_settings()
    except ValueError:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{url}/v1/registry/models/{model_id}/config",
                headers=_headers(api_key),
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch decrypted model config from Model service: %s", e)
        return None


async def get_registry_model_via_service(model_id: str) -> dict | None:
    """Get a registry model by ID via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{url}/v1/registry/models/{model_id}",
            headers=_headers(api_key),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


async def update_registry_model_via_service(model_id: str, body: dict) -> dict | None:
    """Update a registry model via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{url}/v1/registry/models/{model_id}",
            headers=_headers(api_key),
            json=body,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


async def delete_registry_model_via_service(model_id: str) -> bool:
    """Delete a registry model via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            f"{url}/v1/registry/models/{model_id}",
            headers=_headers(api_key),
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True


async def test_connection_via_service(body: dict) -> dict:
    """Test a model connection via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{url}/v1/registry/test-connection",
            headers=_headers(api_key),
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def test_embedding_connection_via_service(body: dict) -> dict:
    """Test an embedding connection via the Model microservice."""
    url, api_key = _get_model_service_settings()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{url}/v1/registry/test-embedding-connection",
            headers=_headers(api_key),
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


def fetch_models_from_service(provider: str, provider_config: dict) -> list[str]:
    """Fetch available models from the Model microservice (synchronous)."""
    try:
        url, api_key = _get_model_service_settings()
    except ValueError:
        return []

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{url}/v1/models/list",
                headers=_headers(api_key),
                json={"provider": provider, "provider_config": provider_config},
            )
            response.raise_for_status()
            return response.json().get("models", [])
    except Exception as e:
        logger.warning("Failed to fetch models from Model service: %s", e)
        return []


# ---------------------------------------------------------------------------
# Proxy LangChain models — delegate all invocations to the microservice
# ---------------------------------------------------------------------------


class MicroserviceChatModel(BaseChatModel):
    """A LangChain BaseChatModel that proxies all calls to the Model microservice.

    The flow engine calls ``build_model()`` which returns this object.
    Subsequent ``.invoke()`` / ``.ainvoke()`` / ``.stream()`` calls are
    translated into HTTP requests to ``{service_url}/v1/chat/completions``.
    """

    service_url: str
    service_api_key: str = ""
    provider: str = ""
    model: str = ""
    registry_model_id: str | None = None
    provider_config: dict = {}
    temperature: float | None = None
    max_tokens: int | None = None
    streaming: bool = False
    bound_tools: list[dict] | None = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "microservice-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "service_url": self.service_url,
            "provider": self.provider,
            "model": self.model,
            "registry_model_id": self.registry_model_id,
        }

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MicroserviceChatModel":
        """Return a copy of this model with tools bound for tool calling."""
        from langchain_core.tools import BaseTool
        from langchain_core.utils.function_calling import convert_to_openai_tool

        openai_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                openai_tools.append(tool)
            elif isinstance(tool, BaseTool):
                openai_tools.append(convert_to_openai_tool(tool))
            elif callable(tool):
                openai_tools.append(convert_to_openai_tool(tool))
            else:
                openai_tools.append(convert_to_openai_tool(tool))

        return self.model_copy(update={"bound_tools": openai_tools})

    def _build_payload(self, messages: list[BaseMessage], stream: bool = False) -> dict:
        config: dict = dict(self.provider_config)
        if self.registry_model_id:
            config["registry_model_id"] = self.registry_model_id

        payload: dict = {
            "provider": self.provider,
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "provider_config": config,
            "stream": stream,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.bound_tools:
            payload["tools"] = self.bound_tools
        return payload

    def _parse_response(self, data: dict) -> ChatResult:
        content = ""
        finish_reason = "stop"
        tool_calls_raw: list[dict] | None = None
        if data.get("choices"):
            choice = data["choices"][0]
            msg_data = choice.get("message", {})
            content = msg_data.get("content") or ""
            finish_reason = choice.get("finish_reason", "stop")
            tool_calls_raw = msg_data.get("tool_calls")

        usage = data.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        model_name = data.get("model", self.model)
        response_metadata = {
            "token_usage": token_usage,
            "model_name": model_name,
            "finish_reason": finish_reason,
        }

        # Build AIMessage with tool_calls if present
        msg_kwargs: dict[str, Any] = {
            "content": content,
            "response_metadata": response_metadata,
        }
        if tool_calls_raw:
            lc_tool_calls = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                lc_tool_calls.append({
                    "name": func.get("name", ""),
                    "args": args,
                    "id": tc.get("id", ""),
                })
            msg_kwargs["tool_calls"] = lc_tool_calls

        message = AIMessage(**msg_kwargs)
        # llm_output is what LangChain callbacks (including Langfuse) read
        # for token usage when the model is used inside agents/chains.
        llm_output = {"token_usage": token_usage, "model_name": model_name}
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False)
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{self.service_url}/v1/chat/completions",
                headers=_headers(self.service_api_key),
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(resp.json())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stream=False)
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.service_url}/v1/chat/completions",
                headers=_headers(self.service_api_key),
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True)
        if stop:
            payload["stop"] = stop
        # Request token usage in the streaming response
        payload["stream_options"] = {"include_usage": True}

        stream_usage: dict | None = None
        # Accumulate tool call deltas from streaming chunks
        tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}

        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{self.service_url}/v1/chat/completions",
                headers=_headers(self.service_api_key),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Capture usage from any chunk (typically the last one)
                        if chunk.get("usage"):
                            stream_usage = chunk["usage"]
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                msg_chunk = AIMessageChunk(content=content)
                                gen_chunk = ChatGenerationChunk(message=msg_chunk)
                                if run_manager:
                                    run_manager.on_llm_new_token(content)
                                yield gen_chunk

                            # Accumulate tool call deltas
                            tc_deltas = delta.get("tool_calls")
                            if tc_deltas:
                                for tc_delta in tc_deltas:
                                    idx = tc_delta.get("index", 0)
                                    if idx not in tool_call_accum:
                                        tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                                    if tc_delta.get("id"):
                                        tool_call_accum[idx]["id"] = tc_delta["id"]
                                    func = tc_delta.get("function", {})
                                    if func.get("name"):
                                        tool_call_accum[idx]["name"] += func["name"]
                                    if func.get("arguments"):
                                        tool_call_accum[idx]["arguments"] += func["arguments"]
                    except json.JSONDecodeError:
                        continue

        # If tool calls were accumulated, yield them as a final chunk
        if tool_call_accum:
            tool_call_chunks = []
            for idx in sorted(tool_call_accum.keys()):
                tc = tool_call_accum[idx]
                args_str = tc["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                tool_call_chunks.append({
                    "name": tc["name"],
                    "args": json.dumps(args) if isinstance(args, dict) else str(args),
                    "id": tc["id"],
                    "index": idx,
                    "type": "tool_call_chunk",
                })
            tc_msg = AIMessageChunk(content="", tool_call_chunks=tool_call_chunks)
            yield ChatGenerationChunk(message=tc_msg)

        # Yield a final empty chunk with usage_metadata so callers can extract tokens
        if stream_usage:
            usage_chunk = AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": stream_usage.get("prompt_tokens", 0),
                    "output_tokens": stream_usage.get("completion_tokens", 0),
                    "total_tokens": stream_usage.get("total_tokens", 0),
                },
                response_metadata={
                    "token_usage": stream_usage,
                    "model_name": self.model,
                },
            )
            yield ChatGenerationChunk(message=usage_chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True)
        if stop:
            payload["stop"] = stop
        # Request token usage in the streaming response
        payload["stream_options"] = {"include_usage": True}

        stream_usage: dict | None = None
        # Accumulate tool call deltas from streaming chunks
        tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.service_url}/v1/chat/completions",
                headers=_headers(self.service_api_key),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Capture usage from any chunk (typically the last one)
                        if chunk.get("usage"):
                            stream_usage = chunk["usage"]
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                msg_chunk = AIMessageChunk(content=content)
                                gen_chunk = ChatGenerationChunk(message=msg_chunk)
                                if run_manager:
                                    await run_manager.on_llm_new_token(content)
                                yield gen_chunk

                            # Accumulate tool call deltas
                            tc_deltas = delta.get("tool_calls")
                            if tc_deltas:
                                for tc_delta in tc_deltas:
                                    idx = tc_delta.get("index", 0)
                                    if idx not in tool_call_accum:
                                        tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                                    if tc_delta.get("id"):
                                        tool_call_accum[idx]["id"] = tc_delta["id"]
                                    func = tc_delta.get("function", {})
                                    if func.get("name"):
                                        tool_call_accum[idx]["name"] += func["name"]
                                    if func.get("arguments"):
                                        tool_call_accum[idx]["arguments"] += func["arguments"]
                    except json.JSONDecodeError:
                        continue

        # If tool calls were accumulated, yield them as a final chunk
        if tool_call_accum:
            tool_call_chunks = []
            for idx in sorted(tool_call_accum.keys()):
                tc = tool_call_accum[idx]
                args_str = tc["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                tool_call_chunks.append({
                    "name": tc["name"],
                    "args": json.dumps(args) if isinstance(args, dict) else str(args),
                    "id": tc["id"],
                    "index": idx,
                    "type": "tool_call_chunk",
                })
            tc_msg = AIMessageChunk(content="", tool_call_chunks=tool_call_chunks)
            yield ChatGenerationChunk(message=tc_msg)

        # Yield a final empty chunk with usage_metadata so callers can extract tokens
        if stream_usage:
            usage_chunk = AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": stream_usage.get("prompt_tokens", 0),
                    "output_tokens": stream_usage.get("completion_tokens", 0),
                    "total_tokens": stream_usage.get("total_tokens", 0),
                },
                response_metadata={
                    "token_usage": stream_usage,
                    "model_name": self.model,
                },
            )
            yield ChatGenerationChunk(message=usage_chunk)


class MicroserviceEmbeddings(LCEmbeddings):
    """A LangChain Embeddings model that proxies all calls to the Model microservice.

    The flow engine calls ``build_embeddings()`` which returns this object.
    Subsequent ``.embed_documents()`` / ``.embed_query()`` calls are
    translated into HTTP requests to ``{service_url}/v1/embeddings``.
    """

    service_url: str = ""
    service_api_key: str = ""
    provider: str = ""
    model: str = ""
    registry_model_id: str | None = None
    provider_config: dict = {}
    dimensions: int | None = None

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _build_payload(self, texts: list[str]) -> dict:
        config: dict = dict(self.provider_config)
        if self.registry_model_id:
            config["registry_model_id"] = self.registry_model_id

        payload: dict = {
            "provider": self.provider,
            "model": self.model,
            "input": texts,
            "provider_config": config,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = self._build_payload(texts)
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{self.service_url}/v1/embeddings",
                headers=_headers(self.service_api_key),
                json=payload,
            )
            resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

    def embed_query(self, text: str) -> List[float]:
        result = self.embed_documents([text])
        return result[0] if result else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = self._build_payload(texts)
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.service_url}/v1/embeddings",
                headers=_headers(self.service_api_key),
                json=payload,
            )
            resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

    async def aembed_query(self, text: str) -> List[float]:
        result = await self.aembed_documents([text])
        return result[0] if result else []
