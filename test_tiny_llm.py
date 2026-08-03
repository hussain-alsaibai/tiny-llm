"""Tests for tiny-llm — public API surface, no API keys needed."""
import pytest
from tiny_llm import LLM, _detect_provider


class TestProviderDetection:
    def test_openai_models(self):
        prov, model = _detect_provider("gpt-4o", None, None)
        assert prov.__class__.__name__ == "OpenAIProvider"
        assert model == "gpt-4o"

    def test_anthropic_models(self):
        prov, _ = _detect_provider("claude-3-5-sonnet-20241022", None, None)
        assert prov.__class__.__name__ == "AnthropicProvider"

    def test_gemini_models(self):
        prov, _ = _detect_provider("gemini-1.5-pro", None, None)
        assert prov.__class__.__name__ == "GoogleProvider"

    def test_ollama_local(self):
        prov, _ = _detect_provider("llama3.1", None, None)
        assert prov.__class__.__name__ == "OllamaProvider"

    def test_api_base_openai(self):
        prov, _ = _detect_provider("custom-model", None, "https://api.openai.com/v1")
        assert prov.__class__.__name__ == "OpenAIProvider"


class TestLLMInit:
    def test_init_defaults(self):
        llm = LLM(None)
        assert llm.model == "gpt-4o-mini"
        assert llm.max_retries == 3

    def test_init_custom_model(self):
        llm = LLM(None, model="gpt-4o")
        assert llm.model == "gpt-4o"

    def test_init_custom_api_base(self):
        llm = LLM(None, api_base="https://custom.api/v1")
        assert llm.api_base == "https://custom.api/v1"

    def test_init_timeout(self):
        llm = LLM(None, timeout=60.0)
        assert llm.timeout == 60.0

    def test_init_max_retries(self):
        llm = LLM(None, max_retries=5)
        assert llm.max_retries == 5

    def test_init_backoff_factor(self):
        llm = LLM(None, backoff_factor=2.0)
        assert llm.backoff_factor == 2.0


class TestCostTracking:
    def test_initial_state(self):
        llm = LLM(None)
        assert llm.total_tokens == 0
        assert llm.total_cost == 0.0
        assert llm.total_calls == 0

    def test_provider_property_lazy(self):
        llm = LLM(None)
        # Provider is lazily initialized
        assert llm._provider is None
        # Accessing .provider triggers initialization
        prov, resolved = llm.provider
        assert prov is not None
        assert llm._provider is not None


class TestStreaming:
    def test_chat_stream_exists(self):
        llm = LLM(None)
        assert hasattr(llm, "chat_stream")
        assert callable(llm.chat_stream)


class TestChat:
    def test_chat_method_exists(self):
        llm = LLM(None)
        assert hasattr(llm, "chat")
        assert callable(llm.chat)


class TestDefaultHeaders:
    def test_default_headers(self):
        llm = LLM(None, default_headers={"X-Custom": "value"})
        assert llm.default_headers["X-Custom"] == "value"

    def test_merge_headers(self):
        llm = LLM(None, default_headers={"X-App": "test"})
        merged = llm._merge_headers({"X-Request": "id-123"})
        assert merged["X-App"] == "test"
        assert merged["X-Request"] == "id-123"


class TestStats:
    def test_stats_initial(self):
        llm = LLM(None)
        stats = llm.stats()
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0
        # total_cost may not be in stats dict
        assert stats["total_errors"] == 0
