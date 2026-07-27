# tiny-llm

> **Unified LLM client for Python — zero dependencies, every major provider.**

`tiny-llm` wraps OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and Ollama
behind a single `LLM.chat()` interface. Automatic retries, streaming, cost tracking,
function calling, and model routing — all in one MIT-licensed file you can copy-paste
anywhere.

```bash
pip install tiny-llm
# or: copy tiny_llm.py into your project (MIT licensed)
```

## Features

| Feature | Details |
|---------|---------|
| **Multi-provider** | OpenAI, Anthropic, Google, xAI/Grok, Ollama, + any OpenAI-compatible API |
| **Single interface** | `.chat(messages)` — provider auto-detected from model name |
| **Streaming** | `.chat_stream()` yields tokens in real-time |
| **Function calling** | Unified `tools=[]` schema across all providers |
| **Retry + backoff** | Exponential backoff on 429/5xx, configurable retries |
| **Cost tracking** | Token counts and estimated USD cost per call + session aggregate |
| **Model routing** | Fallback chains when primary model is unavailable |
| **System templates** | Parameterized system prompts with context injection |
| **Zero dependencies** | Stdlib only — no pip install chain, no version conflicts |

## Quick Start

```python
from tiny_llm import LLM

llm = LLM(api_key="sk-...", model="gpt-4o-mini")

# Simple string-in → string-out
response = llm.chat_simple("Explain quantum entanglement to a 10-year-old")
print(response)

# Streaming
for chunk in llm.chat_stream([{"role": "user", "content": "Tell me a story."}]):
    print(chunk.delta, end="", flush=True)

# Function calling
response = llm.chat(messages, tools=[{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}}
    }
}])
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc['function']['name']}, args: {tc['function']['arguments']}")

# Cost tracking
print(llm.total_cost)  # cumulative USD
print(llm.stats())     # full session stats
```

## Supported Providers

| Provider | Model Prefix | Notes |
|----------|-------------|-------|
| OpenAI | `gpt-4o`, `gpt-4`, `gpt-3.5-turbo` | Default if no match |
| Anthropic | `claude-3-5-sonnet`, `claude-3-opus`, etc. | System prompt handled separately |
| Google | `gemini-1.5-pro`, `gemini-1.5-flash` | Via AIStudio API |
| xAI / Grok | `grok-2`, `grok-1.5` | OpenAI-compatible endpoint |
| Ollama | `llama3`, `mistral`, `qwen2.5`, etc. | Local models via localhost:11434 |
| Custom | Any model + `--api-base` | OpenAI-compatible, Anthropic, or Ollama |

### Ollama (Local Models)

```python
llm = LLM(model="llama3.1")  # Auto-detects Ollama if no API key
# Or explicit:
llm = LLM(model="qwen2.5:7b", api_base="http://localhost:11434")
```

### Google Gemini

```python
llm = LLM(api_key="...", model="gemini-1.5-flash")
```

## Advanced Usage

### Model fallback chain

```python
for model in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
    try:
        resp = llm.chat(messages, model=model)
        break
    except Exception as e:
        if "rate_limit" in str(e).lower():
            continue
        raise
```

### Cost budgets

```python
llm = LLM(api_key="...", model="gpt-4o", max_retries=2)

resp = llm.chat(messages)
stats = llm.stats()
print(f"This call cost ${resp.usage['total_cost']:.4f}")
print(f"Session total: ${stats['total_cost_usd']:.4f} ({stats['total_calls']} calls)")

if stats['total_cost_usd'] > 5.0:
    print("Budget exceeded — pause and alert")
```

### Subclass for tool implementation

```python
class MyAgent(LLM):
    def call_tool(self, tool_call):
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        if name == "get_weather":
            return {"city": args["city"], "temp": 22, "condition": "sunny"}
        raise NotImplementedError(f"Unknown tool: {name}")

agent = MyAgent(api_key="sk-...", model="gpt-4o")
resp = agent.chat(messages, tools=TOOL_DEFS)
while resp.tool_calls:
    results = agent.run_tools(resp)
    messages.append({"role": "assistant", "content": resp.content})
    messages.extend(results)
    resp = agent.chat(messages)
```

### System prompt templates

```python
SYSTEM_TEMPLATE = """You are a {role} assistant.
Context: {context}
Your expertise level: {level}"""

messages = [
    {"role": "system", "content": SYSTEM_TEMPLATE.format(
        role="technical", context="Python 3.11", level="expert"
    )},
    {"role": "user", "content": "What's new in dataclasses?"},
]
resp = llm.chat(messages)
```

## CLI

```bash
# Set API key once
export OPENAI_API_KEY="sk-..."

# One-shot prompts
python tiny_llm.py "What is the capital of France?"
python tiny_llm.py "Debug this code" --stream --model gpt-4o

# JSON output (for scripting)
python tiny_llm.py "Count to 10" --json
```

## Architecture

```
tiny_llm.py
├── LLM                          # Main client class
│   ├── chat()                   # Sync chat
│   ├── chat_stream()            # Streaming chat
│   ├── chat_simple()           # String → String
│   ├── call_tool()             # Override for tool execution
│   └── run_tools()             # Execute tool calls from response
├── Providers
│   ├── OpenAIProvider          # OpenAI + OpenAI-compatible endpoints
│   ├── AnthropicProvider      # Claude API
│   ├── GoogleProvider          # Gemini API
│   ├── GrokProvider            # xAI API
│   └── OllamaProvider          # Local models
├── Utilities
│   ├── _detect_provider()       # Model → Provider mapping
│   ├── _estimate_cost()         # Token → USD
│   └── count_tokens()          # Rough token estimation
└── CLI
    └── main()                  # python tiny_llm.py <prompt>
```

## Cost Estimation

`tiny-llm` uses built-in pricing tables (prices per 1M tokens, USD).
Local models (Ollama) are estimated at $0.

Current providers:
- **GPT-4o**: $2.50 / $10.00 (in/out)
- **GPT-4o-mini**: $0.15 / $0.60
- **Claude 3.5 Sonnet**: $3.00 / $15.00
- **Gemini 1.5 Flash**: $0.075 / $0.30
- **DeepSeek Chat**: $0.14 / $0.28

Add custom pricing by modifying the `COST_PER_MILLION` dict.

## Installation

```bash
pip install tiny-llm
```

Or copy `tiny_llm.py` directly — it's MIT licensed and has no dependencies.

## Ecosystem

`tiny-llm` fits into the tiny-* toolkit ecosystem:

| Tool | What it does |
|------|-------------|
| `tiny-retry` | Retry with backoff for LLM calls |
| `tiny-rate` | Rate limiting for LLM APIs |
| `tiny-budget` | Cost caps per agent run |
| `tiny-trace` | Trace multi-step LLM workflows |
| `tiny-stream` | SSE helpers for LLM UI streaming |
| `tiny-embed` | Embedding generation for RAG |
| `fast-cache` | Cache LLM responses (L1) |
| `tiny-store` | Persist LLM memory across restarts (L2) |

## License

MIT — copy, modify, use in production, no attribution required.
