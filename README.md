# LiteLLM Lab

**Local Model Testing & API Gateway** — discover Ollama models, route them through LiteLLM, compare configured aliases, and inspect benchmark history.

## Features

- Local Ollama + LiteLLM gateway with loopback-only host ports.
- Ollama model auto-detection with size, parameter, and quantization details.
- Async benchmarks with configurable concurrency, streaming TTFT, latency percentiles, tokens/sec, and failures.
- Sequential model comparisons with shared prompts and settings.
- Local SQLite aggregate history and CSV/JSON export.
- Streamlit comparison dashboard and history downloads.

## Quick start

Requirements: Docker Compose v2, Python 3.11+, and enough memory for the selected Ollama model. Install uv for the Python CLI.

1. Copy the environment template and replace the placeholder with a unique sk- key:

    cp .env.example .env

Keep .env private; it is gitignored.

2. Start services and pull a model:

    docker compose up -d
    docker compose exec ollama ollama pull llama3.2

3. Install the CLI, discover models, and check the gateway:

    uv sync
    uv run litellm-lab models
    curl http://127.0.0.1:4000/health/liveliness

4. Load the gateway key into the shell:

    export LITELLM_MASTER_KEY="$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-)"
    export LITELLM_API_KEY="$LITELLM_MASTER_KEY"

PowerShell:

    $env:LITELLM_MASTER_KEY = (Get-Content .env | Where-Object { $_ -match '^LITELLM_MASTER_KEY=' }) -replace '^LITELLM_MASTER_KEY=', ''
    $env:LITELLM_API_KEY = $env:LITELLM_MASTER_KEY

## Benchmark and compare

Run one model:

    uv run litellm-lab bench --model local-llama --requests 20 --concurrency 4

Measure time to first content token:

    uv run litellm-lab bench --model local-llama --stream --requests 20 --concurrency 4

Add model aliases in litellm_config.yaml and pull their Ollama models, then compare sequentially:

    uv run litellm-lab compare --models local-llama local-qwen --requests 20 --concurrency 2 --stream

The comparison runs one alias at a time to reduce interference. Results report request latency, p50/p95, TTFT, token usage, throughput, and failures. Streaming TTFT is the first non-empty content chunk observed by the client; providers may omit token usage for streams.

Optional --output results/run.jsonl writes JSONL. Error records store only an error class. SQLite summaries default to ~/.litellm-lab/history.sqlite3; set LITELLM_LAB_DB or pass --history-db to choose another path.

## Dashboard and history exports

    uv sync --extra dashboard
    uv run streamlit run dashboard.py --server.address 127.0.0.1

The dashboard compares p50/p95 latency, TTFT, throughput, success rate, and run history. It can download CSV or JSON. CLI alternatives:

    uv run litellm-lab history --limit 50
    uv run litellm-lab export-history --format csv --output results/history.csv

History stores model alias, endpoint, run settings, metrics, and a SHA-256 prompt hash. It does not store raw prompts or model responses. Protect metrics and history files as local data.

## Add another provider

Edit litellm_config.yaml and use LiteLLM's provider/model syntax. Put provider keys in the ignored .env file (for example OPENAI_API_KEY=...) and reference them with os.environ/OPENAI_API_KEY. Compose passes only .env variables to LiteLLM. Restart after config changes:

    docker compose restart litellm

Remote providers receive requests and may charge for usage. Review their terms before sending sensitive data or using billable models.

## Methodology and security

Record model version/quantization, hardware, Ollama/LiteLLM versions, prompt, request count, concurrency, token cap, stream mode, and warm/cold state. These are request-throughput measurements, not model-quality evaluations. The starter has no DB-backed virtual keys or spend enforcement. Keep both localhost services and the unauthenticated dashboard loopback-only; don't expose on untrusted networks. The project is not production-hardened.

## License

MIT. See LICENSE.

## 👨‍💻 Author

**Nitin Mehta**

🌐 [puffxhost.com](https://puffxhost.com)

📧 [support@puffxhost.com](mailto:support@puffxhost.com)

📸 @nitinmehta.codes
