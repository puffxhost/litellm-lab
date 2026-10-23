# LiteLLM Lab

**Local Model Testing & API Gateway** — a small, reproducible lab for running local language models behind the OpenAI-compatible [LiteLLM Proxy](https://docs.litellm.ai/) and measuring request latency and throughput.

> Designed for local development. The Docker Compose gateway binds to `127.0.0.1`; review the security notes before changing its network exposure.

## What is included

- **LiteLLM Proxy** configuration with a local Ollama model alias, using the same OpenAI-compatible chat API clients use in production.
- **Ollama + LiteLLM Compose stack** for a local, no-provider-key starting point.
- **Async benchmark CLI** with configurable request count and concurrency; records latency, token usage, throughput, and failures as JSON Lines.
- **Reproducible run notes** so model, hardware, quantization, concurrency, and prompt settings can be compared consistently.

This repository is a learning and evaluation scaffold. It does not include a hosted service, authentication UI, database-backed virtual keys, usage billing, or a production deployment.

## Requirements

- Docker Engine and Docker Compose v2.
- Python 3.11+ and `uv` for the benchmark CLI.
- Enough RAM/VRAM for the Ollama model you choose. Model downloads can be several gigabytes.

## Quick start: local gateway

1. Copy the sample environment file and choose a strong local gateway key:

   ```bash
   cp .env.example .env
   ```

   Replace `LITELLM_MASTER_KEY` in `.env` with a unique random value beginning with `sk-`. Keep `.env` private; it is ignored by Git.

2. Start the services:

   ```bash
   docker compose up -d
   docker compose exec ollama ollama pull llama3.2
   ```

   The model download can take a while. Check service logs with `docker compose logs -f litellm`.

3. Confirm the gateway responds:

   ```bash
   curl http://127.0.0.1:4000/health/liveliness
   ```

4. Send a chat request:

   ```bash
   curl http://127.0.0.1:4000/v1/chat/completions \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"local-llama","messages":[{"role":"user","content":"Give me one sentence about local AI."}]}'
   ```

On PowerShell, set the key in the current shell with `$env:LITELLM_MASTER_KEY = "your-key"` before running curl.

The model alias is `local-llama`; change it in both `litellm_config.yaml` and your request if you select another Ollama model. Pull that model into Ollama before sending requests.

## Run the benchmark CLI

In another terminal:

```bash
uv sync
uv run litellm-lab bench --base-url http://127.0.0.1:4000/v1 --model local-llama --requests 20 --concurrency 4
```

The CLI reads `LITELLM_API_KEY` from the environment. Copy the master key from `.env` into this variable before starting the CLI:

```bash
# macOS / Linux
export LITELLM_API_KEY="$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-)"

# PowerShell
$env:LITELLM_API_KEY = (Get-Content .env | Where-Object { $_ -match '^LITELLM_MASTER_KEY=' }) -replace '^LITELLM_MASTER_KEY=', ''
```

Optional flags include `--prompt`, `--max-tokens`, and `--output`. The output file contains one JSON object per request plus a final summary; it records metrics and error types, not prompt or model response text. Keep benchmark logs private if you add custom metadata or prompts in a future extension.

## Configure another provider or model

Edit `litellm_config.yaml` and add a model entry using the LiteLLM provider/model naming convention. Put provider credentials in environment variables referenced from the config (for example, `os.environ/OPENAI_API_KEY`); never commit real keys. Restart the gateway after changing the config:

```bash
docker compose restart litellm
```

For remote providers, calls leave your computer and may incur charges. Read the provider's data and billing terms before using real or sensitive prompts. This starter's Ollama route stays on the Compose network.

## Benchmark methodology

For comparable runs, record the model name and quantization, hardware, Ollama and LiteLLM versions, prompt, maximum output tokens, request count, concurrency, and whether the model was warmed up. The CLI sends independent requests concurrently; results are not a controlled benchmark of model quality. First-token latency is not measured, and streaming is not enabled. Run repeated trials and compare percentiles rather than relying on a single average.

## Security and operations

- The Compose port is published only on loopback. Do not change it to `0.0.0.0` on an untrusted network.
- The master key is a local admin credential. Do not expose it in browser code, logs, screenshots, or commits.
- The starter has no database. Do not assume LiteLLM spend limits or virtual-key features are enforced in this setup; configure provider-side spending controls as well if using billable APIs.
- Avoid sending private or regulated data to a remote model provider unless you have authorization and an appropriate data-processing arrangement.
- This is not production-hardened. For multi-user or public deployments, configure TLS, secret management, a database, access controls, monitoring, rate limits, and network restrictions.

## Project layout

```text
.
├── docker-compose.yml
├── litellm_config.yaml
├── .env.example
├── pyproject.toml
└── src/litellm_lab/
    └── cli.py
```

## License

MIT. See [LICENSE](LICENSE).

## Author

Made with ❤️ by [Nitin Mehta](https://github.com/puffxhost).
