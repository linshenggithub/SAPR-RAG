# SAPR-RAG Online Demo

This directory contains an isolated online demo for the SAPR-RAG research workspace. It does
not change the offline evaluation pipeline. The browser, API gateway, vLLM server, and retrieval
daemon communicate through explicit HTTP boundaries.

## Architecture

```text
Browser
  -> FastAPI gateway (127.0.0.1:8200)
       -> vLLM OpenAI-compatible server (127.0.0.1:8001)
            -> Qwen2.5-7B-Instruct + configurable LoRA
       -> retrieval daemon (127.0.0.1:8100)
            -> BGE on CPU + mmap FAISS index on CPU
```

The gateway emits Server-Sent Events from `POST /api/chat/stream`:

- `status`: current public stage;
- `query`: generated retrieval query;
- `documents`: retrieved document summaries;
- `evidence`: selected evidence;
- `answer_delta`: model-generated answer token increments;
- `done` or `error`: terminal state.

Raw reasoning text is never sent to the browser.

## Server Setup

Run all commands from the repository root. The launchers source `config/env_3090.sh` and reuse
the existing `reasonrag` environment.

Install only the lightweight gateway dependencies when needed:

```bash
conda run -n reasonrag python -m pip install -r demo/requirements.txt
```

Create an optional local configuration:

```bash
cp demo/.env.example demo/.env
```

The current adapter default is:

```text
03_sapr_rag/models/sapr-rag-sft-lora/checkpoint-1650
```

Replace `SAPR_DEMO_LORA_PATH` in `demo/.env` when a better checkpoint is selected. No Python
code change is required.

## Launch

Use three terminal or tmux sessions:

```bash
bash demo/scripts/launch_demo_retriever.sh
bash demo/scripts/launch_demo_model.sh
bash demo/scripts/launch_demo_gateway.sh
```

The first retrieval startup may take several minutes while the corpus cache is prepared. Check
the internal services before exposing the gateway:

```bash
curl -s http://127.0.0.1:8100/health
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8200/api/health
```

For a private workstation test, forward only the gateway:

```bash
ssh -L 8200:127.0.0.1:8200 <user>@<server>
```

Then open `http://127.0.0.1:8200`. Do not expose ports 8001 or 8100 publicly.

## Public Deployment Guardrails

- Keep all three services bound to `127.0.0.1`.
- Publish only port 8200 through an outbound HTTPS tunnel.
- Keep `SAPR_DEMO_MAX_CONCURRENT_REQUESTS=1` for the first public version.
- Set an exact `SAPR_DEMO_ALLOWED_ORIGINS` value if the frontend is later moved to GitHub Pages.
- Set `SAPR_DEMO_TRUST_CLOUDFLARE_IP=1` only when direct access to the gateway is blocked and all
  public traffic comes through Cloudflare Tunnel.
- Never commit `demo/.env`, credentials, tunnel tokens, model files, indexes, corpora, or logs.

For the named Cloudflare Tunnel, privacy hardening, and user-level `systemd` services, follow
[`DEPLOYMENT.md`](./DEPLOYMENT.md). Quick Tunnels are not suitable because the demo uses SSE and
requires a stable hostname.

## Tests

The state-machine tests use in-memory fakes and do not load a model or FAISS index:

```bash
python -m unittest discover -s demo/tests -v
```
