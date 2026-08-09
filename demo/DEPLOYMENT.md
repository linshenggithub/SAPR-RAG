# SAPR-RAG Public Demo Deployment

This deployment exposes only the FastAPI gateway through a Cloudflare named tunnel. The model,
retrieval daemon, SSH service, model paths, and local files remain unavailable from the public
Internet.

## Public Boundary

```text
Public HTTPS hostname
  -> Cloudflare Tunnel
    -> 127.0.0.1:8200 (FastAPI gateway)
      -> 127.0.0.1:8001 (vLLM, private)
      -> 127.0.0.1:8100 (retrieval daemon, private)
```

Do not publish ports 8001 or 8100. Do not configure a public route for SSH.

## Private Files

The following files exist only on the server and must have mode `600`:

```text
~/.config/sapr-rag/service.env
~/.config/sapr-rag/tunnel.env
<repo>/demo/.env
```

`service.env` contains only the checkout location:

```bash
SAPR_RAG_ROOT=/absolute/path/to/the/checkout
```

`tunnel.env` contains the tunnel token obtained from the Cloudflare dashboard:

```bash
TUNNEL_TOKEN=replace-with-the-dashboard-token
```

Never paste the token into a command line, Git file, issue, terminal screenshot, or process title.

## Application Configuration

Copy `demo/.env.example` to the ignored `demo/.env`. Before public access, set:

```bash
SAPR_DEMO_TRUST_CLOUDFLARE_IP=1
SAPR_DEMO_ALLOWED_HOSTS=rag.example.com,localhost,127.0.0.1
SAPR_DEMO_ALLOWED_ORIGINS=https://rag.example.com
SAPR_DEMO_MAX_CONCURRENT_REQUESTS=1
SAPR_DEMO_REQUESTS_PER_WINDOW=20
SAPR_DEMO_RATE_WINDOW_SECONDS=3600
```

Use the actual public hostname. Do not put a personal email, real name, private IP, server name,
local path, API key, or token in frontend files or public API responses.

## Install User Services

```bash
mkdir -p ~/.config/systemd/user ~/.config/sapr-rag
cp demo/deploy/systemd/*.service ~/.config/systemd/user/
chmod 700 ~/.config/sapr-rag
chmod 600 ~/.config/sapr-rag/*.env demo/.env
systemctl --user daemon-reload
systemctl --user enable sapr-demo-retriever.service
systemctl --user enable sapr-demo-model.service
systemctl --user enable sapr-demo-gateway.service
systemctl --user enable sapr-demo-tunnel.service
```

The system administrator must enable user lingering once if services must survive logout and boot
without an interactive session:

```bash
sudo loginctl enable-linger "$USER"
```

If lingering is not permitted, the services run only while the user session remains active.

## Start and Verify

Start private services before the tunnel:

```bash
systemctl --user start sapr-demo-retriever.service
systemctl --user start sapr-demo-model.service
systemctl --user start sapr-demo-gateway.service
curl --fail --silent http://127.0.0.1:8200/api/health
systemctl --user start sapr-demo-tunnel.service
```

Inspect service state without printing private environment files:

```bash
systemctl --user --no-pager --full status sapr-demo-gateway.service
systemctl --user --no-pager --full status sapr-demo-tunnel.service
journalctl --user -u sapr-demo-gateway.service -n 100 --no-pager
```

## Privacy and Security Verification

Before sharing the URL, verify:

- `/openapi.json` returns 404.
- `/robots.txt` disallows indexing.
- `/api/health` contains only service booleans, not paths or corpus sizes.
- response headers contain `X-Robots-Tag`, CSP, HSTS, and `X-Frame-Options`.
- an invalid Host header is rejected.
- oversized requests return 413.
- repeated requests from one client return 429.
- the page and API do not expose a real name, email, local path, private IP, token, or traceback.
- only ports 8001, 8100, and 8200 are used by the application, all bound to loopback.

The application does not persist question bodies. Uvicorn access logging is disabled for the public
gateway. Cloudflare and the host may still retain operational metadata under their own policies, so
the page tells visitors not to submit personal or sensitive information.

## Rollback

To remove public access without stopping the private model and retriever:

```bash
systemctl --user disable --now sapr-demo-tunnel.service
```

To stop the complete demo:

```bash
systemctl --user disable --now sapr-demo-gateway.service
systemctl --user disable --now sapr-demo-model.service
systemctl --user disable --now sapr-demo-retriever.service
```
