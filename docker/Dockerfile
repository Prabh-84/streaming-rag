FROM python:3.11-slim

WORKDIR /app

COPY requirements.lock .
# The spaCy model is installed as a plain pip wheel (not `spacy download`) — the two use different
# download clients, and this network has been observed to reset connections specifically against
# spacy download's redirect path while the direct wheel URL succeeds reliably.
# --timeout/--retries: BuildKit's build-time network path has been observed to hit pip's default
# 15s read timeout on large wheels (e.g. cryptography, pulled in transitively by google-genai)
# even though the same host's runtime network path is unaffected.
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.lock \
    && pip install --no-cache-dir --timeout=120 --retries=5 "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

COPY app ./app
COPY scripts ./scripts
COPY data/corpus ./data/corpus

EXPOSE 8000

# Liveness only — never depends on Qdrant or ingestion (REQ-DEPLOY-01). Ingestion runs inside the
# FastAPI startup event (app/main.py), not as a blocking pre-step here, so /health is reachable as
# soon as uvicorn binds the port.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
