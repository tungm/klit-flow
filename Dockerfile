# syntax=docker/dockerfile:1

###############################################################################
# klit-flow — local, offline code-intelligence tool
#
# This image is fully self-contained: the tree-sitter parser binaries and the
# embedding model are copied from the pre-downloaded release bundle
# (release/v1.1.0/) and baked into the image, so the container performs **no
# network calls at runtime** — consistent with the project's offline guarantee.
# Baking the assets in (rather than downloading them at build time) also means
# the build works behind corporate proxies that mangle HTTP Range requests.
#
# Build:
#   docker build -t klit-flow .
#
# Index a repo (mount it at /workspace):
#   docker run --rm -v "$(pwd)/my-app:/workspace" klit-flow \
#       analyze /workspace --platform android
#
# Serve the web portal + MCP server (bind to 0.0.0.0 so the host can reach it):
#   docker run --rm -p 5173:5173 -v "$(pwd)/my-app:/workspace" klit-flow \
#       serve --host 0.0.0.0
###############################################################################

FROM python:3.11-slim AS base

# Where the bundled offline assets live inside the image.
ENV KLIT_FLOW_PARSER_CACHE_DIR=/opt/klit-flow/parsers \
    KLIT_FLOW_MODEL_DIR=/opt/klit-flow/models/bge-small-en-v1.5 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ── Corporate TLS interception ───────────────────────────────────────────────
# Behind a proxy that re-signs HTTPS, the pip downloads below (PyPI + the
# PyTorch CPU index) need your company root CA to be trusted. Drop one or more
# PEM-encoded ``*.crt`` files into ``./certs/`` next to this Dockerfile and they
# are installed into the system trust store at build time. With only the
# placeholder ``.gitkeep`` present this is a no-op, so the image still builds
# normally outside a corporate network.
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Point pip and every requests/urllib/curl-based tool at the combined system
# bundle, which now includes the corporate CA (and all public CAs).
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /src

# Copy only what is needed to install the package first, to maximise layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install klit-flow plus CPU-only PyTorch (required by sentence-transformers).
# CPU wheels keep the image dramatically smaller than the default CUDA build.
# --timeout / --retries make large downloads (the ~190 MB torch CPU wheel)
# survive slow corporate proxies that would otherwise trip a read timeout.
RUN pip install --upgrade pip \
    && pip install --timeout 1000 --retries 10 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --timeout 1000 --retries 10 "."

# Bake in the parser binaries and the embedding model from the pre-downloaded
# release bundle so the container never needs the network — at build OR runtime.
# Only the Linux parser binaries are copied (this is a Linux image); the runtime
# auto-detects the matching <platform>/ subdir under KLIT_FLOW_PARSER_CACHE_DIR.
COPY release/v1.1.0/parsers/linux-x86_64/  /opt/klit-flow/parsers/linux-x86_64/
COPY release/v1.1.0/parsers/linux-aarch64/ /opt/klit-flow/parsers/linux-aarch64/
COPY release/v1.1.0/models/bge-small-en-v1.5/ /opt/klit-flow/models/bge-small-en-v1.5/

# Target repositories are mounted here.
WORKDIR /workspace

# Web portal default port (only used by `serve`).
EXPOSE 5173

ENTRYPOINT ["klit-flow"]
CMD ["--help"]
