# syntax=docker/dockerfile:1

###############################################################################
# klit-flow — local, offline code-intelligence tool
#
# This image is fully self-contained: the tree-sitter parser binaries and the
# embedding model are downloaded at build time and baked into the image, so the
# container performs **no network calls at runtime** — consistent with the
# project's offline guarantee.
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
# Behind a proxy that re-signs HTTPS, every download below (pip, PyTorch index,
# tree-sitter parsers, HuggingFace model) needs your company root CA to be
# trusted. Drop one or more PEM-encoded ``*.crt`` files into ``./certs/`` next
# to this Dockerfile and they are installed into the system trust store at
# build time. With only the placeholder ``.gitkeep`` present this is a no-op,
# so the image still builds normally outside a corporate network.
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
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[release]"

# Bake in the parser binaries (current platform) and the embedding model so the
# container never needs the network at runtime.
RUN klit-flow download-parsers --cache-dir "$KLIT_FLOW_PARSER_CACHE_DIR" \
    && klit-flow download-model "$KLIT_FLOW_MODEL_DIR"

# Target repositories are mounted here.
WORKDIR /workspace

# Web portal default port (only used by `serve`).
EXPOSE 5173

ENTRYPOINT ["klit-flow"]
CMD ["--help"]
