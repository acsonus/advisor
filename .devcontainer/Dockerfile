FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Install system dependencies required by matplotlib/mplfinance, reportlab, and git tooling
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libfreetype6-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    pkg-config \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment outside the bind-mounted workspace
# so it does not conflict with the host's .venv directory
RUN python -m venv "${VIRTUAL_ENV}" \
    && mkdir -p /home/vscode/.cache/huggingface \
    && chown -R vscode:vscode "${VIRTUAL_ENV}" /home/vscode/.cache

USER vscode

# Upgrade packaging and common Python development tools
RUN pip install --upgrade pip setuptools wheel ruff black isort debugpy

# Pre-install project dependencies for fast container startup
COPY --chown=vscode:vscode requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt
