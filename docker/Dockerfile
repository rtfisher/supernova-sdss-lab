# Linux + Python base image maintained for devcontainers
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Useful system tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential less unzip tar ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Workdir inside Codespaces
WORKDIR /workspaces/supernova-sdss-lab

# Python deps (adjust as needed)
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your exact repo contents
COPY sdss_sne.py ./sdss_sne.py
COPY README ./README
COPY data.tar ./data.tar

# Entrypoint launches Jupyter and unpacks data on first run
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

