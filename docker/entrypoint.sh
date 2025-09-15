#!/usr/bin/env bash
set -e

# Ensure a data/ folder exists and unpack once
mkdir -p data
if [ -f "data.tar" ] && [ ! -f ".data_unpacked" ]; then
  echo "Extracting data.tar into ./data ..."
  tar -xf data.tar -C data
  touch .data_unpacked
fi

# Drop into an interactive shell
exec bash
