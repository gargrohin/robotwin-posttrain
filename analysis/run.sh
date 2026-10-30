#!/usr/bin/env bash
# Start the eval explorer on this machine's localhost (CPU only; no GPU is visible to it).
# VS Code Remote-SSH forwards the port: open it from the pop-up or the Ports tab.
# usage: analysis/run.sh [port]    (default 8611; Ctrl+C stops it)
set -euo pipefail
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES="" exec .venv/bin/streamlit run app.py --server.port "${1:-8611}" \
  --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
