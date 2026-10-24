#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${CHROME_BIN:-}" ]]; then
  for candidate in chromium chromium-browser google-chrome google-chrome-stable; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CHROME_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "${CHROME_BIN:-}" ]]; then
  echo "Chromium or Chrome was not found. Install Chromium and retry." >&2
  exit 1
fi
export CHROME_BIN

if [[ -d /tmp/openssl-deb/root/usr ]]; then
  export OPENSSL_DIR=/tmp/openssl-deb/root/usr
  export OPENSSL_LIB_DIR=/tmp/openssl-deb/root/usr/lib/x86_64-linux-gnu
  export OPENSSL_INCLUDE_DIR=/tmp/openssl-deb/root/usr/include
fi

SERVER_PID=""
if ! curl -fsS http://localhost:8000/index.html >/dev/null 2>&1; then
  python3 -m http.server 8000 --directory "$ROOT/site" >/tmp/minishop-http.log 2>&1 &
  SERVER_PID=$!
  trap 'if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
  for _ in $(seq 1 25); do
    if curl -fsS http://localhost:8000/index.html >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
fi

cd "$ROOT/rl"
mkdir -p logs models

cargo build --release
BIN="$ROOT/rl/target/release/minishop-env"

"$BIN" train --seeds 1,2,3 --episodes 20 --eval-episodes 20
"$BIN" random --seeds 1 --episodes 20
for popup_p in 0 0.15 0.4; do
  "$BIN" eval --seeds 1,2,3 --eval-episodes 7 --popup-p "$popup_p"
done

python3 report.py
echo "Done: $ROOT/rl/report.md"
