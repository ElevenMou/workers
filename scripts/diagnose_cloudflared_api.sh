#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-cloudflared-api}"
CONFIG_PATH="${CONFIG_PATH:-/etc/cloudflared/api-clipscut.yml}"
LOCAL_ORIGIN="${LOCAL_ORIGIN:-http://127.0.0.1:7050/ready}"
PUBLIC_URL="${PUBLIC_URL:-https://api.clipscut.pro/ready}"

echo "==> cloudflared service status"
if command -v systemctl >/dev/null 2>&1; then
  systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
else
  echo "systemctl is not available on this machine."
fi

echo ""
echo "==> recent cloudflared logs"
if command -v journalctl >/dev/null 2>&1; then
  journalctl -u "${SERVICE_NAME}.service" -n 80 --no-pager || true
else
  echo "journalctl is not available on this machine."
fi

echo ""
echo "==> tunnel inventory"
if command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel list || true
else
  echo "cloudflared is not installed or not on PATH."
fi

echo ""
echo "==> ingress validation"
if command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel ingress validate --config "${CONFIG_PATH}" || true
else
  echo "Skipping ingress validation because cloudflared is unavailable."
fi

echo ""
echo "==> local origin"
curl -sS -D - "${LOCAL_ORIGIN}" -o - || true

echo ""
echo ""
echo "==> public API"
curl -sS -D - "${PUBLIC_URL}" -o - || true
