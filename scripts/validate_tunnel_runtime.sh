#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=""
COMPOSE_FILE="docker-compose.yml"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$ENV_FILE" ]; then
  echo "Missing required --env-file argument." >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  awk -F= -v key="$key" '
    $0 ~ "^[[:space:]]*" key "=" {
      value = substr($0, index($0, "=") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' "$ENV_FILE"
}

extract_caddy_port_mapping() {
  local mapping_line

  mapping_line="$(
    awk '
    /^[[:space:]]*caddy:[[:space:]]*$/ {
      in_caddy = 1
      next
    }
    in_caddy && /^[[:space:]]{2}[A-Za-z0-9_-]+:[[:space:]]*$/ && $0 !~ /^[[:space:]]*ports:[[:space:]]*$/ {
      in_caddy = 0
      in_ports = 0
    }
    in_caddy && /^[[:space:]]*ports:[[:space:]]*$/ {
      in_ports = 1
      next
    }
    in_ports && /"[0-9]+:[0-9]+"/ {
      print
      exit
    }
  ' "$COMPOSE_FILE"
  )"

  if [ -z "$mapping_line" ]; then
    return 1
  fi

  echo "$mapping_line" | sed -E 's/.*"([0-9]+):([0-9]+)".*/\1 \2/'
}

ENVIRONMENT_VALUE="$(read_env_value ENVIRONMENT)"
CADDY_DOMAIN_VALUE="$(read_env_value CADDY_DOMAIN)"
WORKER_PUBLIC_BASE_URL_VALUE="$(read_env_value WORKER_PUBLIC_BASE_URL)"

if [ -z "$CADDY_DOMAIN_VALUE" ]; then
  echo "Skipping tunnel validation because CADDY_DOMAIN is unset in $ENV_FILE."
  exit 0
fi

read -r CADDY_HOST_PORT CADDY_CONTAINER_PORT <<<"$(extract_caddy_port_mapping)"

if [ -z "${CADDY_HOST_PORT:-}" ] || [ -z "${CADDY_CONTAINER_PORT:-}" ]; then
  echo "Could not detect the Caddy port mapping from $COMPOSE_FILE." >&2
  exit 1
fi

if [[ "$CADDY_DOMAIN_VALUE" == :* ]]; then
  CADDY_LISTEN_PORT="${CADDY_DOMAIN_VALUE#:}"
  if [ -z "$CADDY_LISTEN_PORT" ]; then
    echo "CADDY_DOMAIN=$CADDY_DOMAIN_VALUE is missing a listen port." >&2
    exit 1
  fi

  if [ "$CADDY_LISTEN_PORT" != "$CADDY_HOST_PORT" ] || [ "$CADDY_LISTEN_PORT" != "$CADDY_CONTAINER_PORT" ]; then
    echo "Tunnel-backed Caddy port mismatch detected." >&2
    echo "  CADDY_DOMAIN=$CADDY_DOMAIN_VALUE" >&2
    echo "  docker-compose caddy ports=$CADDY_HOST_PORT:$CADDY_CONTAINER_PORT" >&2
    echo "Set CADDY_DOMAIN=:$CADDY_CONTAINER_PORT and point cloudflared to http://127.0.0.1:$CADDY_HOST_PORT." >&2
    exit 1
  fi

  echo "Validated tunnel-backed Caddy origin at http://127.0.0.1:$CADDY_HOST_PORT"
  exit 0
fi

case "${ENVIRONMENT_VALUE,,}" in
  production|prod)
    if [[ "$WORKER_PUBLIC_BASE_URL_VALUE" =~ ^https:// ]]; then
      echo "CADDY_DOMAIN=$CADDY_DOMAIN_VALUE does not match this tunnel-backed compose file." >&2
      echo "This deployment publishes Caddy on $CADDY_HOST_PORT:$CADDY_CONTAINER_PORT and expects CADDY_DOMAIN=:$CADDY_CONTAINER_PORT." >&2
      echo "For Cloudflare Tunnel production, point api.clipscut.pro to http://127.0.0.1:$CADDY_HOST_PORT." >&2
      echo "If you intentionally want direct Caddy TLS instead, update docker-compose and edge routing together first." >&2
      exit 1
    fi
    ;;
esac

echo "Skipping tunnel port validation for non-tunnel Caddy domain: $CADDY_DOMAIN_VALUE"
