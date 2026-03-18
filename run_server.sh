#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/opt/clipscut/workers"
DEFAULT_ENV_FILE=".env.production"

cd "$PROJECT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Expected a git repository at $PROJECT_DIR" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed." >&2
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [ -z "$BRANCH" ]; then
  echo "Could not determine the current git branch." >&2
  exit 1
fi

ENV_FILE="${ENV_FILE:-$DEFAULT_ENV_FILE}"
if [ ! -f "$ENV_FILE" ]; then
  if [ "$ENV_FILE" = ".env.production" ] && [ -f ".env" ]; then
    ENV_FILE=".env"
  else
    echo "Missing env file. Expected $PROJECT_DIR/.env.production or $PROJECT_DIR/.env" >&2
    exit 1
  fi
fi

COMPOSE_ARGS=(--env-file "$ENV_FILE")

echo "Using env file: $ENV_FILE"

git fetch origin
git pull --ff-only origin "$BRANCH"

docker compose "${COMPOSE_ARGS[@]}" config >/dev/null
docker compose "${COMPOSE_ARGS[@]}" up -d --build --remove-orphans

if [ "${RUN_DOCKER_PRUNE:-false}" = "true" ]; then
  docker image prune -af
fi

if [ "${FOLLOW_LOGS:-false}" = "true" ]; then
  docker compose "${COMPOSE_ARGS[@]}" logs -f
else
  docker compose "${COMPOSE_ARGS[@]}" ps
fi

# run: sudo bash run_server.sh
