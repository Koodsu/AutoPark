# ParkPilot developer entry points.
SHELL := /bin/bash
BACKEND := backend
FRONTEND := frontend
UV := uv run --project $(BACKEND)
SEEDS ?= 5

.PHONY: install dev backend-dev frontend-dev lint lint-backend lint-frontend test test-backend \
	test-frontend bench bench-smoke schema e2e build format

install:
	cd $(BACKEND) && uv sync
	cd $(FRONTEND) && pnpm install --frozen-lockfile

dev:
	@echo "Backend on :8000, Vite on :5173 (proxying /api and /ws)"
	@trap 'kill 0' EXIT; \
	(cd $(BACKEND) && uv run parkpilot serve --port 8000) & \
	(cd $(FRONTEND) && pnpm dev) & \
	wait

lint: lint-backend lint-frontend

lint-backend:
	cd $(BACKEND) && uv run ruff check . && uv run ruff format --check . && uv run mypy parkpilot tests

lint-frontend:
	cd $(FRONTEND) && pnpm exec tsc --noEmit && pnpm exec eslint . && pnpm exec prettier --check .

format:
	cd $(BACKEND) && uv run ruff format . && uv run ruff check --fix .
	cd $(FRONTEND) && pnpm exec prettier --write .

test: test-backend test-frontend

test-backend:
	cd $(BACKEND) && uv run pytest --cov --cov-report=term

test-frontend:
	cd $(FRONTEND) && pnpm exec vitest run

build:
	cd $(FRONTEND) && pnpm build
