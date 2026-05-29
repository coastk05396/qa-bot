SHELL := /bin/bash

KB_DIR := knowledge_base_qa_bot
MARKDOWN_DIR := $(KB_DIR)/scaffold/markdown_kb
VECTOR_DIR := $(KB_DIR)/scaffold/vector_rag
MARKDOWN_PORT ?= 8000
VECTOR_PORT ?= 8001

.PHONY: help install markdown vector run-both stop-both test-markdown test-vector health

help:
	@echo "Targets:"
	@echo "  make install       Install dependencies for both backends (.venv/requirements)"
	@echo "  make markdown      Start the Markdown KB backend on port $(MARKDOWN_PORT)"
	@echo "  make vector        Start the Vector RAG backend on port $(VECTOR_PORT)"
	@echo "  make run-both      Start both backends and stop both on Ctrl+C"
	@echo "  make stop-both     Stop processes listening on ports $(MARKDOWN_PORT) and $(VECTOR_PORT)"
	@echo "  make test-markdown Run markdown_kb tests"
	@echo "  make test-vector   Run vector_rag tests"
	@echo "  make health        Check both health endpoints"

install:
	@echo "🚀 Setting up Markdown KB environment..."
	cd $(MARKDOWN_DIR) && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
	@echo "🚀 Setting up Vector RAG environment..."
	cd $(VECTOR_DIR) && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
	@echo "✅ All dependencies installed successfully!"

markdown:
	cd $(MARKDOWN_DIR) && ./.venv/bin/uvicorn app.main:app --reload --port $(MARKDOWN_PORT)

vector:
	cd $(VECTOR_DIR) && ./.venv/bin/uvicorn app.main:app --reload --port $(VECTOR_PORT)

run-both:
	@set -euo pipefail; \
	markdown_pid=""; \
	vector_pid=""; \
	cleanup() { \
		if [[ -n "$$markdown_pid" ]] && kill -0 "$$markdown_pid" 2>/dev/null; then kill "$$markdown_pid"; fi; \
		if [[ -n "$$vector_pid" ]] && kill -0 "$$vector_pid" 2>/dev/null; then kill "$$vector_pid"; fi; \
	}; \
	trap cleanup INT TERM EXIT; \
	if lsof -tiTCP:$(MARKDOWN_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(MARKDOWN_PORT) is already in use. Stop the running server or run 'make health'."; \
		exit 1; \
	fi; \
	if lsof -tiTCP:$(VECTOR_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(VECTOR_PORT) is already in use. Stop the running server or run 'make health'."; \
		exit 1; \
	fi; \
	(cd $(MARKDOWN_DIR) && ./.venv/bin/uvicorn app.main:app --reload --port $(MARKDOWN_PORT)) & \
	markdown_pid=$$!; \
	(cd $(VECTOR_DIR) && ./.venv/bin/uvicorn app.main:app --reload --port $(VECTOR_PORT)) & \
	vector_pid=$$!; \
	wait $$markdown_pid $$vector_pid

stop-both:
	@set -euo pipefail; \
	pids="$$(printf '%s\n' "$$(lsof -tiTCP:$(MARKDOWN_PORT) -sTCP:LISTEN 2>/dev/null || true)" "$$(lsof -tiTCP:$(VECTOR_PORT) -sTCP:LISTEN 2>/dev/null || true)" | awk 'NF' | sort -u)"; \
	if [[ -z "$$pids" ]]; then \
		echo "No backend processes are listening on ports $(MARKDOWN_PORT) or $(VECTOR_PORT)."; \
		exit 0; \
	fi; \
	echo "Stopping backend processes: $$pids"; \
	kill -TERM $$pids 2>/dev/null || true; \
	for _ in 1 2 3 4 5; do \
		remaining="$$(printf '%s\n' "$$(lsof -tiTCP:$(MARKDOWN_PORT) -sTCP:LISTEN 2>/dev/null || true)" "$$(lsof -tiTCP:$(VECTOR_PORT) -sTCP:LISTEN 2>/dev/null || true)" | awk 'NF' | sort -u)"; \
		if [[ -z "$$remaining" ]]; then \
			echo "Backend ports are clear."; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	remaining="$$(printf '%s\n' "$$(lsof -tiTCP:$(MARKDOWN_PORT) -sTCP:LISTEN 2>/dev/null || true)" "$$(lsof -tiTCP:$(VECTOR_PORT) -sTCP:LISTEN 2>/dev/null || true)" | awk 'NF' | sort -u)"; \
	if [[ -n "$$remaining" ]]; then \
		echo "Force stopping backend processes: $$remaining"; \
		kill -KILL $$remaining; \
	fi; \
	echo "Backend ports are clear."

test-markdown:
	cd $(MARKDOWN_DIR) && ./.venv/bin/python -m unittest tests.test_app

test-vector:
	cd $(VECTOR_DIR) && ./.venv/bin/python -m unittest tests.test_app

health:
	@curl --max-time 3 -fsS http://127.0.0.1:$(MARKDOWN_PORT)/health && echo
	@curl --max-time 3 -fsS http://127.0.0.1:$(VECTOR_PORT)/health && echo
