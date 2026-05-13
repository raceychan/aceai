.PHONY: server schema test

HOST ?= 127.0.0.1
PORT ?= 8765

server:
	uv run python -m src.cli --host $(HOST) --port $(PORT)

schema:
	uv run python -m src.cli schema --out ../frontend/openapi.json

test:
	uv run pytest
