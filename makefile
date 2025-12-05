.PHONY: test debug cov
test:
	uv run pytest -v tests/

debug:
	uv run pytest -m debug

cov:
	uv run pytest --cov=aceai --cov-report=term-missing
