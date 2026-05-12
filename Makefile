.PHONY: test debug cov demo server
test:
	uv run pytest -v tests/

debug:
	uv run pytest -m debug

cov:
	uv run pytest --cov=aceai --cov-report=term-missing

demo:
	uv run python examples/logistics_agent_demo.py

tui:
	uv run aceai

HOST ?= 127.0.0.1
PORT ?= 8765
server:
	uv run --extra gui aceai-gui --host $(HOST) --port $(PORT)


VERSION ?=
DEFAULT_BASE_BRANCH := $(strip $(shell git remote show origin 2>/dev/null | sed -n '/HEAD branch/s/.*: //p'))
BASE_BRANCH ?= $(if $(DEFAULT_BASE_BRANCH),$(DEFAULT_BASE_BRANCH),main)
RELEASE_SCRIPT := scripts.release

INCREMENT ?= patch
.PHONY: new-branch delete-branch
new-branch:
	@uv run --group dev python -m $(RELEASE_SCRIPT) new-branch --increment $(INCREMENT) --base-branch $(BASE_BRANCH)

delete-branch:
	@if [ -z "$(VERSION)" ]; then \
		echo "VERSION must be set for delete-branch target"; \
		exit 1; \
	fi
	@uv run --group dev python -m $(RELEASE_SCRIPT) delete-branch --version $(VERSION)
