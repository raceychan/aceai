.PHONY: test debug cov demo
test:
	uv run pytest -v tests/

debug:
	uv run pytest -m debug

cov:
	uv run pytest --cov=aceai --cov-report=term-missing

demo:
	uv run python demo.py


VERSION ?=
DEFAULT_BASE_BRANCH := $(strip $(shell git remote show origin 2>/dev/null | sed -n '/HEAD branch/s/.*: //p'))
BASE_BRANCH ?= $(if $(DEFAULT_BASE_BRANCH),$(DEFAULT_BASE_BRANCH),main)
SKIP_VERSION_UPDATE ?= 0

RELEASE_SCRIPT := scripts.release
RELEASE_CMD := uv run --group dev python -m $(RELEASE_SCRIPT) release --base-branch $(BASE_BRANCH)
ifneq ($(strip $(VERSION)),)
RELEASE_CMD += --version $(VERSION)
endif
ifeq ($(SKIP_VERSION_UPDATE),1)
RELEASE_CMD += --skip-version-update
endif

.PHONY: release
release:
	@$(RELEASE_CMD)

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

pypi-release:
	@echo "Publishing to PyPI with skip-existing flag..."
	@uv run twine upload dist/* --skip-existing
