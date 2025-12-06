.PHONY: test debug cov
test:
	uv run pytest -v tests/

debug:
	uv run pytest -m debug

cov:
	uv run pytest --cov=aceai --cov-report=term-missing


# ================ CI =======================

VERSION ?= x.x.x
BRANCH = version/$(VERSION)
DEFAULT_BASE_BRANCH := $(strip $(shell git remote show origin 2>/dev/null | sed -n '/HEAD branch/s/.*: //p'))
BASE_BRANCH ?= $(if $(DEFAULT_BASE_BRANCH),$(DEFAULT_BASE_BRANCH),main)
SKIP_VERSION_UPDATE ?= 0

# Command definitions
UV_CMD = uv run
HATCH_VERSION_CMD = $(UV_CMD) hatch version
CURRENT_VERSION = $(shell $(HATCH_VERSION_CMD))

# Main release target
.PHONY: release check-branch check-version update-version git-commit git-merge git-tag git-push build pypi-release delete-branch new-branch new_branch

release: check-branch check-version update-version git-commit git-merge git-tag git-push build

# Version checking and updating
check-branch:
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "$(BRANCH)" ]; then \
		echo "Current branch is not $(BRANCH). Switching to it..."; \
		git switch -c $(BRANCH); \
		echo "Switched to $(BRANCH)"; \
	fi

check-version:
	@if [ "$(CURRENT_VERSION)" = "" ]; then \
		echo "Error: Unable to retrieve current version."; \
		exit 1; \
	fi
	$(call check_version_order,$(CURRENT_VERSION),$(VERSION))

update-version:
	@if [ "$(SKIP_VERSION_UPDATE)" = "1" ]; then \
		echo "Skipping version update because SKIP_VERSION_UPDATE=1."; \
	elif [ "$(CURRENT_VERSION)" = "$(VERSION)" ]; then \
		echo "Version already set to $(VERSION); skipping hatch version bump."; \
	else \
		echo "Updating hatch version to $(VERSION)..."; \
		$(HATCH_VERSION_CMD) $(VERSION); \
	fi

# Git operations
git-commit:
	@echo "Committing changes..."
	@git add -A
	@if git diff --cached --quiet; then \
		echo "No staged changes detected; skipping git commit."; \
	else \
		git commit -m "Release version $(VERSION)"; \
	fi

git-merge:
	@echo "Merging $(BRANCH) into $(BASE_BRANCH)..."
	@git checkout $(BASE_BRANCH)
	@git merge "$(BRANCH)"

git-tag:
	@echo "Tagging the release..."
	@git tag -a "v$(VERSION)" -m "Release version $(VERSION)"

git-push:
	@echo "Pushing to remote repository..."
	@git push origin $(BASE_BRANCH)
	@git push origin "v$(VERSION)"

# Build and publish operations
build:
	@echo "Building version $(VERSION)..."
	@uv build

pypi-release:
	@echo "Publishing to PyPI with skip-existing flag..."
	@uv run twine upload dist/* --skip-existing

# Branch management
delete-branch:
	@git branch -d $(BRANCH)
	@git push origin --delete $(BRANCH)

new-branch:
	@echo "Creating new version branch..."
	@if [ "$(CURRENT_VERSION)" = "" ]; then \
		echo "Error: Unable to retrieve current version."; \
		exit 1; \
	fi
	$(call increment_patch_version,$(CURRENT_VERSION))
	@echo "Creating branch version/$(NEW_VERSION)"
	@git checkout -b "version/$(NEW_VERSION)"

.PHONY: new_branch
new_branch:
	@echo "Creating the next version branch from $(BASE_BRANCH)..."
	@./scripts/create_version_branch.py --base $(BASE_BRANCH)
