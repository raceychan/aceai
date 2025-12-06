#!/usr/bin/env python3
"""Automate release workflow using GitPython."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from git import GitCommandError, Repo
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "aceai" / "__init__.py"


def detect_default_base_branch(repo: Repo) -> str:
    try:
        ref = repo.git.symbolic_ref("refs/remotes/origin/HEAD")
    except GitCommandError:
        return "main"
    return ref.rsplit("/", 1)[-1]


def ensure_local_branch(repo: Repo, branch_name: str) -> None:
    if branch_name in repo.heads:
        return
    origin = repo.remotes.origin
    origin.fetch(branch_name)


def ensure_release_branch(repo: Repo, base_branch: str, release_branch: str) -> None:
    if repo.head.is_detached:
        raise SystemExit("Repository HEAD is detached. Checkout a branch first.")
    if repo.active_branch.name == release_branch:
        return
    if release_branch in repo.heads:
        repo.git.checkout(release_branch)
        return
    ensure_local_branch(repo, base_branch)
    repo.git.checkout(base_branch)
    repo.git.checkout("-b", release_branch)
    print(f"Created release branch {release_branch} from {base_branch}")


def read_current_version() -> str:
    namespace: dict[str, object] = {}
    exec(VERSION_FILE.read_text(), namespace)  # noqa: S102 - trusted file
    return str(namespace["__version__"])


def write_version(new_version: str) -> None:
    pattern = re.compile(r"__version__\s*=\s*\"([^\"]+)\"")
    content = VERSION_FILE.read_text()
    if not pattern.search(content):
        raise SystemExit("Could not locate __version__ assignment in aceai/__init__.py")
    updated = pattern.sub(f'__version__ = "{new_version}"', content, count=1)
    VERSION_FILE.write_text(updated)
    print(f"Updated __version__ to {new_version}")


def ensure_version_order(
    current: str, target: str, *, skip_update: bool
) -> tuple[Version, Version]:
    current_v = Version(current)
    target_v = Version(target)
    if skip_update:
        if current_v != target_v:
            raise SystemExit(
                "--skip-version-update set but __version__ does not match target"
            )
    else:
        if target_v <= current_v:
            raise SystemExit(
                f"Target version {target_v} must be greater than current {current_v}"
            )
    return current_v, target_v


def bump_version(version: Version, increment: str) -> Version:
    if increment == "patch":
        return Version(f"{version.major}.{version.minor}.{version.micro + 1}")
    if increment == "minor":
        return Version(f"{version.major}.{version.minor + 1}.0")
    if increment == "major":
        return Version(f"{version.major + 1}.0.0")
    raise ValueError(f"Unsupported increment: {increment}")


def stage_and_commit(repo: Repo, version: str) -> None:
    repo.git.add(all=True)
    staged = repo.git.diff("--cached", "--name-only")
    if not staged.strip():
        print("No staged changes; skipping commit.")
        return
    repo.index.commit(f"Release version {version}")
    print("Created release commit")


def merge_into_base(repo: Repo, base_branch: str, source_branch: str) -> None:
    ensure_local_branch(repo, base_branch)
    repo.git.checkout(base_branch)
    repo.git.merge(source_branch)
    print(f"Merged {source_branch} into {base_branch}")


def tag_release(repo: Repo, version: str) -> str:
    tag_name = f"v{version}"
    if any(tag.name == tag_name for tag in repo.tags):
        raise SystemExit(f"Tag {tag_name} already exists")
    repo.create_tag(tag_name, message=f"Release version {version}")
    print(f"Tagged release {tag_name}")
    return tag_name


def push_changes(repo: Repo, base_branch: str, tag_name: str) -> None:
    origin = repo.remotes.origin
    origin.push(base_branch)
    origin.push(tag_name)
    print("Pushed base branch and tag to origin")


def run_build() -> None:
    subprocess.run(["uv", "build"], cwd=PROJECT_ROOT, check=True)
    print("uv build completed")


def create_new_branch(repo: Repo, *, base_branch: str | None, increment: str) -> None:
    base = base_branch or detect_default_base_branch(repo)
    ensure_local_branch(repo, base)
    repo.git.checkout(base)
    current_version = Version(read_current_version())
    next_version = bump_version(current_version, increment)
    new_branch_name = f"version/{next_version}"
    if new_branch_name in repo.heads:
        raise SystemExit(f"Branch {new_branch_name} already exists")
    repo.git.checkout("-b", new_branch_name)
    print(f"Created branch {new_branch_name} off {base}")


def delete_branch(repo: Repo, version: str) -> None:
    branch_name = f"version/{version}"
    if branch_name in repo.heads:
        repo.git.branch("-D", branch_name)
        print(f"Deleted local branch {branch_name}")
    try:
        repo.git.push("origin", "--delete", branch_name)
        print(f"Deleted remote branch {branch_name}")
    except GitCommandError:
        print(f"Remote branch {branch_name} was not found; skipping remote delete")


def create_next_version_branch(
    repo: Repo, *, base_branch: str, released_version: str, increment: str = "patch"
) -> None:
    """Create version/<next> for the upcoming release cycle."""
    next_version = bump_version(Version(released_version), increment)
    next_branch = f"version/{next_version}"

    if next_branch in repo.heads:
        print(f"Next-version branch {next_branch} already exists; skipping creation.")
        return

    ensure_local_branch(repo, base_branch)
    repo.git.checkout(base_branch)

    try:
        repo.git.checkout("-b", next_branch)
    except GitCommandError as exc:
        print(f"Failed to create next-version branch {next_branch}: {exc}")
        repo.git.checkout(base_branch)
        return

    print(f"Created next-version branch {next_branch} from {base_branch}")


def handle_release(repo: Repo, args: argparse.Namespace) -> None:
    base_branch = args.base_branch or detect_default_base_branch(repo)
    release_branch = f"version/{args.version}"

    ensure_release_branch(repo, base_branch, release_branch)

    current_version = read_current_version()
    ensure_version_order(
        current_version, args.version, skip_update=args.skip_version_update
    )

    if not args.skip_version_update and current_version != args.version:
        write_version(args.version)

    stage_and_commit(repo, args.version)
    merge_into_base(repo, base_branch, release_branch)
    tag_name = tag_release(repo, args.version)
    push_changes(repo, base_branch, tag_name)
    run_build()
    create_next_version_branch(
        repo, base_branch=base_branch, released_version=args.version, increment="patch"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AceAI release automation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    release_parser = subparsers.add_parser(
        "release", help="Perform a full release (merge, tag, push, build)"
    )
    release_parser.add_argument(
        "--version", required=True, help="Target semantic version"
    )
    release_parser.add_argument(
        "--base-branch",
        help="Base branch to merge into (defaults to origin/HEAD or main)",
    )
    release_parser.add_argument(
        "--skip-version-update",
        action="store_true",
        help="Skip editing __version__ (expects file already updated)",
    )

    new_branch_parser = subparsers.add_parser(
        "new-branch", help="Create a new version/<x.y.z> branch from the base branch"
    )
    new_branch_parser.add_argument(
        "--base-branch",
        help="Base branch to branch off (defaults to origin/HEAD or main)",
    )
    new_branch_parser.add_argument(
        "--increment",
        choices=["patch", "minor", "major"],
        default="patch",
        help="Which part of the semantic version to bump (default: patch)",
    )

    delete_parser = subparsers.add_parser(
        "delete-branch", help="Delete a version/<x.y.z> branch locally and remotely"
    )
    delete_parser.add_argument(
        "--version",
        required=True,
        help="Version suffix of the branch to delete (e.g., 0.2.0)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo = Repo(PROJECT_ROOT)

    if args.command == "release":
        handle_release(repo, args)
    elif args.command == "new-branch":
        create_new_branch(
            repo,
            base_branch=getattr(args, "base_branch", None),
            increment=getattr(args, "increment", "patch"),
        )
    elif args.command == "delete-branch":
        delete_branch(repo, args.version)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
