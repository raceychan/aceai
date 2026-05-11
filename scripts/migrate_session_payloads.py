import argparse
import json
from pathlib import Path
from typing import Any


class MigrationResult:
    def __init__(
        self,
        *,
        files_scanned: int,
        files_changed: int,
        payloads_changed: int,
    ) -> None:
        self.files_scanned = files_scanned
        self.files_changed = files_changed
        self.payloads_changed = payloads_changed


def default_sessions_root() -> Path:
    return Path.home() / ".aceai" / "sessions"


def migrate_session_payloads(
    root: Path,
    *,
    dry_run: bool = False,
) -> MigrationResult:
    files_dir = root / "files"
    if not files_dir.is_dir():
        raise ValueError(f"Session files directory not found: {files_dir}")

    files_scanned = 0
    files_changed = 0
    payloads_changed = 0
    for path in sorted(files_dir.glob("*.events.jsonl")):
        files_scanned += 1
        next_lines, changed_count = _migrated_event_file_lines(path)
        if changed_count == 0:
            continue
        files_changed += 1
        payloads_changed += changed_count
        if dry_run:
            continue
        backup_path = path.with_suffix(path.suffix + ".payload-migration.bak")
        if not backup_path.exists():
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        temp_path = path.with_suffix(path.suffix + ".payload-migration.tmp")
        temp_path.write_text("".join(next_lines), encoding="utf-8")
        temp_path.replace(path)

    return MigrationResult(
        files_scanned=files_scanned,
        files_changed=files_changed,
        payloads_changed=payloads_changed,
    )


def _migrated_event_file_lines(path: Path) -> tuple[list[str], int]:
    next_lines: list[str] = []
    changed_count = 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.strip() == "":
            next_lines.append(line)
            continue
        newline = "\n" if line.endswith("\n") else ""
        raw = line[:-1] if newline else line
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("Session event must be a mapping")
        changed_count += _migrate_event_payload(payload)
        next_lines.append(json.dumps(payload, ensure_ascii=False) + newline)
    return next_lines, changed_count


def _migrate_event_payload(event: dict[str, Any]) -> int:
    payload = event["payload"]
    if not isinstance(payload, dict):
        raise TypeError("Session event payload must be a mapping")
    changed = _migrate_tool_result_payload(event, payload)
    citations = payload.get("citations")
    if citations is None:
        return changed
    if not isinstance(citations, list):
        raise TypeError("Session event citations must be a list")
    for citation in citations:
        if not isinstance(citation, dict):
            raise TypeError("Session citation must be a mapping")
        if _migrate_citation_payload(citation):
            changed += 1
    return changed


def _migrate_tool_result_payload(event: dict[str, Any], payload: dict[str, Any]) -> int:
    if event["kind"] != "tool_result":
        return 0
    if "truncated_output" in payload and "model_output" not in payload:
        return 0
    if "model_output" in payload:
        truncated_output = payload["model_output"]
    else:
        truncated_output = payload["output"]
    if type(truncated_output) is not str:
        raise TypeError("Tool result truncated output must be str")
    payload["truncated_output"] = truncated_output
    payload.pop("model_output", None)
    return 1


def _migrate_citation_payload(citation: dict[str, Any]) -> bool:
    origin = citation["origin"]
    if not isinstance(origin, dict):
        raise TypeError("Citation origin must be a mapping")
    if origin["kind"] == "file":
        quote = origin["path"]
        if type(quote) is not str:
            raise TypeError("File citation path must be str")
    else:
        quote = citation.get("quote", citation.get("content"))
        if type(quote) is not str:
            raise TypeError("Citation quote must be str")

    changed = citation.get("quote") != quote or "content" in citation
    citation["quote"] = quote
    citation.pop("content", None)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-off migration for pre-0.2.27 AceAI session JSONL payloads."
    )
    parser.add_argument(
        "sessions_root",
        nargs="?",
        type=Path,
        default=default_sessions_root(),
        help="AceAI sessions root. Defaults to ~/.aceai/sessions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    args = parser.parse_args()
    result = migrate_session_payloads(args.sessions_root, dry_run=args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"{action} {result.payloads_changed} session payloads "
        f"in {result.files_changed}/{result.files_scanned} event files."
    )


if __name__ == "__main__":
    main()
