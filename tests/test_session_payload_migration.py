import json

from scripts.migrate_session_payloads import migrate_session_payloads


def test_migrate_session_payloads_updates_jsonl(tmp_path) -> None:
    root = tmp_path / "sessions"
    files_dir = root / "files"
    files_dir.mkdir(parents=True)
    event_path = files_dir / "session-1.events.jsonl"
    event_path.write_text(
        "\n".join(
            [
                (
                    '{"kind":"user_message","payload":{"content":"read",'
                    '"citations":[{"content":"Local file: /tmp/a.md",'
                    '"origin":{"kind":"file","path":"/tmp/a.md"}},'
                    '{"content":"quoted","origin":{"kind":"ad_hoc","label":"note"}}]}}'
                ),
                (
                    '{"kind":"tool_result","payload":{"content":"completed",'
                    '"tool_name":"search_text","tool_call_id":"call-1",'
                    '"tool_arguments":"{}","output":"full","model_output":"small",'
                    '"status":"completed"}}'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = migrate_session_payloads(root)

    assert result.files_scanned == 1
    assert result.files_changed == 1
    assert result.payloads_changed == 3
    assert event_path.with_suffix(".jsonl.payload-migration.bak").exists()
    lines = event_path.read_text(encoding="utf-8").splitlines()
    user_message = json.loads(lines[0])
    tool_result = json.loads(lines[1])
    assert user_message["payload"]["citations"] == [
        {"origin": {"kind": "file", "path": "/tmp/a.md"}, "quote": "/tmp/a.md"},
        {"origin": {"kind": "ad_hoc", "label": "note"}, "quote": "quoted"},
    ]
    assert "content" not in user_message["payload"]["citations"][0]
    assert "content" not in user_message["payload"]["citations"][1]
    assert tool_result["payload"]["truncated_output"] == "small"
    assert "model_output" not in tool_result["payload"]


def test_migrate_session_payloads_dry_run_does_not_write(tmp_path) -> None:
    root = tmp_path / "sessions"
    files_dir = root / "files"
    files_dir.mkdir(parents=True)
    event_path = files_dir / "session-1.events.jsonl"
    original = (
        '{"kind":"user_message","payload":{"content":"read",'
        '"citations":[{"content":"quoted","origin":{"kind":"ad_hoc","label":"note"}}]}}\n'
    )
    event_path.write_text(original, encoding="utf-8")

    result = migrate_session_payloads(root, dry_run=True)

    assert result.files_scanned == 1
    assert result.files_changed == 1
    assert result.payloads_changed == 1
    assert event_path.read_text(encoding="utf-8") == original
    assert not event_path.with_suffix(".jsonl.payload-migration.bak").exists()
