---
name: developer
description: Practical software development workflow for editing, testing, searching, and explaining code changes.
---
# Developer Workflow

Use this skill when the user asks to inspect, modify, test, or explain a codebase.

Start by locating the relevant files with `search_text` or `list_directory`.
Read the smallest useful set of files before editing.
Use `replace_text_in_file` for precise edits and `write_text_file` when creating a new file or replacing a whole file.
Run tests with the repository's declared test command after behavior changes.
Report the files changed and the exact verification that ran.

