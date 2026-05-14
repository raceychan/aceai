import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "aceai"


def test_llm_layer_does_not_import_framework_or_product_layers() -> None:
    forbidden = ("aceai.core", "agent_core")

    violations = _find_forbidden_imports(PACKAGE_ROOT / "llm", forbidden)

    assert violations == []


def test_framework_layer_does_not_import_product_layer() -> None:
    framework_paths = [
        PACKAGE_ROOT / "core",
    ]

    violations: list[str] = []
    for path in framework_paths:
        violations.extend(_find_forbidden_imports(path, ("agent_core",)))

    assert violations == []


def _find_forbidden_imports(path: Path, forbidden: tuple[str, ...]) -> list[str]:
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    violations: list[str] = []
    for file_path in files:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in ast.walk(tree):
            imported_name = _imported_name(node)
            if imported_name is None:
                continue
            for forbidden_name in forbidden:
                if imported_name == forbidden_name or imported_name.startswith(
                    f"{forbidden_name}."
                ):
                    relative_path = file_path.relative_to(PACKAGE_ROOT.parent)
                    violations.append(f"{relative_path}: {imported_name}")
    return violations


def _imported_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None
