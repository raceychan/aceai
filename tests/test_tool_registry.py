import sys
from uuid import uuid4

from aceai.tools import tool
from aceai.tools._tool_sig import Annotated, spec
from aceai.tools.registry import ToolRegistry


def test_tool_registry_groups_tools_by_tag() -> None:
    @tool(tags=["math", "arithmetic"])
    def add(
        a: Annotated[int, spec(description="a")],
        b: Annotated[int, spec(description="b")],
    ) -> int:
        return a + b

    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry(add, ping)

    math_tools = registry.get_tools("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "add"

    arithmetic_tools = registry.get_tools("arithmetic")
    assert len(arithmetic_tools) == 1
    assert arithmetic_tools[0].name == "add"

    public_tools = registry.get_tools("public")
    assert len(public_tools) == 1
    assert public_tools[0].name == "ping"


def test_tool_registry_remove_tool_drops_from_all_tags() -> None:
    @tool(tags=["math", "arithmetic"])
    def add(
        a: Annotated[int, spec(description="a")],
        b: Annotated[int, spec(description="b")],
    ) -> int:
        return a + b

    registry = ToolRegistry(add)
    registry.remove_tool(add)

    assert registry.get_tools("math") == []
    assert registry.get_tools("arithmetic") == []


def test_tool_registry_register_module_supports_lazy_import(
    tmp_path, monkeypatch
) -> None:
    module_name = f"tmp_tools_{uuid4().hex}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "from aceai.tools import tool",
                "from aceai.tools._tool_sig import Annotated, spec",
                "",
                '@tool(tags=[\"math\"])',
                "def add(a: Annotated[int, spec(description=\"a\")], b: Annotated[int, spec(description=\"b\")]) -> int:",
                "    return a + b",
                "",
                "not_a_tool = 123",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    registry = ToolRegistry()
    registry.register_module(module_path, lazy=True)
    assert registry.get_tools("math") == []

    registry.load_modules()
    math_tools = registry.get_tools("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "add"


def test_tool_registry_register_module_loads_immediately_for_str_path(
    tmp_path, monkeypatch
) -> None:
    module_name = f"tmp_tools_{uuid4().hex}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "from aceai.tools import tool",
                "from aceai.tools._tool_sig import Annotated, spec",
                "",
                '@tool(tags=["math"])',
                "def add(a: Annotated[int, spec(description=\"a\")], b: Annotated[int, spec(description=\"b\")]) -> int:",
                "    return a + b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    registry = ToolRegistry()
    registry.register_module(str(module_path), lazy=False)

    math_tools = registry.get_tools("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "add"


def test_tool_registry_register_module_accepts_module_name_string(
    tmp_path, monkeypatch
) -> None:
    package_name = f"tmp_pkg_{uuid4().hex}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    module_name = f"{package_name}.mod"
    module_path = package_dir / "mod.py"
    module_path.write_text(
        "\n".join(
            [
                "from aceai.tools import tool",
                "from aceai.tools._tool_sig import Annotated, spec",
                "",
                '@tool(tags=["math"])',
                "def add(a: Annotated[int, spec(description=\"a\")], b: Annotated[int, spec(description=\"b\")]) -> int:",
                "    return a + b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    registry = ToolRegistry()
    registry.register_module(module_name, lazy=False)

    math_tools = registry.get_tools("math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "add"


def test_tool_registry_register_module_raises_for_missing_py_file_path(tmp_path) -> None:
    registry = ToolRegistry()
    missing = tmp_path / "missing_module.py"

    try:
        registry.register_module(str(missing), lazy=True)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_tool_registry_register_module_tracks_registered_module_names(
    tmp_path, monkeypatch
) -> None:
    module_name_1 = f"tmp_tools_{uuid4().hex}"
    module_path_1 = tmp_path / f"{module_name_1}.py"
    module_path_1.write_text("", encoding="utf-8")

    module_name_2 = f"tmp_tools_{uuid4().hex}"
    module_path_2 = tmp_path / f"{module_name_2}.py"
    module_path_2.write_text("", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolRegistry()
    registry.register_module(module_path_1, lazy=True)
    registry.register_module(str(module_path_2), lazy=True)

    assert registry.modules == [module_name_1, module_name_2]
