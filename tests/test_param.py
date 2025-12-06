from inspect import signature

import pytest

from ididi import use

from aceai.interface import Record
from aceai.tools import tool
from aceai.tools._param import (
    Annotated,
    ToolParam,
    ToolSignature,
    get_param_meta,
    get_param_spec,
    spec,
)
from aceai.tools.schema_generator import MSGSPEC_REF_PREFIX, inline_schema


def add(
    x: Annotated[int, spec(description="The first number to add")],
    y: Annotated[int, spec(description="The second number to add")],
) -> int:
    return x + y


def test_build_tool_param() -> None:
    add_sig = signature(add)
    params = list(add_sig.parameters.values())
    assert get_param_spec(get_param_meta(params[0]))
    assert get_param_spec(get_param_meta(params[1]))


def test_inline_schema_expand_refs() -> None:

    class A(Record):
        x: int

    class B(Record):
        a: A
        y: str

    schema = inline_schema(B)

    def check_refs(node: dict) -> None:
        if isinstance(node, dict):
            if "$ref" in node:
                ref: str = node["$ref"]
                assert ref.startswith(MSGSPEC_REF_PREFIX)
            for value in node.values():
                check_refs(value)
        elif isinstance(node, list):
            for item in node:
                check_refs(item)

    assert schema == {
        "title": "B",
        "type": "object",
        "properties": {
            "a": {
                "title": "A",
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            "y": {"type": "string"},
        },
        "required": ["a", "y"],
    }


def test_tool_param_from_param() -> None:
    add_sig = signature(add)
    params = list(add_sig.parameters.values())
    param_spec_x = get_param_spec(get_param_meta(params[0]))
    param_spec_y = get_param_spec(get_param_meta(params[1]))
    assert param_spec_x is not None
    assert param_spec_y is not None

    tool_param_x = ToolParam.from_param(params[0], param_spec_x)
    tool_param_y = ToolParam.from_param(params[1], param_spec_y)

    assert tool_param_x.name == "x"
    assert tool_param_x.alias == "x"
    assert tool_param_x.required is True
    assert tool_param_x.type_ == int
    assert (
        tool_param_x.annotation
        == Annotated[int, spec(description="The first number to add")]
    )
    assert tool_param_x.schema == {
        "description": "The first number to add",
        "type": "integer",
    }

    assert tool_param_y.name == "y"
    assert tool_param_y.alias == "y"
    assert tool_param_y.required is True
    assert tool_param_y.type_ == int
    assert (
        tool_param_y.annotation
        == Annotated[int, spec(description="The second number to add")]
    )
    assert tool_param_y.schema == {
        "description": "The second number to add",
        "type": "integer",
    }


def test_tool_from_func() -> None:

    add_tool = tool(add)

    assert add_tool.name == "add"
    assert add_tool.description == ""
    assert len(add_tool.signature.params) == 2
    assert "x" in add_tool.signature.params
    assert "y" in add_tool.signature.params
    assert add_tool.signature.return_type == int


def test_get_param_meta_returns_none_for_missing_annotation() -> None:
    def bare(x):
        return x

    param = signature(bare).parameters["x"]

    assert get_param_meta(param) is None


def test_get_param_spec_returns_none_without_spec_marker() -> None:
    assert get_param_spec(["other metadata"]) is None


def test_tool_param_applies_examples_extra_schema_and_constraints() -> None:

    def constrained(
        limit: Annotated[
            int,
            spec(
                description="Limit value",
                examples=[1, 2],
                extra_json_schema={"format": "int64"},
                multiple_of=2,
                gt=0,
            ),
        ],
    ) -> int:
        return limit

    param = signature(constrained).parameters["limit"]
    param_spec = get_param_spec(get_param_meta(param))
    assert param_spec is not None

    tool_param = ToolParam.from_param(param, param_spec)

    assert tool_param.schema["examples"] == [1, 2]
    assert tool_param.schema["format"] == "int64"
    meta = tool_param.annotation.__metadata__[0]
    assert meta.multiple_of == 2
    assert meta.gt == 0


def test_tool_signature_raises_when_param_is_both_dependency_and_tool_param() -> None:

    class Repo(Record):
        name: str

    def provide_repo() -> Repo:
        return Repo(name="main")

    def both(
        repo: Annotated[
            Repo,
            spec(description="repo"),
            use(provide_repo),
        ],
    ) -> Repo:
        return repo

    with pytest.raises(
        ValueError, match="Parameter 'repo' is defined as both ToolParam and DependentNode"
    ):
        ToolSignature.from_signature(signature(both))
