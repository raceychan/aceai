import pytest
from ididi import use
from msgspec import DecodeError, ValidationError

from aceai.tools import tool
from aceai.tools._param import Annotated, spec


def multiply(
    x: Annotated[int, spec(description="Left factor")],
    y: Annotated[int, spec(description="Right factor")],
) -> int:
    """Multiply two integers."""
    return x * y


def greet(
    name: Annotated[str, spec(description="Name to greet", alias="who")],
    punctuation: Annotated[str, spec(description="Sentence ending")] = "!",
) -> str:
    """Return a friendly greeting."""
    return f"Hello {name}{punctuation}"


def increment(
    value: Annotated[int, spec(description="Value to bump")],
) -> int:
    """Increment the supplied value by one."""
    return value + 1


def test_tool_call_and_encode_return() -> None:
    multiply_tool = tool(multiply)

    assert multiply_tool(3, 4) == 12
    assert multiply_tool.encode_return(12) == "12"


def test_tool_decode_params_uses_virtual_struct_defaults() -> None:
    greet_tool = tool(greet)

    decoded = greet_tool.decode_params('{"who":"Ava"}')

    assert decoded == {"who": "Ava", "punctuation": "!"}


def test_tool_schema_cached_until_reset() -> None:
    greet_tool = tool(greet)

    first_schema = greet_tool.tool_schema
    second_schema = greet_tool.tool_schema

    assert first_schema is second_schema

    greet_tool.reset_tool_schema()

    third_schema = greet_tool.tool_schema

    assert third_schema is not first_schema
    assert third_schema == first_schema


def test_tool_schema_honors_aliases_in_properties() -> None:
    greet_tool = tool(greet)
    properties = greet_tool.tool_schema["parameters"]["properties"]

    assert "who" in properties
    assert properties["who"]["description"] == "Name to greet"
    assert "name" not in properties


def test_tool_from_func_uses_docstring_for_description() -> None:
    increment_tool = tool(increment)

    assert increment_tool.description == "Increment the supplied value by one."


def test_tool_decode_params_invalid_json_raises_decode_error() -> None:
    greet_tool = tool(greet)

    with pytest.raises(DecodeError):
        greet_tool.decode_params('{"who"')


def test_tool_decode_params_missing_required_field() -> None:
    greet_tool = tool(greet)

    with pytest.raises(ValidationError):
        greet_tool.decode_params('{"punctuation":"?"}')


def test_tool_from_func_raises_when_param_annotation_is_none() -> None:

    def bad_function(x: None, y: Annotated[int, spec(description="Not used")]) -> int:
        return y

    with pytest.raises(ValueError, match="Parameter 'x' is missing type annotation"):
        tool(bad_function)


class UserService: ...


@pytest.mark.debug
def test_tool_with_both_dep_and_tool_params():

    def func(
        user_service: Annotated[
            UserService,
            use(),
        ],
        user_id: Annotated[int, spec(description="The ID of the user")],
    ) -> str:
        return f"User {user_id}"

    my_tool = tool(func)

    assert "user_service" in my_tool.signature.dep_nodes
    assert "user_id" in my_tool.signature.params
