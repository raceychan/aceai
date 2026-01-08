import pytest
from ididi import use
from msgspec import DecodeError, ValidationError

from aceai.errors import UnannotatedToolParamError
from aceai.tools import tool
from aceai.tools._tool_sig import Annotated, spec


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


def test_tool_spec_cached_until_reset() -> None:
    greet_tool = tool(greet)

    first_spec = greet_tool.tool_spec
    second_spec = greet_tool.tool_spec

    assert first_spec is second_spec

    greet_tool.reset_tool_schema()

    third_spec = greet_tool.tool_spec

    assert third_spec is not first_spec


def test_tool_schema_honors_aliases_in_properties() -> None:
    greet_tool = tool(greet)
    properties = greet_tool.tool_schema["parameters"]["properties"]

    assert "who" in properties
    assert properties["who"]["description"] == "Name to greet"
    assert "name" not in properties


def test_tool_from_func_uses_docstring_for_description() -> None:
    increment_tool = tool(increment)

    assert increment_tool.description == "Increment the supplied value by one."


def test_tool_description_prefers_meta_description_over_docstring() -> None:

    def docstring_func(
        value: Annotated[int, spec(description="Value")],
    ) -> int:
        """Docstring description."""
        return value

    my_tool = tool(description="Meta description.")(docstring_func)

    assert my_tool.description == "Meta description."


def test_tool_description_uses_docstring_when_meta_missing() -> None:

    def docstring_func(
        value: Annotated[int, spec(description="Value")],
    ) -> int:
        """Docstring description."""
        return value

    my_tool = tool()(docstring_func)

    assert my_tool.description == "Docstring description."


def test_tool_description_is_empty_when_both_missing() -> None:

    def no_docstring_func(
        value: Annotated[int, spec(description="Value")],
    ) -> int:
        return value

    my_tool = tool()(no_docstring_func)

    assert my_tool.description == ""


def test_tool_description_prefers_meta_description_when_func_has_no_docstring() -> None:

    def no_docstring_func(
        value: Annotated[int, spec(description="Value")],
    ) -> int:
        return value

    my_tool = tool(description="Meta description.")(no_docstring_func)

    assert my_tool.description == "Meta description."


@pytest.mark.parametrize(
    ("tool_kwargs", "docstring", "expected"),
    [
        ({"description": "Meta description."}, "Docstring description.", "Meta description."),
        ({}, "Docstring description.", "Docstring description."),
        ({}, None, ""),
        ({"description": "Meta description."}, None, "Meta description."),
    ],
)
def test_tool_schema_description_follows_meta_docstring_precedence(
    tool_kwargs: dict[str, str],
    docstring: str | None,
    expected: str,
) -> None:
    def func(value: Annotated[int, spec(description="Value")]) -> int:
        return value

    func.__doc__ = docstring

    my_tool = tool(**tool_kwargs)(func) if tool_kwargs else tool()(func)

    assert my_tool.tool_schema["description"] == expected


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

    with pytest.raises(
        UnannotatedToolParamError,
        match="Tool parameter 'x' must use typing.Annotated",
    ):
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

    my_tool = tool(description="Fetch user info")(func)

    assert "user_service" in my_tool.signature.dep_nodes
    assert "user_id" in my_tool.signature.params
