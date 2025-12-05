from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Annotated as Annotated
from typing import Any, TypedDict, Unpack, get_args, get_origin

from msgspec import Meta, Struct, defstruct

from aceai.interface import MISSING, JsonSchema, Maybe, is_present
from aceai.tools.schema_generator import inline_schema


class ParamConstraint(TypedDict, total=False):
    gt: int | float
    ge: int | float
    lt: int | float
    le: int | float
    multiple_of: int | float
    pattern: str
    min_length: int
    max_length: int
    tz: bool
    extra: dict


@dataclass(frozen=True, kw_only=True, slots=True)
class ParamSpec[T]:
    alias: Maybe[str] = MISSING
    description: str
    required: Maybe[bool] = MISSING
    annotation: Maybe[type[T]] = MISSING
    title: Maybe[str] = MISSING
    examples: list
    extra_json_schema: dict
    constraint: ParamConstraint


def spec[T](
    description: str = "",
    alias: Maybe[str] = MISSING,
    required: Maybe[bool] = MISSING,
    annotation: Maybe[type[T]] = MISSING,
    title: Maybe[str] = MISSING,
    examples: Maybe[list] = MISSING,
    extra_json_schema: Maybe[dict] = MISSING,
    **constraint: Unpack[ParamConstraint],
) -> ParamSpec[T]:
    """
    Args:
        alias: The name of the parameter in the tool schema.
        description: A description of the parameter.
        required: Whether the parameter is required, if not specified, defaults to whether a default value for the param is provided.
        annotation: The type annotation override of the parameter. default is to use the annotated type.
        constraint: Additional constraints for the parameter, see `ParamConstraint` for details.
    """
    if not is_present(examples):
        examples = []
    if not is_present(extra_json_schema):
        extra_json_schema = {}
    return ParamSpec(
        alias=alias,
        description=description,
        required=required,
        annotation=annotation,
        title=title,
        examples=examples,
        extra_json_schema=extra_json_schema,
        constraint=constraint,
    )


def get_param_spec(param: Parameter) -> ParamSpec | None:
    if not param.annotation:
        return None

    if not (t_origin := get_origin(param.annotation)) or t_origin is not Annotated:
        return None

    param_meta = param.annotation.__metadata__
    for meta in param_meta:
        if isinstance(meta, ParamSpec):
            return meta


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolParam[T]:
    name: str
    alias: str
    required: bool
    type_: type[T]
    annotation: Any
    default: Maybe[T] = MISSING
    schema: JsonSchema

    @classmethod
    def from_param(
        cls,
        param: Parameter,
        param_spec: ParamSpec[T],
    ) -> "ToolParam[T]":
        default = MISSING if param.default is Parameter.empty else param.default
        alias = param_spec.alias if is_present(param_spec.alias) else param.name

        annotation = (
            param_spec.annotation
            if is_present(param_spec.annotation)
            else param.annotation
        )
        param_type = get_args(annotation)[0]
        required = (
            param_spec.required
            if is_present(param_spec.required)
            else default is MISSING
        )

        param_schema = inline_schema(param_type, default)
        param_schema["description"] = param_spec.description

        if param_spec.examples:
            param_schema["examples"] = param_spec.examples
        if param_spec.extra_json_schema:
            param_schema.update(param_spec.extra_json_schema)
        if param_spec.constraint:
            annotation = Annotated[param_type, Meta(**param_spec.constraint)]

        return cls(
            name=param.name,
            alias=alias,
            type_=param_type,
            annotation=annotation,
            required=required,
            default=default,
            schema=param_schema,
        )


@dataclass(kw_only=True, slots=True)
class ToolSignature:
    params: dict[str, ToolParam]
    return_type: Maybe[type]

    def generate_params_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self.params.values():
            properties[param.alias] = param.schema
            if param.required:
                required.append(param.alias)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            parameters["required"] = required

        return parameters

    @property
    def virtual_struct(self) -> type[Struct]:
        fields = []

        for param in self.params.values():
            field_tuple = (param.alias, param.annotation)
            if is_present(param.default):
                field_tuple += (param.default,)
            fields.append(field_tuple)

        typed_struct = defstruct(f"ToolParams", fields)
        return typed_struct

    @classmethod
    def from_signature(cls, func_sig: Signature) -> "ToolSignature":
        params: dict[str, ToolParam] = {}
        for param in func_sig.parameters.values():
            if not param.annotation:
                raise ValueError(f"Parameter {param.name!r} is missing type annotation")
            param_spec = get_param_spec(param)
            if param_spec is None:
                continue
            tool_param = ToolParam.from_param(param, param_spec)
            params[param.name] = tool_param

        return_type = (
            func_sig.return_annotation
            if func_sig.return_annotation is not Signature.empty
            else MISSING
        )

        return cls(params=params, return_type=return_type)
