from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Annotated as Annotated, Callable
from typing import Any, TypedDict, Unpack, get_args, get_origin

from ididi import USE_FACTORY_MARK, DependentNode
from ididi.utils.typing_utils import flatten_annotated
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


def get_param_meta(param: Parameter) -> list[Any] | None:
    if not param.annotation:
        return None

    if not (t_origin := get_origin(param.annotation)) or t_origin is not Annotated:
        return None

    param_meta = flatten_annotated(param.annotation)
    return param_meta


def get_param_spec(param_metas: list[Any] | None) -> ParamSpec | None:
    if not param_metas:
        return None

    for meta in param_metas:
        if isinstance(meta, ParamSpec):
            return meta


def get_dep_node(param_metas: list[Any], param_type: Any) -> DependentNode | None:
    try:
        mark_idx = param_metas.index(USE_FACTORY_MARK)
    except ValueError as ve:
        return None

    dep = param_metas[mark_idx + 1]
    dep = dep or param_type
    dep_node = DependentNode.from_node(dep)
    return dep_node


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
    dep_nodes: dict[str, Callable[..., Any]]
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
        dep_nodes: dict[str, Callable[..., Any]] = {}
        for param in func_sig.parameters.values():
            if not param.annotation:
                raise ValueError(f"Parameter {param.name!r} is missing type annotation")

            param_metas = get_param_meta(param)
            if not param_metas:
                continue

            param_spec = get_param_spec(param_metas)
            if param_spec:
                tool_param = ToolParam.from_param(param, param_spec)
                params[param.name] = tool_param

            param_type = get_args(param.annotation)[0]
            dep_node = get_dep_node(param_metas, param_type)
            if dep_node:
                if param.name in params:
                    raise ValueError(
                        f"Parameter {param.name!r} is defined as both ToolParam and DependentNode"
                    )
                dep_nodes[param.name] = dep_node.factory

        return_type = (
            func_sig.return_annotation
            if func_sig.return_annotation is not Signature.empty
            else MISSING
        )

        return cls(params=params, dep_nodes=dep_nodes, return_type=return_type)
