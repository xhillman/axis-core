"""Internal tool input/output schema inference helpers."""

import inspect
from collections.abc import Callable
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from typing_extensions import NotRequired, Required


def generate_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON schema from function signature.

    Inspects the function's parameters and type hints to create a JSON schema
    compatible with LLM tool use APIs. Supports basic Python types, optionals,
    and Pydantic models.

    Type mapping:
        - str -> "string"
        - int -> "integer"
        - float -> "number"
        - bool -> "boolean"
        - list -> {"type": "array"}
        - dict -> {"type": "object"}
        - T | None -> Same as T, but not required
        - Pydantic models -> model_json_schema()

    Args:
        func: Function to generate schema for

    Returns:
        JSON schema dict with "properties" and "required" keys

    Raises:
        TypeError: If function uses unsupported Union types (multiple non-None types)
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "ctx":
            continue

        param_type = type_hints.get(param_name, Any)
        param_type, type_is_optional = _unwrap_optional_type(param_type)
        is_optional = param.default != inspect.Parameter.empty or type_is_optional

        properties[param_name] = _python_type_to_json_schema(param_type, path=param_name)
        if not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def generate_tool_output_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Infer descriptive output metadata from the function return annotation.

    The output schema is advisory metadata only. If the return annotation is absent or cannot be
    represented honestly with the supported JSON schema subset, this returns `{}` to indicate that
    the tool output is unconstrained.
    """
    type_hints = get_type_hints(func, include_extras=True)
    if "return" not in type_hints:
        return {}

    return_type = _unwrap_annotated_type(type_hints["return"])
    if return_type is Any:
        return {}

    if _is_union_type(return_type):
        union_members = tuple(_unwrap_annotated_type(arg) for arg in get_args(return_type))
        non_none_members = tuple(member for member in union_members if member is not type(None))
        if len(non_none_members) == 1 and len(non_none_members) != len(union_members):
            try:
                return {
                    "anyOf": [
                        _python_type_to_json_schema(non_none_members[0], path="return"),
                        {"type": "null"},
                    ]
                }
            except TypeError:
                return {}
        return {}

    try:
        return _python_type_to_json_schema(return_type, path="return")
    except TypeError:
        return {}


def _python_type_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert a Python type to JSON schema type."""
    python_type = _unwrap_annotated_type(python_type)
    python_type, _ = _unwrap_optional_type(python_type)

    if _is_union_type(python_type):
        raise TypeError(
            f"Unsupported Union type for {_describe_schema_path(path)}: {python_type}. "
            "Only Optional[T] (T | None) is supported."
        )

    if python_type is Any:
        return {"type": "object"}
    if python_type is type(None):
        return {"type": "null"}
    if python_type is str:
        return {"type": "string"}
    if python_type is int:
        return {"type": "integer"}
    if python_type is float:
        return {"type": "number"}
    if python_type is bool:
        return {"type": "boolean"}
    if python_type is list:
        return {"type": "array"}
    if python_type is dict:
        return {"type": "object"}

    if is_typeddict(python_type):
        return _typed_dict_to_json_schema(python_type, path=path)

    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type)
        if len(args) != 1 or args[0] is Any:
            return {"type": "array"}
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0], path=f"{path}[]"),
        }
    if origin is dict:
        args = get_args(python_type)
        if len(args) != 2:
            return {"type": "object"}

        key_type, value_type = args
        key_type = _unwrap_annotated_type(key_type)
        if key_type not in (str, Any):
            raise TypeError(
                f"Dictionary types for {_describe_schema_path(path)} must use string keys "
                f"for JSON schema compatibility: {python_type}."
            )

        schema: dict[str, Any] = {"type": "object"}
        if value_type is not Any:
            schema["additionalProperties"] = _python_type_to_json_schema(
                value_type,
                path=f"{path}.*",
            )
        return schema
    if origin is Literal:
        return _literal_to_json_schema(python_type, path=path)
    if origin in {Required, NotRequired}:
        args = get_args(python_type)
        if len(args) != 1:
            return {"type": "object"}
        return _python_type_to_json_schema(args[0], path=path)

    if hasattr(python_type, "model_json_schema"):
        pydantic_schema: dict[str, Any] = python_type.model_json_schema()
        return pydantic_schema

    return {"type": "object"}


def _unwrap_annotated_type(python_type: Any) -> Any:
    """Strip Annotated metadata while preserving the underlying schema type."""
    while get_origin(python_type) is Annotated:
        args = get_args(python_type)
        if not args:
            return Any
        python_type = args[0]
    return python_type


def _unwrap_optional_type(python_type: Any) -> tuple[Any, bool]:
    """Return the underlying type for Optional[T] and whether it was optional."""
    python_type = _unwrap_annotated_type(python_type)
    origin = get_origin(python_type)
    if origin not in {Union, UnionType}:
        return (python_type, False)

    args = tuple(_unwrap_annotated_type(arg) for arg in get_args(python_type))
    non_none_types = tuple(arg for arg in args if arg is not type(None))
    if len(non_none_types) == 1 and len(non_none_types) != len(args):
        return (non_none_types[0], True)
    return (python_type, False)


def _is_union_type(python_type: Any) -> bool:
    """Return True when the annotation is a non-optional union."""
    return get_origin(_unwrap_annotated_type(python_type)) in {Union, UnionType}


def _literal_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert Literal values to a deterministic enum schema."""
    values = get_args(python_type)
    if not values:
        return {"type": "object"}

    json_type = _literal_json_type(values[0])
    for value in values[1:]:
        value_type = _literal_json_type(value)
        if value_type != json_type:
            raise TypeError(
                f"Literal values for {_describe_schema_path(path)} must share the same JSON type: "
                f"{python_type}."
            )

    return {"type": json_type, "enum": list(values)}


def _literal_json_type(value: Any) -> str:
    """Map a literal value to its JSON schema scalar type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise TypeError(
        "Only string, boolean, integer, and number Literal values are supported for "
        f"tool schemas: {value!r}."
    )


def _typed_dict_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert a TypedDict class into an object schema."""
    annotations = get_type_hints(python_type, include_extras=True)
    required_keys = cast(set[str], set(getattr(python_type, "__required_keys__", set())))
    optional_keys = cast(set[str], set(getattr(python_type, "__optional_keys__", set())))
    total = bool(getattr(python_type, "__total__", True))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field_name, field_type in annotations.items():
        is_required = field_name in required_keys or (
            field_name not in optional_keys and total and field_name not in required_keys
        )

        origin = get_origin(field_type)
        if origin in {Required, NotRequired}:
            args = get_args(field_type)
            if len(args) == 1:
                field_type = args[0]
            is_required = origin is Required

        properties[field_name] = _python_type_to_json_schema(
            field_type,
            path=f"{path}.{field_name}",
        )
        if is_required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _describe_schema_path(path: str) -> str:
    """Return a readable label for error messages."""
    if any(token in path for token in (".", "[", "*")):
        return f"field '{path}'"
    return f"parameter '{path}'"
