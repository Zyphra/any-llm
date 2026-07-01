"""Helpers for adapting typed OpenResponses content across provider request paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

from pydantic import BaseModel

JsonDict: TypeAlias = dict[str, object]
RequestContentLike: TypeAlias = BaseModel | Sequence[BaseModel] | None


def dump_json_model(model: BaseModel) -> JsonDict:
    """Serialize a pydantic model to the JSON dict shape expected by providers."""
    return cast("JsonDict", model.model_dump(mode="json", by_alias=True, exclude_none=True))


def _unwrap_content_model(model: BaseModel) -> BaseModel | None:
    """Unwrap OpenResponses root-model wrappers.

    Returns:
    - ``None`` for string root models such as ``Content1(root="hello")``
    - the wrapped model for wrappers such as ``Content2(root=InputTextContentParam(...))``
    - the model itself for ordinary content-part models
    """
    root = getattr(model, "root", None)
    if isinstance(root, str):
        return None
    if isinstance(root, BaseModel):
        return root
    return model


def request_content_parts(content: RequestContentLike) -> list[BaseModel]:
    """Normalize OpenResponses content unions into a list of typed content-part models."""
    if content is None:
        return []
    if isinstance(content, BaseModel):
        part = _unwrap_content_model(content)
        return [part] if part is not None else []

    parts: list[BaseModel] = []
    for model in content:
        part = _unwrap_content_model(model)
        if part is not None:
            parts.append(part)
    return parts


def request_content_to_text(content: RequestContentLike) -> str:
    """Extract concatenated text from typed OpenResponses message or reasoning content."""
    if isinstance(content, BaseModel):
        root = getattr(content, "root", None)
        if isinstance(root, str):
            return root

    text_parts: list[str] = []
    for part in request_content_parts(content):
        if isinstance(text := getattr(part, "text", None), str):
            text_parts.append(text)
        elif isinstance(refusal := getattr(part, "refusal", None), str):
            text_parts.append(refusal)
    return "".join(text_parts)


__all__ = ["RequestContentLike", "dump_json_model", "request_content_parts", "request_content_to_text"]
