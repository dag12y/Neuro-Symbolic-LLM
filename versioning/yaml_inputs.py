"""Strict YAML input for offline definition snapshots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode
from yaml.nodes import Node as YAMLNode

_YAML_TAGS = {
    f"tag:yaml.org,2002:{tag}"
    for tag in ("map", "seq", "str", "int", "float", "bool", "null")
}


class _ProjectionLoader(yaml.SafeLoader):
    """Use SafeLoader scalars while rejecting ambiguous structure."""

    def compose_node(self, parent: YAMLNode | None, index: Any) -> YAMLNode:
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ValueError("YAML anchors and aliases are unsupported")
        node = super().compose_node(parent, index)
        if node is None:
            raise ValueError("Expected a YAML node")
        if node.tag not in _YAML_TAGS:
            raise ValueError(f"Unsupported YAML tag: {node.tag}")
        return node

    def construct_mapping(self, node: YAMLNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ValueError("Expected a YAML mapping node")
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in keys:
                raise ValueError(f"Duplicate YAML key: {key!r}")
            keys.add(key)
        return cast(dict[Any, Any], super().construct_mapping(node, deep=deep))


def mapping_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def read_yaml(
    path: Path, read_bytes: Callable[[Path], bytes]
) -> tuple[str, dict[str, Any]]:
    # Preserve the runtime loaders' UTF-8 text-mode newline handling.
    text = read_bytes(path).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    try:
        value = yaml.load(text, Loader=_ProjectionLoader)
        return text, mapping_value(value, path.name)
    except (yaml.YAMLError, ValueError, RecursionError) as error:
        raise ValueError(f"{path.name}: {error}") from error
