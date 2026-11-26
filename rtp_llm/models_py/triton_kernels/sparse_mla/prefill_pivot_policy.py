"""Resolve immutable prefill choices before constructing model layers."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate PIVOT configuration key: {key!r}")
        result[key] = value
    return result


def _setting(value: Any) -> str:
    if not isinstance(value, str) or value not in ("native", "G4P4K"):
        raise ValueError("PIVOT layer setting must be 'native' or 'G4P4K'")
    return value


def resolve_prefill_pivot_layers(config: Any) -> Optional[tuple[Optional[str], ...]]:
    """Read once per model; None preserves the legacy off/refine environment.

    Entries use physical model layer IDs, not compact indexer/KV-slot IDs.
    Shared layers have no Indexer and get None. MTP layers stay native and do
    not interpret the target model's layer IDs as their local layer IDs.
    """
    num_layers = int(config.num_layers)
    # The 78-layer GLM-5.3 target always uses the layerwise prefill policy.
    glm53_target = (
        getattr(config, "_is_glm52_architecture", False)
        and num_layers == 78
        and not getattr(config, "is_mtp", False)
    )
    if not glm53_target:
        mode = os.environ.get("GLM5_PREFILL_PIVOT_MODE", "off").strip().lower()
        if mode != "layerwise":
            return None
    if bool(getattr(config, "is_mtp", False)):
        return ("native",) * num_layers

    attn = config.attn_config
    if not (attn.use_mla and attn.is_sparse) or (
        attn.indexer_head_num,
        attn.indexer_head_dim,
        attn.indexer_topk,
    ) != (32, 128, 2048):
        raise ValueError(
            "Layerwise PIVOT requires MLA with 32 indexer heads, dim 128, Top-2048"
        )
    full_layers = {i for i in range(num_layers) if dsa_layer_has_indexer(config, i)}
    if not full_layers:
        raise ValueError("Layerwise PIVOT requires at least one indexer layer")

    path = os.environ.get("GLM5_PREFILL_PIVOT_LAYER_CONFIG", "").strip()
    if glm53_target:
        native_layers = ("34", "38", "62")
    else:
        raw_layers = os.environ.get("GLM5_PREFILL_PIVOT_NATIVE_LAYERS", "").strip()
        native_layers = (
            tuple(key.strip() for key in raw_layers.split(",")) if raw_layers else ()
        )
    if path:
        try:
            raw = Path(path).read_bytes()
        except OSError as error:
            raise ValueError(
                f"Cannot read PIVOT layer configuration {path!r}: {error}"
            ) from error
        data = json.loads(raw, object_pairs_hook=_unique_keys)
        source = f"{path} sha256={hashlib.sha256(raw).hexdigest()}"
    elif native_layers:
        data = {"default": "G4P4K"}
        source = "GLM-5.3 built-in policy" if glm53_target else "environment"
    else:
        raise ValueError(
            "GLM5_PREFILL_PIVOT_LAYER_CONFIG or GLM5_PREFILL_PIVOT_NATIVE_LAYERS "
            "is required in layerwise mode"
        )
    if not isinstance(data, dict) or set(data) - {"default", "layers"}:
        raise ValueError(
            "PIVOT configuration must be an object with only 'default' and 'layers'"
        )
    default = _setting(data.get("default", "native"))
    overrides = data.get("layers", {})
    if not isinstance(overrides, dict):
        raise ValueError("PIVOT 'layers' must map model layer IDs to settings")
    choices = list(overrides.items())
    if native_layers:
        native_ids = _unique_keys((key, "native") for key in native_layers)
        choices.extend(native_ids.items())
    settings = [default if i in full_layers else None for i in range(num_layers)]
    for key, value in choices:
        if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(
                f"PIVOT layer ID must be a canonical nonnegative integer: {key!r}"
            )
        layer = int(key)
        if layer not in full_layers:
            raise ValueError(
                f"PIVOT layer {layer} is not an indexer layer; valid IDs: {sorted(full_layers)}"
            )
        settings[layer] = _setting(value)
    logging.info(
        "Layerwise prefill PIVOT config=%s settings=%s "
        "(G4P4K: group=4, candidates=4096, window=4, min_kv=32768)",
        source,
        {i: settings[i] for i in sorted(full_layers)},
    )
    return tuple(settings)
