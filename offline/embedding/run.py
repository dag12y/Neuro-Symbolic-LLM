"""Command-line entry point for offline template embedding generation."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from .context_builder import load_pattern_contexts
from .encoder import FrozenLLMEncoder
from .template_embeddings import (
    DEFAULT_PROJECTION_SEED,
    DEFAULT_SYMBOLIC_DIM,
    initialize_template_kv,
    save_template_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate source-grounded template K/V embeddings."
    )

    parser.add_argument("--documents", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument(
        "--symbolic-dim",
        type=int,
        default=DEFAULT_SYMBOLIC_DIM,
    )
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=DEFAULT_PROJECTION_SEED,
    )

    args = parser.parse_args()

    generate_embeddings(**vars(args))


def generate_embeddings(
    documents: str | Path,
    patterns: str | Path,
    output: str | Path,
    *,
    model: str,
    max_length: int,
    layer_idx: int | None,
    symbolic_dim: int,
    projection_seed: int,
    observe_encoder: Callable[[FrozenLLMEncoder], None] | None = None,
) -> None:
    """Run the CLI's context selection, frozen encoding and K/V writer."""
    templates = load_pattern_contexts(
        pattern_records_path=patterns,
        documents_path=documents,
    )

    if not templates:
        raise ValueError("No pattern records found to embed")

    encoder = FrozenLLMEncoder(
        model_name=model,
        max_length=max_length,
        layer_idx=layer_idx,
    )

    if observe_encoder is not None:
        observe_encoder(encoder)
    records = []

    for template in templates:
        record = initialize_template_kv(
            template,
            encoder,
            symbolic_dim=symbolic_dim,
            projection_seed=projection_seed,
        )

        records.append(record)

        print(
            f"Embedded {template.pattern_id}: "
            f"{record['num_contexts']} contexts, "
            f"{record['embedding_dim']} dimensions"
        )

    save_template_embeddings(records, output)

    print(f"Saved {len(records)} template embeddings to {output}")


if __name__ == "__main__":
    main()
