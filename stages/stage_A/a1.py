"""Stage A1: plain residual baseline (no predictive coding, no symbolic head).

L_A1 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2
"""

from __future__ import annotations

import jax

from frozenllm.substrate import FrozenSubstrate
from residual import AdapterParams, ResidualAdapter
from stages.config import A1Config as A1Config
from stages.stage_base import base_logits


def a1_objective(
    params: AdapterParams,
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A1Config,
    input_ids: jax.Array,
    labels: jax.Array | None = None,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    adapted = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(params),
        intercept_layers=adapter.layers,
    ).logits
    target_labels = input_ids if labels is None else labels
    wd = ResidualAdapter.l2_norm_sq(params)

    # Skip the base forward; omit "kl" rather than NaN, which strict JSON rejects.
    if config.lambda_kl == 0 and base is None:
        task = substrate.compute_loss(adapted, target_labels, losses=("CE",))
        total = task + config.lambda_wd * wd
        return total, {"total": total, "task": task, "wd": wd}

    base = (
        base_logits(substrate, input_ids)
        if base is None
        else jax.lax.stop_gradient(base)
    )
    total_model_loss, components = substrate.compute_loss(
        adapted,
        target_labels,
        losses=("CE", "KL"),
        base_logits=base,
        weights={"KL": config.lambda_kl},
        return_components=True,
    )
    task = components["ce"]
    kl = components["kl"]
    total = total_model_loss + config.lambda_wd * wd
    return total, {"total": total, "task": task, "kl": kl, "wd": wd}


__all__ = ["A1Config", "a1_objective"]
