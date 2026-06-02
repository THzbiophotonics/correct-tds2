"""Optax-based optimizer for the THz correction loop.

Wraps optax Adam with cosine/exp/piecewise LR schedules and a fori_loop training block.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit

from .jax_ops import batch_gradients

if TYPE_CHECKING:
    import optax

__all__ = [
    "make_optax_optimizer",
    "optax_step",
    "optax_train_block",
    "infer_initial_lr",
]


def make_optax_optimizer(
    init_lr: float,
    decay_steps: int,
    schedule_type: str = "cosine",
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Adam + gradient clipping with a choice of LR schedule.

    schedule_type: "cosine" (default), "exp", or "piecewise".
    Returns (tx, schedule) where tx is the GradientTransformation.
    """
    try:
        import optax
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to import optax — JAX/optax version conflict. "
            "Fix: pip install 'optax>=0.2.8'."
        ) from exc

    schedule_key = (schedule_type or "cosine").lower()
    if schedule_key == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=init_lr, decay_steps=decay_steps, alpha=0.0
        )
    elif schedule_key == "exp":
        schedule = optax.exponential_decay(
            init_value=init_lr,
            transition_steps=max(1, decay_steps // 4),
            decay_rate=0.5,
        )
    elif schedule_key == "piecewise":
        schedule = optax.piecewise_constant_schedule(
            init_value=init_lr,
            boundaries_and_scales={
                decay_steps // 3: 0.3,
                2 * decay_steps // 3: 0.1,
            },
        )
    else:
        raise ValueError("schedule_type must be 'cosine', 'exp', or 'piecewise'")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-2),  # AdamW: anchors logit-space params near 0 (= zero correction)
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    return tx, schedule


@partial(jit, static_argnums=(0,))
def optax_step(
    tx: optax.GradientTransformation,
    params: jax.Array,
    opt_state: optax.OptState,
    grads: jax.Array,
) -> tuple[jax.Array, optax.OptState]:
    """Apply one gradient update and return updated ``(params, opt_state)``."""
    try:
        import optax
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to import optax — JAX/optax version conflict. "
            "Fix: pip install 'optax>=0.2.8'."
        ) from exc

    updates, new_opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state


@partial(jit, static_argnums=(0, 9))
def optax_train_block(
    tx: optax.GradientTransformation,
    params: jax.Array,
    opt_state: optax.OptState,
    pulses: jax.Array,
    reference: jax.Array,
    time_axis: jax.Array,
    angular_freqs: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
    steps: int,
) -> tuple[jax.Array, optax.OptState]:
    """Run all gradient updates on-device via fori_loop — no Python overhead per step."""

    @partial(jit, static_argnums=(0,))
    def _train_step(tx, params, opt_state):
        grads = batch_gradients(
            params, pulses, reference, time_axis, angular_freqs, lower_bounds, upper_bounds
        )
        return optax_step(tx, params, opt_state, grads)

    def _body(_, carry):
        curr_params, curr_state = carry
        return _train_step(tx, curr_params, curr_state)

    return jax.lax.fori_loop(0, steps, _body, (params, opt_state))


def infer_initial_lr(
    parameter_matrix: jax.Array,
    pulses: jax.Array,
    reference: jax.Array,
    time_axis: jax.Array,
    angular_freqs: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
    target_step: float = 1e-2,
    min_lr: float = 5e-3,
) -> float:
    """Guess a good initial LR so the first update moves roughly target_step in param space."""
    grads = batch_gradients(
        parameter_matrix, pulses, reference, time_axis, angular_freqs, lower_bounds, upper_bounds
    )
    grad_norm = float(jnp.linalg.norm(grads))
    if grad_norm < 1e-8:
        return min_lr
    return max(float(target_step / grad_norm), min_lr)
