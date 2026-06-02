import math
from contextlib import nullcontext
from typing import Any

import numpy as np

__all__ = ["periodic_sampling_correct_tds"]


def _wrap_phase(phi: float) -> float:
    """Wrap a phase into [-pi, pi]."""
    two_pi = 2.0 * math.pi
    return ((phi + math.pi) % two_pi) - math.pi


def _fft_error(signal: np.ndarray, index_limit: int) -> float:
    """Measure the high-frequency tail."""
    spectrum = np.fft.rfft(signal)
    if index_limit <= 0:
        tail = spectrum
    elif index_limit >= spectrum.shape[0]:
        return 0.0
    else:
        tail = spectrum[index_limit:]
    if tail.size == 0:
        return 0.0
    return float(np.sum(np.abs(tail)))


def _safe_logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply a stable logit transform."""
    clipped = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def _periodic_cpu_search(
    mean_signal: np.ndarray,
    time_axis: np.ndarray,
    gradient: np.ndarray,
    index_limit: int,
    minval_ps: np.ndarray,
    maxval_ps: np.ndarray,
    x0_norm: np.ndarray,
    random_trials: int,
    coordinate_rounds: int,
) -> tuple[np.ndarray, float, str]:
    """Run the CPU search."""
    span_ps = maxval_ps - minval_ps

    def evaluate(x_norm: np.ndarray) -> float:
        x_norm = np.clip(np.asarray(x_norm, dtype=float), 0.0, 1.0)
        x = x_norm * span_ps + minval_ps
        ct = x[0] * np.cos(x[1] * time_axis + x[2])
        corrected = mean_signal - gradient * ct
        return _fft_error(corrected, index_limit)

    x_best = np.clip(np.asarray(x0_norm, dtype=float), 0.0, 1.0)
    best_error = float(evaluate(x_best))
    optimizer_name = "cpu_fallback_random"
    maxiter = max(20, int(coordinate_rounds) * 3)

    try:
        from scipy import optimize

        res = optimize.dual_annealing(
            evaluate,
            x0=x_best,
            maxiter=maxiter,
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )
        x_best = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
        best_error = float(res.fun)
        optimizer_name = "cpu_dual_annealing"
        return x_best, best_error, optimizer_name
    except Exception:
        pass

    rng = np.random.default_rng(0)
    for _ in range(max(16, int(random_trials))):
        candidate = rng.uniform(0.0, 1.0, size=3)
        curr_err = float(evaluate(candidate))
        if curr_err < best_error:
            x_best = candidate
            best_error = curr_err

    step = np.array([0.2, 0.2, 0.2], dtype=float)
    for _ in range(max(1, int(coordinate_rounds))):
        improved = False
        for axis in range(3):
            for direction in (-1.0, 1.0):
                trial = x_best.copy()
                trial[axis] = np.clip(trial[axis] + direction * step[axis], 0.0, 1.0)
                curr_err = float(evaluate(trial))
                if curr_err + 1e-18 < best_error:
                    x_best = trial
                    best_error = curr_err
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if np.all(step <= 1e-3):
                break

    return x_best, best_error, optimizer_name


def _periodic_jax_optax_search(
    mean_signal: np.ndarray,
    time_axis: np.ndarray,
    gradient: np.ndarray,
    index_limit: int,
    minval_ps: np.ndarray,
    maxval_ps: np.ndarray,
    x0_norm: np.ndarray,
    mode: str,
    device: Any | None = None,
) -> tuple[np.ndarray, float, str] | None:
    """Run the JAX multi-start search."""
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except Exception:
        return None

    mode_key = str(mode).strip().lower()
    if mode_key == "strict":
        n_starts = 64
        n_steps = 260
        learning_rate = 4e-2
    else:
        n_starts = 20
        n_steps = 100
        learning_rate = 6e-2

    mean_np = np.asarray(mean_signal, dtype=np.float32)
    time_np = np.asarray(time_axis, dtype=np.float32)
    grad_np = np.asarray(gradient, dtype=np.float32)
    min_np = np.asarray(minval_ps, dtype=np.float32)
    span_np = np.asarray(maxval_ps - minval_ps, dtype=np.float32)

    n_rfft = mean_np.shape[0] // 2 + 1
    tail_mask_np = np.zeros(n_rfft, dtype=np.float32)
    tail_mask_np[int(np.clip(index_limit, 0, n_rfft)) :] = 1.0

    starts_norm = np.random.default_rng(0).uniform(0.0, 1.0, size=(n_starts, 3)).astype(np.float32)
    starts_norm[0] = np.clip(np.asarray(x0_norm, dtype=np.float32), 0.0, 1.0)
    if n_starts > 1:
        starts_norm[1] = np.array([0.5, starts_norm[0, 1], 0.5], dtype=np.float32)
    raw_init = _safe_logit(starts_norm)

    try:
        mean_j = jnp.asarray(mean_np)
        time_j = jnp.asarray(time_np)
        grad_j = jnp.asarray(grad_np)
        min_j = jnp.asarray(min_np)
        span_j = jnp.asarray(span_np)
        tail_mask_j = jnp.asarray(tail_mask_np)

        def loss_single(raw: "jax.Array") -> "jax.Array":
            x_norm = jax.nn.sigmoid(raw)
            x = min_j + span_j * x_norm
            ct = x[0] * jnp.cos(x[1] * time_j + x[2])
            corrected = mean_j - grad_j * ct
            tail = jnp.fft.rfft(corrected)
            return jnp.sum(jnp.abs(tail) * tail_mask_j)

        # value_and_grad avoids a redundant forward pass: loss is free once we have grad.
        batch_val_grad = jax.jit(jax.vmap(jax.value_and_grad(loss_single)))

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale(-learning_rate),
        )

        params = jnp.asarray(raw_init, dtype=jnp.float32)
        opt_state = tx.init(params)

        @jax.jit
        def step(curr_params, curr_opt_state):
            losses_step, grads = batch_val_grad(curr_params)
            updates, next_opt_state = tx.update(grads, curr_opt_state, curr_params)
            next_params = optax.apply_updates(curr_params, updates)
            return next_params, next_opt_state, losses_step

        ctx = jax.default_device(device) if device is not None else nullcontext()
        with ctx:
            losses = None
            for _ in range(n_steps):
                params, opt_state, losses = step(params, opt_state)
        # losses holds the per-start losses from the last step

        params_np = np.asarray(params)
        losses_np = np.asarray(losses)
        best_idx = int(np.argmin(losses_np))
        best_raw = params_np[best_idx]
        x_best = 1.0 / (1.0 + np.exp(-best_raw))
        return np.clip(x_best, 0.0, 1.0), float(losses_np[best_idx]), f"jax_optax_{mode_key}"
    except Exception:
        return None


def periodic_sampling_correct_tds(
    mean_signal: np.ndarray,
    time_axis: np.ndarray,
    freq_limit_thz: float,
    random_trials: int = 128,
    coordinate_rounds: int = 40,
    mode: str = "cpu",
    device: Any | None = None,
) -> dict[str, object]:
    """Estimate and apply the periodic sampling correction."""

    mean_signal = np.asarray(mean_signal, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    if mean_signal.ndim != 1 or time_axis.ndim != 1:
        raise ValueError("mean_signal and time_axis must be 1-D arrays.")
    if mean_signal.shape != time_axis.shape:
        raise ValueError("mean_signal and time_axis must share the same length.")
    if mean_signal.size < 4:
        raise ValueError("Need at least 4 samples for periodic correction.")

    dt = float(time_axis[1] - time_axis[0])
    if dt == 0.0:
        raise ValueError("Time axis has zero spacing; cannot compute gradient.")

    gradient = np.gradient(mean_signal, dt)
    freqs = np.fft.rfftfreq(mean_signal.size, d=dt)
    freq_limit_hz = max(float(freq_limit_thz) * 1e12, 0.0)
    if freqs.size > 1:
        delta_nu = freqs[-1] / (len(freqs) - 1)
        index_limit = int(freq_limit_hz / max(delta_nu, 1e-30))
    else:
        index_limit = 0
    index_limit = int(np.clip(index_limit, 0, freqs.size))

    # Keep the legacy Correct-TDS parameter bounds.
    maxval_ps = np.array([abs(dt) / 10.0, 12.0 * 2.0 * math.pi * 1e12, math.pi], dtype=float)
    minval_ps = np.array([0.0, 6.0 * 2.0 * math.pi * 1e12, -math.pi], dtype=float)
    span_ps = maxval_ps - minval_ps

    guess_ps = np.array([0.0, 9.0 * 2.0 * math.pi * 1e12, 0.0], dtype=float)
    if index_limit < freqs.size and freqs.size > 0:
        fft_mean = np.fft.rfft(mean_signal)
        tail = np.abs(fft_mean[index_limit:]) if index_limit < fft_mean.size else np.array([])
        if tail.size > 0 and np.any(np.isfinite(tail)):
            dominant = index_limit + int(np.argmax(tail))
            dom_omega = 2.0 * math.pi * freqs[dominant]
            guess_ps[1] = float(np.clip(dom_omega, minval_ps[1], maxval_ps[1]))
    x0_norm = (guess_ps - minval_ps) / np.maximum(span_ps, 1e-30)
    x0_norm = np.clip(x0_norm, 0.0, 1.0)

    mode_key = str(mode or "cpu").strip().lower()
    if mode_key not in ("cpu", "fast", "strict"):
        mode_key = "cpu"

    solution = None
    if mode_key in ("fast", "strict"):
        solution = _periodic_jax_optax_search(
            mean_signal,
            time_axis,
            gradient,
            index_limit,
            minval_ps,
            maxval_ps,
            x0_norm,
            mode=mode_key,
            device=device,
        )

    if solution is None:
        x_best, best_error, optimizer_name = _periodic_cpu_search(
            mean_signal,
            time_axis,
            gradient,
            index_limit,
            minval_ps,
            maxval_ps,
            x0_norm,
            random_trials=random_trials,
            coordinate_rounds=coordinate_rounds,
        )
        if mode_key in ("fast", "strict"):
            optimizer_name = f"{mode_key}_fallback_{optimizer_name}"
    else:
        x_best, best_error, optimizer_name = solution

    params = x_best * span_ps + minval_ps
    amp_best = float(np.clip(params[0], 0.0, maxval_ps[0]))
    omega_best = float(np.clip(params[1], minval_ps[1], maxval_ps[1]))
    phi_best = _wrap_phase(float(params[2]))

    ct = amp_best * np.cos(omega_best * time_axis + phi_best)
    correction = gradient * ct
    corrected_signal = mean_signal - correction
    residual_error = _fft_error(corrected_signal, index_limit)

    return {
        "corrected_signal": corrected_signal,
        "correction_waveform": correction,
        "ct": ct,
        "params": {
            "A": amp_best,
            "omega": omega_best,
            "phi": phi_best,
        },
        "error": residual_error,
        "freq_limit_hz": freq_limit_hz,
        "index_freq_limit": index_limit,
        "mode": mode_key,
        "optimizer": optimizer_name,
        "optimizer_error": best_error,
    }
