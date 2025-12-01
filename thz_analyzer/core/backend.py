
"""Backend facade for CPU/GPU compute.
Selects JAX on GPU (if available) or CPU otherwise; falls back to NumPy interface where needed.
"""
import numpy as _np
try:
    import jax
    import jax.numpy as _jnp
    from jax import jit as _jit, vmap as _vmap
except Exception:  # JAX not installed for some environments
    jax = None
    _jnp = None
    def _jit(f): return f
    def _vmap(f, in_axes=0, out_axes=0): return f

def has_gpu():
    if jax is None: 
        return False
    return any(d.platform == "gpu" for d in jax.devices())

def preferred_device(prefer_gpu: bool = True):
    if jax is None:
        return None
    if prefer_gpu:
        g = [d for d in jax.devices() if d.platform == "gpu"]
        if g:
            return g[0]
    return jax.devices("cpu")[0]

def resolve_device(preference: str = "cpu"):
    """
    Return a JAX device matching the requested preference.
    Also returns a boolean indicating whether the preference was satisfied.
    """
    if jax is None:
        raise RuntimeError("JAX is not installed: cannot select a device.")
    requested = (preference or "cpu").lower()
    matching = [d for d in jax.devices() if d.platform == requested]
    if matching:
        return matching[0], True
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("No JAX CPU device available.")
    return cpu_devices[0], requested == "cpu"

# Expose array namespace (xp)
xp = _jnp if _jnp is not None else _np

# Basic ops
jit = _jit
vmap = _vmap

def to_device(x, device=None):
    if jax is None:
        return _np.asarray(x)
    arr = _jnp.asarray(x)
    return jax.device_put(arr, device=device) if device is not None else arr

def rfft_omega(time_axis):
    dt = float(time_axis[1] - time_axis[0])
    freqs = xp.fft.rfftfreq(time_axis.shape[0], d=dt)
    return 2.0 * xp.pi * freqs
