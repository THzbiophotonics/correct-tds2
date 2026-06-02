import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["validate_2d_array", "validate_square_matrix", "ensure_numpy", "ensure_jax"]


def validate_2d_array(arr: ArrayLike, name: str = "array") -> NDArray[np.float64]:
    """Return a finite 2D float64 array."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def validate_square_matrix(arr: ArrayLike, name: str = "matrix") -> NDArray[np.float64]:
    """Return a finite square float64 matrix."""
    arr = validate_2d_array(arr, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got {arr.shape}")
    return arr


def ensure_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert to NumPy."""
    return np.asarray(arr)


def ensure_jax(arr: ArrayLike) -> "jax.Array":
    """Convert to JAX."""
    import jax
    import jax.numpy as jnp

    return jnp.asarray(arr)
