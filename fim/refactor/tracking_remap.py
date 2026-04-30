"""Post-processing helpers for tracked displacement fields.

Includes:

- Isotropic / diffusion-style smoothing (:func:`smooth_displacement_field`).
- Coarse-grid reference axes that align with :func:`scipy.ndimage.zoom`
  upsampling (:func:`coarse_reference_axes_m`).
- Lagrangian remapping that inverts :math:`x = X + u(x)` via fixed-point
  iteration (:func:`remap_displacement_lagrangian_griddata`).

All routines are pure NumPy/SciPy and independent of :mod:`torch`.
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.ndimage


def smooth_displacement_field(
    U: np.ndarray,
    method: str,
    sigma: float,
) -> np.ndarray:
    """Apply post-processing smoothing to a 3D displacement component.

    Parameters
    ----------
    U : ndarray, shape (nx, ny, nz)
        One component of the displacement field (e.g. Ux).
    method : str
        ``"gaussian"``  — Gaussian low-pass filter (isotropic, ``sigma`` in pixels).
            Directly smooths the field. Good for general noise reduction.
        ``"laplacian"`` — Iterative Laplacian diffusion smoothing.
            Solves ``U_new = U + sigma * laplacian(U)`` for one step.
            ``sigma`` controls the diffusion strength (typical: 0.1–1.0).
            Smooths while respecting the local structure of the field;
            commonly used for mesh/displacement field regularization.
    sigma : float
        Kernel size / diffusion strength (see *method*).

    Returns
    -------
    ndarray, same shape as *U*.
    """
    if method == "gaussian":
        return scipy.ndimage.gaussian_filter(
            U, sigma=sigma, output=np.empty_like(U)
        )  # Blur each component with a Gaussian bell of width sigma (voxels).
    if method == "laplacian":
        # Smooth by repeatedly adding a small Laplacian step (like heat diffusion on the field).
        # More iterations when sigma is large; step size alpha stays small so it stays stable.
        n_iter = max(1, int(round(sigma)))  # How many diffusion steps to take.
        alpha = np.float32(min(sigma / max(n_iter, 1), 1.0 / 6.0))  # Step size per iteration (capped for 3D stability).
        result = U.copy()  # Working copy we update in place.
        lap = np.empty_like(result)  # Scratch array holding the Laplacian of result.
        for _ in range(n_iter):
            scipy.ndimage.laplace(result, output=lap)
            result += alpha * lap
        return result
    raise ValueError(f"Unknown smoothing method: {method}")


def coarse_reference_axes_m(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    nx_c: int,
    ny_c: int,
    nz_c: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical coordinates (m) at coarse deformation nodes, aligned with ``ndimage.zoom`` layout.

    Coarse index ``ic`` maps to a fractional index along the full ``x_axis_m`` so that
    upsampling coarse data with ``zoom=(nx/nx_c, ...)`` targets the same geometry as the
    fine reference voxel centers.
    """
    nx_f, ny_f, nz_f = len(x_axis_m), len(y_axis_m), len(z_axis_m)

    def centers_1d(ax: np.ndarray, n_c: int, n_full: int) -> np.ndarray:
        if n_c < 1 or n_full < 1:
            raise ValueError("n_c and n_full must be positive")
        ax = np.asarray(ax, dtype=np.float64)
        if n_c == 1:
            idx = np.array([(n_full - 1) / 2.0], dtype=np.float64)
        else:
            idx = (np.arange(n_c, dtype=np.float64) + 0.5) * (n_full / n_c) - 0.5
        idx = np.clip(idx, 0.0, float(n_full - 1))
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_full - 1)
        w = idx - i0.astype(np.float64)
        return (1.0 - w) * ax[i0] + w * ax[i1]

    return (
        centers_1d(x_axis_m, nx_c, nx_f),
        centers_1d(y_axis_m, ny_c, ny_f),
        centers_1d(z_axis_m, nz_c, nz_f),
    )


def _sample_displacement_at_world_m(
    Ux: np.ndarray,
    Uy: np.ndarray,
    Uz: np.ndarray,
    xm: np.ndarray,
    ym: np.ndarray,
    zm: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trilinear (order=1) or nearest (order=0) sample of (Ux,Uy,Uz) at world coords (m)."""
    fx = (xm - x0) / dx
    fy = (ym - y0) / dy
    fz = (zm - z0) / dz
    coords = np.stack([fx, fy, fz], axis=0)
    kw = {"order": order, "mode": "nearest", "prefilter": False}
    ux_s = scipy.ndimage.map_coordinates(Ux, coords, **kw)
    uy_s = scipy.ndimage.map_coordinates(Uy, coords, **kw)
    uz_s = scipy.ndimage.map_coordinates(Uz, coords, **kw)
    return ux_s, uy_s, uz_s


def remap_displacement_lagrangian_griddata(
    Ux_m: np.ndarray,
    Uy_m: np.ndarray,
    Uz_m: np.ndarray,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    *,
    method: str = "linear",
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate displacement onto a regular grid in reference (Lagrangian) coordinates.

    Eulerian storage: ``U*[i,j,k]`` is u at the imaging voxel indexed by ``(i,j,k)``.
    For each reference lattice node ``X`` (same ``meshgrid`` as ``X.npy``/``Y.npy``/``Z.npy``),
    the deformed position obeys ``x = X + u(x)`` with u sampled on the Eulerian grid.
    We solve that with fixed-point iteration and :func:`scipy.ndimage.map_coordinates`
    (O(N) per iteration, modest memory). Equivalent to inverting the scatter ``X = x - u``
    without building a 3D Delaunay triangulation over all voxels.

    Parameters
    ----------
    Ux_m, Uy_m, Uz_m
        Displacement components (m), shape ``(nx, ny, nz)``, convention
        ``u = x_deformed - X_reference``.
    x_axis_m, y_axis_m, z_axis_m
        Uniform 1D voxel-center coordinates (m), length ``nx, ny, nz``.
    method
        ``linear`` → ``order=1``; ``nearest`` → ``order=0`` for ``map_coordinates``.
    max_iter
        Maximum fixed-point steps (typically converges in few iterations for moderate strain).

    Returns
    -------
    tuple
        ``(Ux, Uy, Uz)`` same shape, float64, u expressed on the reference grid.
    """
    if Ux_m.shape != Uy_m.shape or Ux_m.shape != Uz_m.shape:
        raise ValueError("Ux_m, Uy_m, Uz_m must have the same shape")
    nx, ny, nz = Ux_m.shape
    if (len(x_axis_m), len(y_axis_m), len(z_axis_m)) != (nx, ny, nz):
        raise ValueError("Axis lengths must match displacement shape")

    order = 1 if method == "linear" else 0
    x_axis_m = np.asarray(x_axis_m, dtype=np.float64)
    y_axis_m = np.asarray(y_axis_m, dtype=np.float64)
    z_axis_m = np.asarray(z_axis_m, dtype=np.float64)
    dx = float((x_axis_m[-1] - x_axis_m[0]) / (nx - 1)) if nx > 1 else 1.0
    dy = float((y_axis_m[-1] - y_axis_m[0]) / (ny - 1)) if ny > 1 else 1.0
    dz = float((z_axis_m[-1] - z_axis_m[0]) / (nz - 1)) if nz > 1 else 1.0
    x0, y0, z0 = float(x_axis_m[0]), float(y_axis_m[0]), float(z_axis_m[0])

    Ux = np.asarray(Ux_m, dtype=np.float64, order="C")
    Uy = np.asarray(Uy_m, dtype=np.float64, order="C")
    Uz = np.asarray(Uz_m, dtype=np.float64, order="C")

    Xg, Yg, Zg = np.meshgrid(x_axis_m, y_axis_m, z_axis_m, indexing="ij")
    xm = np.asarray(Xg, dtype=np.float64)
    ym = np.asarray(Yg, dtype=np.float64)
    zm = np.asarray(Zg, dtype=np.float64)

    ux, uy, uz = _sample_displacement_at_world_m(Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order)
    xm = Xg + ux
    ym = Yg + uy
    zm = Zg + uz

    tol = max(dx, dy, dz) * 1e-8
    if tol <= 0:
        tol = 1e-15

    last_delta = float("inf")
    for _ in range(max_iter):
        ux, uy, uz = _sample_displacement_at_world_m(Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order)
        xm_new = Xg + ux
        ym_new = Yg + uy
        zm_new = Zg + uz
        last_delta = float(np.max(np.abs(xm_new - xm) + np.abs(ym_new - ym) + np.abs(zm_new - zm)))
        xm, ym, zm = xm_new, ym_new, zm_new
        if last_delta < tol:
            break

    if last_delta >= tol:
        print(
            f"Warning: remap_to_reference fixed-point did not reach tol={tol:g} "
            f"(last max node update L1 ≈ {last_delta:g} m); "
            "try increasing --remap_max_iter.",
            file=sys.stderr,
            flush=True,
        )

    ux_out, uy_out, uz_out = _sample_displacement_at_world_m(
        Ux, Uy, Uz, xm, ym, zm, x0, y0, z0, dx, dy, dz, order=order
    )
    return ux_out, uy_out, uz_out
