"""
eigen_classifier.py
--------------------
Core mathematics for 2x2 linear ODE system analysis.

Given a system  x' = Ax  where A is a 2x2 real matrix, this module
computes eigenvalues, eigenvectors, and classifies the fixed point at
the origin using the trace-determinant plane.

Classification rules (λ₁, λ₂ are eigenvalues):
  - Stable Node      : both real, both negative
  - Unstable Node    : both real, both positive
  - Saddle Point     : both real, opposite signs
  - Stable Spiral    : complex conjugate pair, negative real part
  - Unstable Spiral  : complex conjugate pair, positive real part
  - Centre           : purely imaginary (real part = 0)
  - Degenerate cases : repeated eigenvalues handled separately
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EigenResult:
    """Container for eigenanalysis output."""
    matrix: np.ndarray
    eigenvalues: np.ndarray           # complex array of length 2
    eigenvectors: np.ndarray          # columns are eigenvectors
    trace: float
    determinant: float
    discriminant: float               # trace² - 4*det
    classification: str
    is_stable: Optional[bool]         # None for centres (neutrally stable)
    description: str


# ─── Tolerance for floating-point comparisons ────────────────────────────────
_EPS = 1e-10


def analyse(matrix: np.ndarray) -> EigenResult:
    """
    Analyse a 2x2 real matrix and return a full EigenResult.

    Parameters
    ----------
    matrix : np.ndarray, shape (2, 2)

    Returns
    -------
    EigenResult
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 matrix, got shape {matrix.shape}")

    A = matrix.astype(float)

    # Trace and determinant via closed form (more numerically transparent)
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = tr ** 2 - 4 * det

    # Numpy eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)

    classification, is_stable, description = _classify(tr, det, disc, eigenvalues)

    return EigenResult(
        matrix=A,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        trace=tr,
        determinant=det,
        discriminant=disc,
        classification=classification,
        is_stable=is_stable,
        description=description,
    )


def _classify(tr, det, disc, eigenvalues) -> tuple[str, Optional[bool], str]:
    """Return (classification_name, is_stable, plain_english_description)."""

    lam1, lam2 = eigenvalues

    # ── Determinant < 0  →  eigenvalues real, opposite signs ────────────────
    if det < -_EPS:
        return (
            "Saddle Point",
            None,
            "The eigenvalues are real with opposite signs. Trajectories are "
            "attracted along one eigendirection and repelled along the other — "
            "the origin is an unstable saddle point."
        )

    # ── det = 0  →  at least one zero eigenvalue (non-isolated fixed points) ─
    if abs(det) <= _EPS:
        return (
            "Non-Isolated Fixed Points",
            None,
            "The determinant is zero, meaning there is a line of fixed points. "
            "The origin is not an isolated equilibrium."
        )

    # ── det > 0 from here ────────────────────────────────────────────────────

    # Complex eigenvalues (discriminant < 0)
    if disc < -_EPS:
        real_part = tr / 2
        if abs(real_part) <= _EPS:
            return (
                "Centre",
                None,
                "The eigenvalues are purely imaginary. Trajectories form closed "
                "ellipses around the origin — neutrally stable."
            )
        elif real_part < 0:
            return (
                "Stable Spiral",
                True,
                "The eigenvalues are complex with negative real part. Trajectories "
                "spiral inward toward the origin — asymptotically stable."
            )
        else:
            return (
                "Unstable Spiral",
                False,
                "The eigenvalues are complex with positive real part. Trajectories "
                "spiral outward away from the origin — unstable."
            )

    # Real repeated eigenvalue (discriminant ≈ 0)
    if abs(disc) <= _EPS:
        if abs(tr) <= _EPS:
            return (
                "Centre (Repeated Zero)",
                None,
                "Both eigenvalues are zero — highly degenerate case."
            )
        elif tr < 0:
            return (
                "Stable Star / Degenerate Node",
                True,
                "Repeated negative eigenvalue. Trajectories approach the origin "
                "along every direction (star node) or asymptotically along one "
                "direction (degenerate node) — stable."
            )
        else:
            return (
                "Unstable Star / Degenerate Node",
                False,
                "Repeated positive eigenvalue. Trajectories recede from the origin — unstable."
            )

    # Real distinct eigenvalues, both same sign (disc > 0, det > 0)
    r1, r2 = np.real(lam1), np.real(lam2)

    if r1 < -_EPS and r2 < -_EPS:
        return (
            "Stable Node",
            True,
            "Both eigenvalues are real and negative. All trajectories flow into "
            "the origin — asymptotically stable."
        )
    elif r1 > _EPS and r2 > _EPS:
        return (
            "Unstable Node",
            False,
            "Both eigenvalues are real and positive. All trajectories flow away "
            "from the origin — unstable."
        )
    else:
        # Shouldn't reach here given det > 0, but handle gracefully
        return (
            "Node (Mixed Signs)",
            None,
            "Mixed real eigenvalue signs — see values above."
        )


def format_report(result: EigenResult) -> str:
    """Return a formatted plain-text report of the analysis."""

    lam1, lam2 = result.eigenvalues
    v1 = result.eigenvectors[:, 0]
    v2 = result.eigenvectors[:, 1]

    def fmt_complex(z):
        if abs(z.imag) < _EPS:
            return f"{z.real:+.4f}"
        return f"{z.real:+.4f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.4f}i"

    def fmt_vec(v):
        return f"[{v[0].real:+.4f},  {v[1].real:+.4f}]"

    stability = (
        "Stable" if result.is_stable is True
        else "Unstable" if result.is_stable is False
        else "Neutrally Stable / Non-isolated"
    )

    lines = [
        "=" * 52,
        " EIGENVALUE PHASE PORTRAIT ANALYSIS",
        "=" * 52,
        f"  Matrix A:",
        f"    [ {result.matrix[0,0]:7.3f}  {result.matrix[0,1]:7.3f} ]",
        f"    [ {result.matrix[1,0]:7.3f}  {result.matrix[1,1]:7.3f} ]",
        "",
        f"  Trace       : {result.trace:.4f}",
        f"  Determinant : {result.determinant:.4f}",
        f"  Discriminant: {result.discriminant:.4f}",
        "",
        f"  λ₁ = {fmt_complex(lam1)}    eigenvector: {fmt_vec(v1)}",
        f"  λ₂ = {fmt_complex(lam2)}    eigenvector: {fmt_vec(v2)}",
        "",
        f"  ┌─ Classification: {result.classification}",
        f"  └─ Stability     : {stability}",
        "",
        "  " + result.description,
        "=" * 52,
    ]
    return "\n".join(lines)