"""
phase_portrait.py
-----------------
Generates a phase portrait for the 2x2 linear ODE system  x' = Ax.

Features:
  - Quiver plot (direction field) showing vector field across the plane
  - Streamlines for several representative initial conditions
  - Eigenvector lines overlaid (for real eigenvalues)
  - Colour-coded stability indicator in the title
  - Clean, publication-quality styling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from .eigen_classifier import EigenResult


# ─── Colour palette ──────────────────────────────────────────────────────────
STABLE_COLOUR    = "#4a9eff"   # blue  — stable
UNSTABLE_COLOUR  = "#ff6b6b"   # red   — unstable
NEUTRAL_COLOUR   = "#a8c4a2"   # green — centre / neutral
ARROW_COLOUR     = "#cccccc"
EV1_COLOUR       = "#f5c842"   # eigenvector 1
EV2_COLOUR       = "#ff9f43"   # eigenvector 2
BG_COLOUR        = "#0f1e3c"   # dark navy (matches portfolio)
GRID_COLOUR      = "#1e3a6e"


def _trajectory_colour(result: EigenResult) -> str:
    if result.is_stable is True:
        return STABLE_COLOUR
    elif result.is_stable is False:
        return UNSTABLE_COLOUR
    return NEUTRAL_COLOUR


def _choose_initial_conditions(classification: str, n: int = 12) -> np.ndarray:
    """
    Choose starting points for streamlines spread around the origin.
    Adjusts strategy slightly for saddle points to capture both manifolds.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 2.5

    if "Saddle" in classification:
        # Extra points near the axes to catch stable/unstable manifolds
        extra = np.array([
            [0.1,  2.0], [-0.1,  2.0],
            [0.1, -2.0], [-0.1, -2.0],
            [ 2.0,  0.1], [ 2.0, -0.1],
            [-2.0,  0.1], [-2.0, -0.1],
        ])
        ring = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        return np.vstack([ring, extra])

    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def plot_phase_portrait(
    result: EigenResult,
    xlim: tuple[float, float] = (-4, 4),
    ylim: tuple[float, float] = (-4, 4),
    grid_n: int = 22,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Render the phase portrait for the given EigenResult.

    Parameters
    ----------
    result     : EigenResult from eigen_classifier.analyse()
    xlim       : x-axis limits
    ylim       : y-axis limits
    grid_n     : number of grid points per axis for the quiver field
    save_path  : if provided, save figure to this path
    show       : if True, call plt.show()

    Returns
    -------
    matplotlib Figure
    """
    A = result.matrix
    traj_col = _trajectory_colour(result)

    # ── Set up figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOUR)
    ax.set_facecolor(BG_COLOUR)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ── Grid lines ────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOUR)
    ax.tick_params(colors="#8fa8c4", labelsize=9)
    ax.axhline(0, color=GRID_COLOUR, linewidth=0.8, zorder=1)
    ax.axvline(0, color=GRID_COLOUR, linewidth=0.8, zorder=1)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.4, alpha=0.6, zorder=0)

    # ── Quiver (direction field) ──────────────────────────────────────────────
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    UV = A @ np.array([X.ravel(), Y.ravel()])
    U, V = UV[0].reshape(X.shape), UV[1].reshape(Y.shape)
    speed = np.sqrt(U**2 + V**2)
    speed_norm = np.where(speed > 0, speed, 1)
    Un, Vn = U / speed_norm, V / speed_norm

    ax.quiver(
        X, Y, Un, Vn,
        speed,
        cmap="Blues",
        alpha=0.45,
        scale=28,
        width=0.003,
        headwidth=4,
        headlength=5,
        zorder=2,
    )

    # ── Streamlines ───────────────────────────────────────────────────────────
    ics = _choose_initial_conditions(result.classification)
    t_span = np.linspace(0, 6, 800)
    t_back = np.linspace(0, -6, 800)

    for ic in ics:
        for t_arr in (t_span, t_back):
            traj = _integrate_rk4(A, ic, t_arr, xlim, ylim)
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=traj_col, linewidth=1.0, alpha=0.7, zorder=3,
            )
            # Arrow mid-trajectory to show direction
            if len(traj) > 10:
                mid = len(traj) // 2
                dx = traj[mid, 0] - traj[mid - 1, 0]
                dy = traj[mid, 1] - traj[mid - 1, 1]
                ax.annotate(
                    "", xy=(traj[mid, 0], traj[mid, 1]),
                    xytext=(traj[mid, 0] - dx * 8, traj[mid, 1] - dy * 8),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=traj_col,
                        lw=1.0,
                    ),
                    zorder=4,
                )

    # ── Eigenvector lines (real eigenvalues only) ─────────────────────────────
    legend_handles = []
    lam1, lam2 = result.eigenvalues

    if abs(lam1.imag) < 1e-9:
        v1 = result.eigenvectors[:, 0].real
        _draw_eigenvector(ax, v1, EV1_COLOUR, xlim, ylim, label=f"EV₁ (λ={lam1.real:.3f})")
        legend_handles.append(Line2D([0], [0], color=EV1_COLOUR, lw=1.5,
                                     linestyle="--", label=f"EV₁  λ = {lam1.real:.3f}"))

    if abs(lam2.imag) < 1e-9:
        v2 = result.eigenvectors[:, 1].real
        _draw_eigenvector(ax, v2, EV2_COLOUR, xlim, ylim, label=f"EV₂ (λ={lam2.real:.3f})")
        legend_handles.append(Line2D([0], [0], color=EV2_COLOUR, lw=1.5,
                                     linestyle="--", label=f"EV₂  λ = {lam2.real:.3f}"))

    # Fixed point marker
    ax.plot(0, 0, "o", color="white", markersize=6, zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground=traj_col)])

    # ── Labels & title ────────────────────────────────────────────────────────
    stability_str = (
        "Stable" if result.is_stable is True
        else "Unstable" if result.is_stable is False
        else "Neutral"
    )
    colour_map = {True: STABLE_COLOUR, False: UNSTABLE_COLOUR, None: NEUTRAL_COLOUR}
    title_col = colour_map[result.is_stable]

    a00, a01, a10, a11 = A[0,0], A[0,1], A[1,0], A[1,1]
    matrix_str = (
        f"x' = {a00:+.2f}x {'+' if a01 >= 0 else ''}{a01:.2f}y\n"
        f"y' = {a10:+.2f}x {'+' if a11 >= 0 else ''}{a11:.2f}y"
    )

    ax.set_title(
        f"{result.classification}  ·  {stability_str}\n"
        f"tr = {result.trace:.3f},   det = {result.determinant:.3f}",
        color=title_col, fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("x", color="#8fa8c4", fontsize=11)
    ax.set_ylabel("y", color="#8fa8c4", fontsize=11, rotation=0, labelpad=12)

    # Matrix annotation
    ax.text(
        0.02, 0.02, matrix_str, transform=ax.transAxes,
        fontsize=8.5, color="#8fa8c4", verticalalignment="bottom",
        fontfamily="monospace",
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper right",
            facecolor=GRID_COLOUR, edgecolor=GRID_COLOUR,
            labelcolor="#d0e0f0", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOUR)
        print(f"  Figure saved → {save_path}")

    if show:
        plt.show()

    return fig


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _integrate_rk4(
    A: np.ndarray,
    ic: np.ndarray,
    t_arr: np.ndarray,
    xlim: tuple,
    ylim: tuple,
) -> np.ndarray:
    """
    RK4 integrator for  x' = Ax  from initial condition ic.
    Stops early if trajectory leaves the plot bounds.
    """
    traj = [ic.copy()]
    state = ic.copy().astype(float)
    pad = 0.5  # allow a little overshoot before clipping

    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        k1 = A @ state
        k2 = A @ (state + 0.5 * dt * k1)
        k3 = A @ (state + 0.5 * dt * k2)
        k4 = A @ (state + dt * k3)
        state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        if (state[0] < xlim[0] - pad or state[0] > xlim[1] + pad or
                state[1] < ylim[0] - pad or state[1] > ylim[1] + pad):
            break
        traj.append(state.copy())

    return np.array(traj)


def _draw_eigenvector(ax, v, colour, xlim, ylim, label=""):
    """Draw an eigenvector line through the origin."""
    if np.linalg.norm(v) < 1e-12:
        return
    v = v / np.linalg.norm(v)
    # Find the t values where the line exits the axes
    t_max = max(abs(xlim[1] / v[0]) if abs(v[0]) > 1e-12 else 0,
                abs(ylim[1] / v[1]) if abs(v[1]) > 1e-12 else 0)
    t_max = min(t_max, 10)
    pts = np.array([[-t_max * v[0], -t_max * v[1]],
                    [ t_max * v[0],  t_max * v[1]]])
    ax.plot(pts[:, 0], pts[:, 1], "--", color=colour, linewidth=1.4,
            alpha=0.85, zorder=5)