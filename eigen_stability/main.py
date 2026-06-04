"""
main.py  —  Eigenvalue & Phase Portrait Analyser
Interactive terminal application for 2D linear ODE systems.

Run:
    python main.py
    python main.py --examples
"""

import sys, math, argparse
import numpy as np

from .ui              import clear, header, section, success, warn, info, error, hr, c, menu, prompt_input, confirm
from .equation_parser import parse_matrix_from_equations, parse_eigenvalue, parse_eigenvector
from .eigen_classifier import analyse, format_report, EigenResult
from .phase_portrait  import plot_phase_portrait

# ── Preset exercises ──────────────────────────────────────────────────────────

PRESETS = [
    {
        "label":  "Exercise 4  —  dx=2x−y,  dy=y",
        "desc":   "Unstable node  λ=2,1",
        "eq1": "2*x - y",  "eq2": "y",
    },
    {
        "label":  "Exercise 5  —  dx=−x+y,  dy=x−3y",
        "desc":   "Stable node  λ≈−0.6, −3.4",
        "eq1": "-x + y",   "eq2": "x - 3*y",
    },
    {
        "label":  "Exercise 6  —  dx=−2x+y,  dy=y",
        "desc":   "Unstable node  λ=−2, 1",
        "eq1": "-2*x + y", "eq2": "y",
    },
    {
        "label":  "Exercise 7  —  dx=3y,  dy=−x+2y",
        "desc":   "Unstable spiral  λ=1±√2 i",
        "eq1": "3*y",       "eq2": "-x + 2*y",
    },
    {
        "label":  "Exercise 11 —  λ=1,3  v=(3,1),(1,1)",
        "desc":   "Unstable node (custom eigendata)",
        "eq1": None,        "eq2": None,
        "custom_eigs": [
            {"lam": complex(1,0), "vec": [3/math.sqrt(10), 1/math.sqrt(10)]},
            {"lam": complex(3,0), "vec": [1/math.sqrt(2),  1/math.sqrt(2)]},
        ],
    },
    {
        "label":  "Stable spiral  —  dx=−x−2y,  dy=2x−y",
        "desc":   "λ=−1±2i",
        "eq1": "-x - 2*y",  "eq2": "2*x - y",
    },
    {
        "label":  "Centre  —  dx=−y,  dy=x",
        "desc":   "Purely imaginary eigenvalues",
        "eq1": "-y",         "eq2": "x",
    },
    {
        "label":  "Saddle  —  dx=x,  dy=−y",
        "desc":   "Opposite-sign real eigenvalues",
        "eq1": "x",          "eq2": "-y",
    },
]

# ── Input mode helpers ────────────────────────────────────────────────────────

def _validate_eq(val):
    from .equation_parser import parse_linear
    try:
        parse_linear(val)
        return val
    except ValueError as e:
        raise ValueError(str(e))

def _validate_eig(val):
    return parse_eigenvalue(val)

def _validate_vec(val):
    return parse_eigenvector(val)

def _build_matrix_from_eigs(lam1, vec1, lam2, vec2):
    """Reconstruct A = V · diag(λ) · V⁻¹ from eigendata."""
    V  = np.column_stack([vec1, vec2]).astype(complex)
    D  = np.diag([lam1, lam2])
    try:
        A = V @ D @ np.linalg.inv(V)
    except np.linalg.LinAlgError:
        raise ValueError("Eigenvectors are linearly dependent — cannot reconstruct A.")
    return np.real(A)

# ── Input modes ───────────────────────────────────────────────────────────────

def mode_equations() -> EigenResult:
    """User types  dx/dt = ...  and  dy/dt = ..."""
    section("Enter the system equations")
    info("Write each equation as a linear expression in x and y.")
    info("Examples:  -x + y     2*x - 3*y     3*y     -2*x")
    print()

    eq1 = prompt_input("dx/dt  =", validator=_validate_eq)
    eq2 = prompt_input("dy/dt  =", validator=_validate_eq)

    try:
        M = parse_matrix_from_equations(eq1, eq2)
    except ValueError as e:
        error(str(e)); sys.exit(1)

    A = np.array(M)
    return analyse(A)


def mode_manual_matrix() -> EigenResult:
    """User types the four matrix entries individually."""
    section("Enter the Jacobian matrix  A = [[a, b], [c, d]]")
    info("For the system  x' = ax + by,  y' = cx + dy")
    print()

    def _float(val):
        try:   return float(val)
        except: raise ValueError("Enter a single number, e.g.  -2  or  0.5")

    a = prompt_input("a  (row 1, col 1)", validator=_float)
    b = prompt_input("b  (row 1, col 2)", validator=_float)
    c_ = prompt_input("c  (row 2, col 1)", validator=_float)
    d = prompt_input("d  (row 2, col 2)", validator=_float)

    A = np.array([[a, b], [c_, d]])
    return analyse(A)


def mode_eigenvalues() -> EigenResult:
    """User provides eigenvalues + eigenvectors (e.g. from a problem sheet)."""
    section("Enter eigenvalues and eigenvectors")
    info("Eigenvalue formats:  2   -1.5   1+2i   -1-sqrt(2)*i   1+√2i")
    info("Eigenvector formats: (3, 1)   0.9 0.4   [0.4, -0.9]")
    print()

    lam1 = prompt_input("λ₁  =", validator=_validate_eig)
    vec1 = prompt_input("v₁  =  (v₁, v₂)", validator=_validate_vec)
    print()

    # For complex conjugate pairs, auto-fill λ₂ and v₂
    if abs(lam1.imag) > 1e-10:
        lam2 = lam1.conjugate()
        vec2 = [vec1[0], -vec1[1]]
        info(f"Complex conjugate detected  →  λ₂ = {lam2.real:.4f} {'-' if lam2.imag < 0 else '+'} {abs(lam2.imag):.4f}i  (auto-filled)")
    else:
        lam2 = prompt_input("λ₂  =", validator=_validate_eig)
        vec2 = prompt_input("v₂  =  (v₁, v₂)", validator=_validate_vec)

    try:
        A = _build_matrix_from_eigs(lam1, vec1, lam2, vec2)
    except ValueError as e:
        error(str(e)); sys.exit(1)

    return analyse(A)


def mode_preset(idx: int) -> EigenResult:
    """Load a preset exercise."""
    p = PRESETS[idx]
    if p.get("custom_eigs"):
        eigs = p["custom_eigs"]
        A = _build_matrix_from_eigs(
            eigs[0]["lam"], eigs[0]["vec"],
            eigs[1]["lam"], eigs[1]["vec"],
        )
    else:
        M = parse_matrix_from_equations(p["eq1"], p["eq2"])
        A = np.array(M)
    return analyse(A)

# ── Result display ─────────────────────────────────────────────────────────────

_STAB_COLOUR = {"Stable Node": "bgreen", "Stable Spiral": "bgreen",
                "Unstable Node": "bred", "Unstable Spiral": "bred",
                "Saddle Point": "byellow", "Centre": "bcyan",
                "Stable Star / Degenerate Node": "bgreen",
                "Unstable Star / Degenerate Node": "bred"}

def display_result(result: EigenResult):
    section("Analysis results")
    print()

    # Matrix
    A = result.matrix
    print(c("  Jacobian A:", "bold"))
    print(c(f"    ┌ {A[0,0]:8.4f}  {A[0,1]:8.4f} ┐", "white"))
    print(c(f"    └ {A[1,0]:8.4f}  {A[1,1]:8.4f} ┘", "white"))
    print()

    # Scalar invariants
    print(c(f"  trace(A)  =  {result.trace:.4f}     det(A)  =  {result.determinant:.4f}     disc  =  {result.discriminant:.4f}", "cyan"))
    print()

    # Eigenvalues
    def fmt_lam(z):
        if abs(z.imag) < 1e-10:
            return f"{z.real:+.4f}"
        sign = "+" if z.imag >= 0 else "−"
        return f"{z.real:+.4f}  {sign}  {abs(z.imag):.4f}i"

    l1, l2 = result.eigenvalues
    v1 = result.eigenvectors[:, 0]
    v2 = result.eigenvectors[:, 1]

    print(c(f"  λ₁  =  {fmt_lam(l1)}", "byellow") +
          c(f"        eigenvector:  ({v1[0].real:+.4f},  {v1[1].real:+.4f})ᵀ", "yellow"))
    print(c(f"  λ₂  =  {fmt_lam(l2)}", "byellow") +
          c(f"        eigenvector:  ({v2[0].real:+.4f},  {v2[1].real:+.4f})ᵀ", "yellow"))
    print()

    # General solution
    is_complex = abs(l1.imag) > 1e-10
    print(c("  General solution:", "bold"))
    if is_complex:
        a  = l1.real; b = abs(l1.imag)
        print(c(f"    z(t)  =  e^({a:+.4f}t) · [ C₁·cos({b:.4f}t) + C₂·sin({b:.4f}t) ]", "bcyan"))
    else:
        print(c(f"    z(t)  =  C₁·e^({l1.real:+.4f}t)·({v1[0].real:+.4f}, {v1[1].real:+.4f})ᵀ", "bcyan"))
        print(c(f"           +  C₂·e^({l2.real:+.4f}t)·({v2[0].real:+.4f}, {v2[1].real:+.4f})ᵀ", "bcyan"))
    print()

    # Classification banner
    col = _STAB_COLOUR.get(result.classification, "white")
    stab = ("STABLE" if result.is_stable is True
            else "UNSTABLE" if result.is_stable is False
            else "NEUTRALLY STABLE")
    banner = f"  ┌─ {result.classification}  ·  {stab} ─┐"
    print(c(banner, col, "bold"))
    print()

    # Description (wrapped)
    words = result.description.split()
    line, lines = "  │  ", []
    for w in words:
        if len(line) + len(w) + 1 > 74:
            lines.append(line); line = "  │  " + w + " "
        else:
            line += w + " "
    if line.strip(): lines.append(line)
    for ln in lines:
        print(c(ln, "cyan"))
    print()

    # Eigenvector line hint
    if not is_complex:
        info("Trajectories near the eigenvector directions become straight lines.")
    if result.classification == "Saddle Point":
        info("Solutions along the stable manifold approach the origin; all others diverge.")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main_loop():
    while True:
        header()

        top_labels = [
            "Enter system equations        ",
            "Enter eigenvalues/eigenvectors",
            "Enter matrix entries          ",
            "Load preset exercise          ",
            "Quit                          ",
        ]
        top_descs = [
            "dx/dt = f(x,y)  ·  dy/dt = g(x,y)  — most natural",
            "λ = a ± bi,  v = (v₁, v₂)  — from a problem sheet",
            "Fill in matrix A directly",
            "Textbook exercises pre-loaded",
            "",
        ]

        choice = menu("Choose input method", top_labels, top_descs)

        if choice < 0 or choice == 4:
            clear()
            print(c("\n  Goodbye.\n", "cyan"))
            break

        # ── Preset sub-menu ────────────────────────────────────────────────
        if choice == 3:
            header()
            preset_labels = [p["label"] for p in PRESETS] + ["← Back"]
            preset_descs  = [p["desc"]  for p in PRESETS] + [""]
            pi = menu("Select exercise", preset_labels, preset_descs)
            if pi < 0 or pi == len(PRESETS):
                continue
            result = mode_preset(pi)

        # ── Equation input ─────────────────────────────────────────────────
        elif choice == 0:
            header()
            result = mode_equations()

        # ── Eigenvalue input ───────────────────────────────────────────────
        elif choice == 1:
            header()
            result = mode_eigenvalues()

        # ── Matrix input ───────────────────────────────────────────────────
        elif choice == 2:
            header()
            result = mode_manual_matrix()

        # ── Display ────────────────────────────────────────────────────────
        display_result(result)
        hr()

        save_path = None
        if confirm("Save phase portrait to file?"):
            save_path = prompt_input("Filename", default="portrait.png")

        plot_phase_portrait(result, save_path=save_path, show=True)

        print()
        if not confirm("Analyse another system?"):
            clear()
            print(c("\n  Goodbye.\n", "cyan"))
            break


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def run_examples():
    """Non-interactive: print reports for all presets."""
    from .eigen_classifier import format_report
    for i, p in enumerate(PRESETS):
        result = mode_preset(i)
        print(format_report(result))
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eigenvalue Phase Portrait Analyser")
    parser.add_argument("--examples", action="store_true",
                        help="Print reports for all preset exercises (non-interactive)")
    args = parser.parse_args(argv)

    if args.examples:
        run_examples()
    else:
        try:
            main_loop()
        except KeyboardInterrupt:
            print(c("\n\n  Interrupted.\n", "dim"))
            sys.exit(0)


if __name__ == "__main__":
    main()