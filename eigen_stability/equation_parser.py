"""
equation_parser.py
------------------
Parses a string like "-2*x + y" or "3y" into the coefficients
(a, b) of a linear expression  a*x + b*y.

Also accepts complex eigenvalue notation:  a + bi  /  a - bi
and sqrt expressions:  1 + sqrt(2)*i  /  1 - sqrt(2)*i
"""

import re, math

# ── Linear expression parser ─────────────────────────────────────────────────

def parse_linear(expr: str) -> tuple[float, float]:
    """
    Parse  "a*x + b*y"  →  (a, b).
    Handles:  -x, 2x, 2*x, -3.5*x + 0.5*y, y, -y, 3*y, etc.
    Raises ValueError with a human-readable message on failure.
    """
    original = expr
    s = expr.replace(" ", "")

    if not s:
        raise ValueError("Empty expression.")

    # Normalise double-negatives
    s = re.sub(r"--", "+", s)

    # Protect leading minus
    if s.startswith("-"):
        s = "0" + s

    # Tokenise into signed terms  (+coeff*var  or  +coeff)
    # We only care about terms that contain x or y
    cx, cy = 0.0, 0.0

    # Find all terms
    term_pattern = re.compile(
        r"[+\-]?"
        r"(?:"
        r"([0-9]*\.?[0-9]+)\*?([xy])"   # e.g. 2*x, 3.5y
        r"|"
        r"([xy])\*?([0-9]*\.?[0-9]*)"   # e.g. x*2, y (coeff after var)
        r"|"
        r"([0-9]*\.?[0-9]+)"             # constant — ignored for linear
        r")"
    )

    # Split by + / - preserving sign, e.g.  "2*x-y"  → ["+2*x", "-y"]
    tokens = re.findall(r"[+\-]?[^+\-]+", s)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        sign = -1.0 if tok.startswith("-") else 1.0
        tok_body = tok.lstrip("+-")

        # coeff * x  or  coeff x
        m = re.fullmatch(r"([0-9]*\.?[0-9]*)\*?x", tok_body)
        if m:
            raw = m.group(1)
            cx += sign * (1.0 if raw == "" else float(raw))
            continue

        # coeff * y  or  coeff y
        m = re.fullmatch(r"([0-9]*\.?[0-9]*)\*?y", tok_body)
        if m:
            raw = m.group(1)
            cy += sign * (1.0 if raw == "" else float(raw))
            continue

        # Pure constant — ignore (not part of linear term)
        m = re.fullmatch(r"[0-9]*\.?[0-9]+", tok_body)
        if m:
            continue

        raise ValueError(
            f"Could not parse term '{tok}' in expression '{original}'.\n"
            f"    Use only x and y terms, e.g.  -2*x + y  or  3y - 0.5x"
        )

    return (cx, cy)


def parse_matrix_from_equations(eq1: str, eq2: str) -> list[list[float]]:
    """
    Build the 2×2 matrix A from
        dx/dt = eq1(x, y)
        dy/dt = eq2(x, y)
    """
    a, b = parse_linear(eq1)
    c, d = parse_linear(eq2)
    return [[a, b], [c, d]]


# ── Eigenvalue string parser (for manual override mode) ──────────────────────

def _sqrt_val(s: str) -> float:
    """Parse 'sqrt(N)' → √N, or a plain number."""
    s = s.strip()
    m = re.fullmatch(r"sqrt\(([0-9]*\.?[0-9]+)\)", s)
    if m:
        return math.sqrt(float(m.group(1)))
    m = re.fullmatch(r"√([0-9]*\.?[0-9]+)", s)
    if m:
        return math.sqrt(float(m.group(1)))
    return float(s)


def parse_eigenvalue(s: str) -> complex:
    """
    Parse eigenvalue strings such as:
        2           →  2 + 0i
        -1.5        →  -1.5 + 0i
        1 + 2i      →  1 + 2i
        -1 - sqrt(2)*i  →  -1 - √2·i
        1 + sqrt(2)i    →  1 + √2·i
        3i          →  0 + 3i
    """
    original = s
    s = s.replace(" ", "").replace("×", "*")

    # Pure imaginary  e.g.  3i  or  -2i
    m = re.fullmatch(r"([+\-]?[0-9]*\.?[0-9]*)i", s)
    if m:
        raw = m.group(1)
        im = 1.0 if raw in ("", "+") else -1.0 if raw == "-" else float(raw)
        return complex(0, im)

    # Real-only  e.g.  2  or  -1.5
    try:
        return complex(float(s), 0)
    except ValueError:
        pass

    # Complex  a ± b*i  or  a ± sqrt(N)*i  or  a ± sqrt(N)i
    pattern = re.compile(
        r"^([+\-]?[0-9]*\.?[0-9]+)"           # real part
        r"([+\-])"                              # sign
        r"(sqrt\([0-9.]+\)|√[0-9.]+|[0-9]*\.?[0-9]+)\*?i$"  # imag part
    )
    m = pattern.match(s)
    if m:
        re_part = float(m.group(1))
        sign    = 1.0 if m.group(2) == "+" else -1.0
        im_part = sign * _sqrt_val(m.group(3))
        return complex(re_part, im_part)

    raise ValueError(
        f"Cannot parse eigenvalue '{original}'.\n"
        f"    Valid forms:  2   -1.5   1+2i   -1-sqrt(2)*i   1+√2i"
    )


def parse_eigenvector(s: str) -> list[float]:
    """
    Parse a 2-component vector from various formats:
        (0.9, 0.4)   [3, 1]   3 1   0.9 0.4
    Returns [v1, v2] normalised to unit length.
    """
    # Strip brackets/parens
    s = re.sub(r"[()[\]]", " ", s)
    # Split by comma or whitespace
    parts = re.split(r"[,\s]+", s.strip())
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise ValueError(
            f"Expected 2 components for eigenvector, got: '{s}'"
        )
    try:
        v = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Non-numeric eigenvector component in '{s}'")
    norm = math.sqrt(v[0]**2 + v[1]**2)
    if norm < 1e-12:
        raise ValueError("Zero eigenvector — cannot normalise.")
    return [v[0] / norm, v[1] / norm]