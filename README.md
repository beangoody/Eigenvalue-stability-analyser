# Eigenvalue & Phase Portrait Analyser

A terminal application for analysing 2×2 linear ODE systems of the form **x' = Ax**. Enter your system as natural equations, raw matrix entries, or eigenvalues from a problem sheet — the tool computes the full eigenanalysis, classifies the fixed point, writes the general solution, and renders the phase portrait.

---

## Features

- **Natural equation input** — type `dx/dt = -x + y` exactly as it appears in a textbook
- **Eigenvalue / eigenvector input** — accepts complex forms like `1 + sqrt(2)*i` or `-1 - √2i`
- **Matrix entry mode** — fill in the four entries of A individually
- **Preset exercises** — 8 textbook exercises pre-loaded, selectable from an arrow-key menu
- **General solution** — written out symbolically after each analysis
- **Phase portrait** — RK4-integrated streamlines, direction field, eigenvector overlays, stability-coded colours

---

## Fixed Point Classifications

| Classification | Eigenvalue condition |
|---|---|
| Stable Node | Both real, both negative |
| Unstable Node | Both real, both positive |
| Saddle Point | Both real, opposite signs |
| Stable Spiral | Complex conjugate, Re(λ) < 0 |
| Unstable Spiral | Complex conjugate, Re(λ) > 0 |
| Centre | Purely imaginary |
| Stable / Unstable Star | Repeated eigenvalue |

---

## Installation

```bash
git clone https://github.com/yourusername/eigen-phase-portrait.git
cd eigen-phase-portrait
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, Matplotlib. No other dependencies.

---

## Usage

### Interactive mode (recommended)
```bash
python main.py
```

Navigate with ↑↓ arrow keys and Enter. You will be prompted to choose an input method:

```
  Choose input method
  › Enter system equations          dx/dt = f(x,y)  ·  dy/dt = g(x,y)
    Enter eigenvalues/eigenvectors  λ = a ± bi,  v = (v₁, v₂)
    Enter matrix entries            Fill in matrix A directly
    Load preset exercise            Textbook exercises pre-loaded
    Quit
```

#### Equation input
Type each equation as a linear expression in `x` and `y`:
```
dx/dt  =  -x + y
dy/dt  =  x - 3*y
```
Accepted forms: `-x + y`, `2*x - y`, `3*y`, `-2*x`, `x - 3*y`

#### Eigenvalue input
Useful when working from a problem sheet where eigenvalues are already given:
```
λ₁  =  1 + sqrt(2)*i
v₁  =  (0.9, 0.4)
```
For complex conjugate pairs, `λ₂` and `v₂` are filled in automatically.

Accepted eigenvalue formats:
| Input | Interpreted as |
|---|---|
| `2` | 2 |
| `-1.5` | −1.5 |
| `1+2i` | 1 + 2i |
| `-1-sqrt(2)*i` | −1 − √2 i |
| `1+√2i` | 1 + √2 i |
| `3i` | 3i |

#### Preset exercises
Pre-loaded exercises from a dynamics course, selectable by name. Includes stable/unstable nodes, spirals, a saddle, and a centre.

### Non-interactive mode
```bash
python main.py --examples
```
Prints the full eigenanalysis report for all preset exercises to stdout — useful for piping or quick reference.

---

## Project Structure

```
eigen-phase-portrait/
├── main.py               # Application entry point & interactive UI loop
├── ui.py                 # Terminal UI helpers (arrow-key menus, colours, prompts)
├── equation_parser.py    # Parses equations, eigenvalues, and eigenvectors from strings
├── eigen_classifier.py   # Eigenvalue computation & fixed point classification
├── phase_portrait.py     # Phase portrait rendering (matplotlib)
├── requirements.txt
└── README.md
```

---

## Mathematical Background

For the system **x' = Ax**, behaviour near the origin is determined entirely by the eigenvalues of **A**, computed from the characteristic equation λ² − tr(A)·λ + det(A) = 0:

```
λ = ( tr(A) ± √(tr(A)² − 4·det(A)) ) / 2
```

When the discriminant is negative, eigenvalues are complex conjugates `a ± bi`, giving spiral or centre behaviour. Classification uses the trace-determinant plane:

- **det(A) < 0** → saddle
- **det(A) > 0, disc > 0** → node (stability from sign of tr)
- **det(A) > 0, disc < 0** → spiral or centre (stability from sign of tr)
- **tr(A) = 0, disc < 0** → centre

The general solution is:

- **Real distinct eigenvalues:** `z(t) = C₁·e^(λ₁t)·v₁ + C₂·e^(λ₂t)·v₂`
- **Complex eigenvalues λ = a ± bi:** `z(t) = e^(at)·[C₁·cos(bt) + C₂·sin(bt)]`

---

## Example Session

```
  Jacobian A:
    ┌  -1.0000   1.0000 ┐
    └   1.0000  -3.0000 ┘

  trace(A) = -4.0000     det(A) = 2.0000     disc = 8.0000

  λ₁  =  -0.5858        eigenvector:  (+0.9239,  +0.3827)ᵀ
  λ₂  =  -3.4142        eigenvector:  (-0.3827,  +0.9239)ᵀ

  General solution:
    z(t)  =  C₁·e^(-0.5858t)·(+0.9239, +0.3827)ᵀ
           +  C₂·e^(-3.4142t)·(-0.3827, +0.9239)ᵀ

  ┌─ Stable Node  ·  STABLE ─┐
```

---

*Part of [Benjamin Goodwin's portfolio](https://yourusername.github.io) — a growing collection of projects in applied maths, algorithms, and systems.*