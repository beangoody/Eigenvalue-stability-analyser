# Eigenvalue & Phase Portrait Classifier

A Python tool for analysing 2×2 linear ODE systems. Enter any matrix **A** for the system **x' = Ax**, and the tool will:

- Compute **eigenvalues** and **eigenvectors** using NumPy's linear algebra routines
- **Classify the fixed point** at the origin using the trace-determinant plane
- **Plot the phase portrait** with a quiver direction field, coloured streamlines, and eigenvector overlays

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

---

## Usage

### Interactive mode
```bash
python main.py
```
Prompts you to enter a 2×2 matrix row by row.

### Command-line mode
```bash
# Stable spiral: eigenvalues -1 ± 2i
python main.py --matrix -1 2 -2 -1

# Saddle point
python main.py --matrix 1 0 0 -1

# Save the figure without opening a window
python main.py --matrix -2 1 0 -3 --save output.png --no-show
```

### Run all built-in examples
```bash
python main.py --examples
```
Cycles through a stable node, unstable node, saddle, stable spiral, unstable spiral, and centre — useful for a quick sanity check.

---

## Project Structure

```
eigen-phase-portrait/
├── main.py               # CLI entry point
├── eigen_classifier.py   # Eigenvalue computation & fixed point classification
├── phase_portrait.py     # Phase portrait rendering (matplotlib)
├── requirements.txt
└── README.md
```

---

## Mathematical Background

For the system **x' = Ax**, the long-term behaviour near the origin is determined by the eigenvalues of **A**.

The classification depends on the **trace** τ = tr(A) and **determinant** Δ = det(A):

- **Δ < 0** → saddle (eigenvalues real, opposite signs)
- **Δ > 0, τ² − 4Δ > 0** → node (real distinct eigenvalues)
- **Δ > 0, τ² − 4Δ < 0** → spiral or centre (complex eigenvalues)
- **Stability** is determined by the sign of τ (the sum of the real parts of the eigenvalues)

---

## Example Output

```
====================================================
 EIGENVALUE PHASE PORTRAIT ANALYSIS
====================================================
  Matrix A:
    [  -1.000    2.000 ]
    [  -2.000   -1.000 ]

  Trace       : -2.0000
  Determinant : 5.0000
  Discriminant: -16.0000

  λ₁ = -1.0000 + 2.0000i    eigenvector: [+0.7071,  +0.0000]
  λ₂ = -1.0000 - 2.0000i    eigenvector: [+0.7071,  +0.0000]

  ┌─ Classification: Stable Spiral
  └─ Stability     : Stable

  The eigenvalues are complex with negative real part. Trajectories
  spiral inward toward the origin — asymptotically stable.
====================================================
```

---

*Part of [Benjamin Goodwin's portfolio](https://yourusername.github.io) — a growing collection of projects in applied maths, algorithms, and systems.*