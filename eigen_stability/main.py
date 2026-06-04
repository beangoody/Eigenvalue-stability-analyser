"""
main.py
-------
CLI entry point for the Eigenvalue & Phase Portrait Classifier.

Usage
-----
  # Interactive mode (prompts for matrix input):
  python main.py

  # Direct mode (pass matrix as arguments):
  python main.py --matrix 1 0 0 -1

  # Save figure without displaying:
  python main.py --matrix -1 2 0 -3 --save portrait.png --no-show

  # Run all built-in examples:
  python main.py --examples
"""

import argparse
import sys
import numpy as np

from eigen_classifier import analyse, format_report
from phase_portrait import plot_phase_portrait


# ─── Built-in example matrices ───────────────────────────────────────────────
EXAMPLES = {
    "Stable Node":      np.array([[-2,  0], [ 0, -3]], dtype=float),
    "Unstable Node":    np.array([[ 2,  0], [ 0,  3]], dtype=float),
    "Saddle Point":     np.array([[ 1,  0], [ 0, -1]], dtype=float),
    "Stable Spiral":    np.array([[-1,  2], [-2, -1]], dtype=float),
    "Unstable Spiral":  np.array([[ 1,  2], [-2,  1]], dtype=float),
    "Centre":           np.array([[ 0,  1], [-1,  0]], dtype=float),
}


def get_matrix_interactive() -> np.ndarray:
    """Prompt the user to enter a 2x2 matrix row by row."""
    print("\n  Enter the 2×2 matrix A for the system  x' = Ax")
    print("  ─────────────────────────────────────────────")
    entries = []
    for i in range(2):
        while True:
            raw = input(f"  Row {i+1} (space-separated, e.g. '1 -2'): ").strip()
            parts = raw.split()
            if len(parts) == 2:
                try:
                    entries.extend([float(p) for p in parts])
                    break
                except ValueError:
                    pass
            print("  ✗  Enter exactly 2 numbers separated by a space.")
    return np.array(entries, dtype=float).reshape(2, 2)


def run_analysis(matrix: np.ndarray, save_path=None, show=True):
    """Run full analysis + plot for a given matrix."""
    result = analyse(matrix)
    print("\n" + format_report(result))
    plot_phase_portrait(result, save_path=save_path, show=show)


def run_examples():
    """Cycle through all built-in example matrices."""
    print("\n  Running all built-in examples...")
    for name, A in EXAMPLES.items():
        print(f"\n  ── {name} ──")
        result = analyse(A)
        print(format_report(result))
        plot_phase_portrait(result, show=True)


def main():
    parser = argparse.ArgumentParser(
        description="Eigenvalue & Phase Portrait Classifier for 2×2 ODE systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--matrix", "-m",
        nargs=4,
        metavar=("A00", "A01", "A10", "A11"),
        type=float,
        help="Matrix entries in row-major order: a00 a01 a10 a11",
    )
    parser.add_argument(
        "--save", "-s",
        metavar="PATH",
        default=None,
        help="Save the phase portrait to this file path (e.g. portrait.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open the plot window (useful when saving headlessly)",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run all built-in example matrices and show their portraits",
    )

    args = parser.parse_args()

    # ── Examples mode ─────────────────────────────────────────────────────────
    if args.examples:
        run_examples()
        return

    # ── Matrix from CLI args ──────────────────────────────────────────────────
    if args.matrix:
        A = np.array(args.matrix, dtype=float).reshape(2, 2)
        run_analysis(A, save_path=args.save, show=not args.no_show)
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Eigenvalue & Phase Portrait Classifier")
    print("=" * 52)
    print("  (Run with --help for command-line options)")

    while True:
        A = get_matrix_interactive()
        run_analysis(A, save_path=args.save, show=not args.no_show)

        again = input("\n  Analyse another matrix? [y/N]: ").strip().lower()
        if again != "y":
            print("\n  Goodbye.\n")
            break


if __name__ == "__main__":
    main()