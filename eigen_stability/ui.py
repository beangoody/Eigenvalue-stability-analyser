"""
ui.py  —  terminal UI helpers (zero external deps, stdlib only)
Uses ANSI escape codes for colour/bold, and a lightweight menu engine.
"""
import sys, os, re, math

# ── ANSI codes ────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"

FG = {
    "black":   "\033[30m", "red":     "\033[31m", "green":   "\033[32m",
    "yellow":  "\033[33m", "blue":    "\033[34m", "magenta": "\033[35m",
    "cyan":    "\033[36m", "white":   "\033[37m",
    "bred":    "\033[91m", "bgreen":  "\033[92m", "byellow": "\033[93m",
    "bblue":   "\033[94m", "bmagenta":"\033[95m", "bcyan":   "\033[96m",
}
BG = {
    "black":  "\033[40m",  "blue":    "\033[44m",
    "cyan":   "\033[46m",  "white":   "\033[47m",
}

def c(text, *styles):
    codes = ""
    for s in styles:
        if s == "bold":    codes += BOLD
        elif s == "dim":   codes += DIM
        elif s == "italic":codes += ITALIC
        elif s in FG:      codes += FG[s]
        elif s in BG:      codes += BG[s]
    return f"{codes}{text}{RESET}"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def width():
    try:    return os.get_terminal_size().columns
    except: return 80

def hr(char="─", col="blue"):
    print(c(char * width(), col))

def header():
    clear()
    w = width()
    print()
    title = "  EIGEN PHASE PORTRAIT ANALYSER  "
    pad = (w - len(title)) // 2
    print(c(" " * pad + title + " " * pad, "bold", "bblue"))
    sub  = "2D Linear ODE Systems  ·  Phase Portrait  ·  Stability Analysis"
    print(c(sub.center(w), "cyan"))
    print()
    hr("═")
    print()

def section(title):
    print()
    print(c(f"  ▸  {title}", "bold", "bcyan"))
    print(c("  " + "─" * (len(title) + 5), "blue"))

def success(msg): print(c(f"  ✓  {msg}", "bgreen"))
def warn(msg):    print(c(f"  ⚠  {msg}", "byellow"))
def info(msg):    print(c(f"  ·  {msg}", "cyan"))
def error(msg):   print(c(f"  ✗  {msg}", "bred"))

# ── Arrow-key menu ─────────────────────────────────────────────────────────────

def _getch():
    """Read a single keypress. Returns the char or a normalized escape sequence."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "\x1b[A"
            if ch2 == "P":
                return "\x1b[B"
            if ch2 == "K":
                return "\x1b[D"
            if ch2 == "M":
                return "\x1b[C"
            return ch + ch2
        return ch

    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":          # escape sequence
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def menu(prompt, options, descriptions=None):
    """
    Interactive arrow-key menu.
    options       : list of short labels
    descriptions  : optional list of same length with detail text
    Returns the index chosen.
    """
    idx = 0
    n   = len(options)
    print(c(f"\n  {prompt}", "bold", "white"))
    print()

    def render():
        # Move cursor up to redraw
        sys.stdout.write(f"\033[{n + 1}A")
        sys.stdout.flush()
        print()
        for i, opt in enumerate(options):
            prefix = c("  › ", "bblue", "bold") if i == idx else "    "
            label  = c(opt, "bold", "white") if i == idx else c(opt, "white")
            desc   = ""
            if descriptions and i < len(descriptions):
                d = descriptions[i]
                desc = "  " + (c(d, "bblue") if i == idx else c(d, "dim"))
            print(f"{prefix}{label}{desc}")

    # Initial draw
    print()
    for i, opt in enumerate(options):
        prefix = c("  › ", "bblue", "bold") if i == idx else "    "
        label  = c(opt, "bold", "white") if i == idx else c(opt, "white")
        desc   = ""
        if descriptions and i < len(descriptions):
            d = descriptions[i]
            desc = "  " + (c(d, "bblue") if i == idx else c(d, "dim"))
        print(f"{prefix}{label}{desc}")

    while True:
        key = _getch()
        if key in ("\x1b[A", "\x1b[D"):   # up / left
            idx = (idx - 1) % n
            render()
        elif key in ("\x1b[B", "\x1b[C"): # down / right
            idx = (idx + 1) % n
            render()
        elif key in ("\r", "\n", " "):     # enter / space
            print()
            return idx
        elif key == "q":
            print()
            return -1   # quit signal

def prompt_input(label, default=None, validator=None):
    """Single-line text prompt with optional validation."""
    hint = f"  [{c(default, 'dim')}]" if default else ""
    while True:
        val = input(c(f"\n  {label}{hint}: ", "bold", "byellow")).strip()
        if val == "" and default is not None:
            val = default
        if validator:
            try:
                result = validator(val)
                return result
            except ValueError as e:
                error(str(e))
        else:
            return val

def confirm(msg):
    ans = input(c(f"\n  {msg}  [y/N]: ", "bold", "white")).strip().lower()
    return ans == "y"