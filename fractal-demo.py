#!/usr/bin/env python3
"""
interactive_fractal_gpu.py
--------------------------
A unified, live-recomputing fractal explorer for Mandelbrot, Julia, and Newton sets.

This script demonstrates the use of PyTorch for massively parallel computation
to generate fractals in real-time. The core algorithm for each fractal is
vectorized and executed on the GPU (if available).

Author : Nathan
Course  : COMP3710
"""

# 0. Imports & config
import torch, numpy as np, matplotlib.pyplot as plt
import argparse  # Import the argument parsing library
plt.ion()

# --- NEW: Command-Line Argument Parsing ---
# This section sets up the ability to choose the fractal from the terminal.
parser = argparse.ArgumentParser(
    description="An interactive fractal explorer using PyTorch.",
    formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
)
# Create a group where only one argument can be chosen at a time
group = parser.add_mutually_exclusive_group()
group.add_argument('--mandelbrot', action='store_true', help='Display the Mandelbrot set (default).')
group.add_argument('--julia', action='store_true', help='Display the Julia set.')
group.add_argument('--newton', action='store_true', help='Display the Newton fractal for z³-1.')

args = parser.parse_args()

# Determine which fractal to display based on the command-line flag
if args.julia:
    FRACTAL = "julia"
elif args.newton:
    FRACTAL = "newton"
else: # Default to Mandelbrot if --mandelbrot is specified or no flag is given
    FRACTAL = "mandelbrot"
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device for computation: {device}")
print(f"[INFO] Selected fractal: {FRACTAL.title()}")

# 1. Parameters
# The FRACTAL variable is now set by the command-line arguments above.
JULIA_K  = torch.complex(torch.tensor(-0.8), torch.tensor(0.156))
BASE_ITER = 200
ITER_CAP = 50000

# 2. Colour-mapper
def fancy_colour(a: np.ndarray) -> np.ndarray:
    cyc = (2*np.pi*a/20.0)[..., None]
    img = np.concatenate([10+20*np.cos(cyc), 30+50*np.sin(cyc), 155-80*np.cos(cyc)], axis=2)
    img[a == a.max()] = 0
    return np.clip(img, 0, 255).astype(np.uint8)

# 3. Fractal computation helper
@torch.no_grad()
def compute_fractal(x_min, x_max, y_min, y_max, width_px, base_iter):
    spacing = (x_max - x_min) / width_px
    Y, X = np.mgrid[y_min:y_max:spacing, x_min:x_max:spacing]
    
    x = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(Y).to(torch.float32)
    
    c_grid = torch.complex(x, y).to(device)
    z = torch.zeros_like(c_grid).to(device)

    if FRACTAL.lower() == "julia":
        z = c_grid.clone()
        c = JULIA_K.to(device).expand_as(z)
    elif FRACTAL.lower() == "newton":
        z = c_grid.clone()
        c = None
    else: # Mandelbrot
        c = c_grid.clone()
        
    zs, ns = z, torch.zeros_like(z, dtype=torch.int32)
    
    span = max(x_max - x_min, y_max - y_min)
    calculated_iter = int(base_iter * max(1, 3 / span))
    max_iter = min(calculated_iter, ITER_CAP)

    for _ in range(max_iter):
        if FRACTAL.lower() == 'newton':
            z_old = zs.clone()
            zs = zs - (zs**3 - 1) / (3 * zs**2 + 1e-6)
            converged = torch.abs(zs - z_old) < 1e-5
            ns += ~converged
            if converged.all(): break
            continue
        else: # Mandelbrot or Julia
            zs = zs*zs + c
        
        bounded = torch.abs(zs) < 4.0
        ns += bounded
        if not bounded.any(): break

    return fancy_colour(ns.cpu().numpy())

# 4. Initial render
if FRACTAL.lower() == 'newton':
    INIT_BOUNDS = (-2.0, 2.0, -2.0, 2.0)
    title_str = "Newton Fractal (z³-1)"
else:
    INIT_BOUNDS = (-2.0, 1.0, -1.3, 1.3)
    title_str = f"{FRACTAL.title()} Set"

fig, ax = plt.subplots(figsize=(8, 8) if FRACTAL.lower() == 'newton' else (10, 6))
ax.set_title(f"{title_str} — zoom/pan")
ax.axis("off")
plt.tight_layout(pad=0)

fig.canvas.draw()
initial_pixel_width = int(ax.get_window_extent().width)
img = compute_fractal(*INIT_BOUNDS, initial_pixel_width, BASE_ITER)
im = ax.imshow(img, origin="lower", extent=INIT_BOUNDS)

# 5. Callback for interactive zoom/pan
def on_release(event):
    if event.inaxes is not ax: return
    x_min, x_max, y_min, y_max = ax.get_xlim() + ax.get_ylim()
    if np.isclose([x_min, x_max, y_min, y_max], im.get_extent()).all(): return

    ax.set_title("Computing…")
    fig.canvas.draw_idle()
    pixel_width = int(ax.get_window_extent().width)
    new_img = compute_fractal(x_min, x_max, y_min, y_max, pixel_width, BASE_ITER)
    im.set_data(new_img)
    im.set_extent([x_min, x_max, y_min, y_max])
    ax.set_title(f"{title_str} — done")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("scroll_event", on_release)
plt.show(block=True)