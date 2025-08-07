#!/usr/bin/env python3
"""
interactive_fractal_gpu.py
--------------------------
A unified, live-recomputing fractal explorer with multiple visualization modes.

This script demonstrates PyTorch for parallel computation and includes an advanced
analysis of the Newton fractal by visualizing its basins of attraction.

Author : Nathan
Course  : COMP3710
"""

# 0. Imports & config
import torch, numpy as np, matplotlib.pyplot as plt
import argparse
plt.ion()

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="An interactive fractal explorer using PyTorch.", formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_mutually_exclusive_group()
group.add_argument('--mandelbrot', action='store_true', help='Display the Mandelbrot set (default).')
group.add_argument('--julia', action='store_true', help='Display the Julia set.')
group.add_argument('--newton', action='store_true', help='Display the Newton fractal for z³-1.')
# --- NEW: Argument for color mode ---
parser.add_argument('--color_mode', type=str, default='fancy', choices=['fancy', 'basin'],
                    help='Coloring mode. "basin" is only for Newton fractal.')
args = parser.parse_args()

if args.julia: FRACTAL = "julia"
elif args.newton: FRACTAL = "newton"
else: FRACTAL = "mandelbrot"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device for computation: {device}")
print(f"[INFO] Selected fractal: {FRACTAL.title()}, Color Mode: {args.color_mode.title()}")

# 1. Parameters
JULIA_K  = torch.complex(torch.tensor(-0.8), torch.tensor(0.156))
BASE_ITER = 200
ITER_CAP = 50000

# 2. Colour-mappers
def fancy_colour(ns: np.ndarray) -> np.ndarray:
    """Standard coloring based on iteration count."""
    cyc = (2*np.pi*ns/20.0)[..., None]
    img = np.concatenate([10+20*np.cos(cyc), 30+50*np.sin(cyc), 155-80*np.cos(cyc)], axis=2)
    img[ns == ns.max()] = 0
    return np.clip(img, 0, 255).astype(np.uint8)

# --- NEW: Colorizer for Basins of Attraction ---
# --- NEW: Colorizer for Basins of Attraction ---
def colour_by_basin(final_zs: np.ndarray, ns: np.ndarray) -> np.ndarray:
    """Colors the Newton fractal based on which root it converges to."""
    # The three roots of z³ - 1 = 0
    # MODIFIED: Replaced deprecated np.complex with Python's built-in complex()
    root1 = complex(1, 0)
    root2 = complex(-0.5, np.sqrt(3)/2)
    root3 = complex(-0.5, -np.sqrt(3)/2)
    
    # Calculate distance to each root
    d1 = np.abs(final_zs - root1)
    d2 = np.abs(final_zs - root2)
    d3 = np.abs(final_zs - root3)
    
    # Find which root is closest for each pixel
    basin_index = np.argmin(np.stack([d1, d2, d3]), axis=0)
    
    # Define a base color for each basin
    colors = np.array([[255, 100, 100], [100, 255, 100], [100, 100, 255]]) # Red, Green, Blue
    
    # Create the image using the basin colors
    img = colors[basin_index]
    
    # Shade the image by the number of iterations (darker = more iterations)
    # This makes the fractal boundaries stand out.
    shade = 0.3 + 0.7 * (ns / ns.max())
    img = img * shade[..., None]
    
    return np.clip(img, 0, 255).astype(np.uint8)

# 3. Fractal computation helper
@torch.no_grad()
def compute_fractal(x_min, x_max, y_min, y_max, width_px, base_iter):
    # ... (setup code for x, y, c_grid, z is the same) ...
    spacing = (x_max - x_min) / width_px
    Y, X = np.mgrid[y_min:y_max:spacing, x_min:x_max:spacing]
    x = torch.from_numpy(X).to(torch.float32); y = torch.from_numpy(Y).to(torch.float32)
    c_grid = torch.complex(x, y).to(device); z = torch.zeros_like(c_grid).to(device)
    if FRACTAL.lower() == "julia": z = c_grid.clone(); c = JULIA_K.to(device).expand_as(z)
    elif FRACTAL.lower() == "newton": z = c_grid.clone(); c = None
    else: c = c_grid.clone()
    zs, ns = z, torch.zeros_like(z, dtype=torch.int32)
    span = max(x_max - x_min, y_max - y_min)
    max_iter = min(int(base_iter * max(1, 3 / span)), ITER_CAP)

    for i in range(max_iter):
        if FRACTAL.lower() == 'newton':
            z_old = zs.clone()
            zs = zs - (zs**3 - 1) / (3 * zs**2 + 1e-6)
            converged = torch.abs(zs - z_old) < 1e-5
            ns += ~converged
            if converged.all() and i > 5: break # Early exit if all points converge
            continue
        else:
            zs = zs*zs + c
        bounded = torch.abs(zs) < 4.0
        ns += bounded
        if not bounded.any(): break
        
    # --- MODIFIED: Return extra data for Newton fractal ---
    if FRACTAL.lower() == 'newton' and args.color_mode == 'basin':
        return ns.cpu().numpy(), zs.cpu().numpy()
    return ns.cpu().numpy(), None

# 4. Initial render & main logic
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

# --- MODIFIED: Handle different color modes ---
ns, final_zs = compute_fractal(*INIT_BOUNDS, initial_pixel_width, BASE_ITER)
if final_zs is not None:
    img = colour_by_basin(final_zs, ns)
else:
    img = fancy_colour(ns)
im = ax.imshow(img, origin="lower", extent=INIT_BOUNDS)

# 5. Callback for interactive zoom/pan
def on_release(event):
    if event.inaxes is not ax: return
    x_min, x_max, y_min, y_max = ax.get_xlim() + ax.get_ylim()
    if np.isclose([x_min, x_max, y_min, y_max], im.get_extent()).all(): return
    ax.set_title("Computing…"); fig.canvas.draw_idle()
    pixel_width = int(ax.get_window_extent().width)
    ns, final_zs = compute_fractal(x_min, x_max, y_min, y_max, pixel_width, BASE_ITER)
    if final_zs is not None:
        new_img = colour_by_basin(final_zs, ns)
    else:
        new_img = fancy_colour(ns)
    im.set_data(new_img)
    im.set_extent([x_min, x_max, y_min, y_max])
    ax.set_title(f"{title_str} — done"); fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("scroll_event", on_release)
plt.show(block=True)