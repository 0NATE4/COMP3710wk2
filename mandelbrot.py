#!/usr/bin/env python3
"""
interactive_fractal_gpu.py
--------------------------
Live-recomputing Mandelbrot / Julia explorer using PyTorch + Matplotlib.

• Dynamically recomputes at full resolution after any zoom/pan.
• Capped iterations prevent freezing on deep zooms.
• Uses GPU if available.

Author : <your-name>
Course  : COMP/INFS xxxx
"""

# ------------------------------------------------------------------ #
# 0. Imports & config
# ------------------------------------------------------------------ #
import torch, numpy as np, matplotlib.pyplot as plt
plt.ion()                                   # interactive mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ------------------------------------------------------------------ #
# 1. Parameters you might tweak
# ------------------------------------------------------------------ #
# Use this variable to switch between fractals
FRACTAL  = "mandelbrot"           # "mandelbrot" or "julia"

# These variables are no longer needed, as resolution is now dynamic
# QUALTIY = 0.5
# MAX_WIDTH_PX = 800 * QUALTIY

JULIA_K  = torch.complex(torch.tensor(-0.8), torch.tensor(0.156))
BASE_ITER = 200                   # iterations for the initial view

# ------------------------------------------------------------------ #
# 2. Colour-mapper
# ------------------------------------------------------------------ #
def fancy_colour(a: np.ndarray) -> np.ndarray:
    cyc = (2*np.pi*a/20.0)[..., None]
    img = np.concatenate(
        [10+20*np.cos(cyc),
         30+50*np.sin(cyc),
         155-80*np.cos(cyc)],
        axis=2)
    img[a == a.max()] = 0
    return np.clip(img, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ #
# 3. Fractal computation helper
# ------------------------------------------------------------------ #
@torch.no_grad()
def compute_fractal(x_min, x_max, y_min, y_max, width_px, base_iter):
    spacing = (x_max - x_min) / width_px
    Y, X = np.mgrid[y_min:y_max:spacing, x_min:x_max:spacing]
    x = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(Y).to(torch.float32)

    # More efficient GPU handling: move tensors to device early
    c = torch.complex(x, y).to(device)
    z = torch.zeros_like(c).to(device)

    # This logic now correctly handles both cases
    if FRACTAL.lower() == "julia":
        z = c.clone()
        c = JULIA_K.to(device).expand_as(z)
    
    # z is already on the device, so zs is just a new name for it
    zs, ns = z, torch.zeros_like(z, dtype=torch.int32)

    # --- UPGRADE: CAPPED ITERATIONS ---
    # Prevents freezing on very deep zooms
    span = max(x_max - x_min, y_max - y_min)
    calculated_iter = int(base_iter * max(1, 3 / span))
    max_iter = min(calculated_iter, 5000)  # CAPPED to 5000
    # ----------------------------------
    
    for _ in range(max_iter):
        # For Mandelbrot, c is the grid; for Julia, c is the constant.
        # This single line works for both.
        zs = zs*zs + c
        bounded = torch.abs(zs) < 4.0
        ns += bounded
        if not bounded.any():
            break

    return fancy_colour(ns.cpu().numpy())

# ------------------------------------------------------------------ #
# 4. Initial render (full set)
# ------------------------------------------------------------------ #
INIT_BOUNDS = (-2.0, 1.0, -1.3, 1.3)
title_str = f"{FRACTAL.title()} Set — zoom/pan then release"

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(title_str)
ax.axis("off")
plt.tight_layout(pad=0)

# --- UPGRADE: DYNAMIC INITIAL RENDER ---
# We must draw the canvas once to get its true pixel size
fig.canvas.draw()
initial_pixel_width = int(ax.get_window_extent().width)
img = compute_fractal(*INIT_BOUNDS, initial_pixel_width, BASE_ITER)
# ---------------------------------------

im = ax.imshow(img, origin="lower", extent=INIT_BOUNDS)

# ------------------------------------------------------------------ #
# 5. Callback: recompute after each mouse-release
# ------------------------------------------------------------------ #
def on_release(event):
    if event.inaxes is not ax:
        return
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if np.isclose([x_min, x_max, y_min, y_max], im.get_extent()).all():
        return

    ax.set_title("Computing…")
    fig.canvas.draw_idle()

    # --- UPGRADE: DYNAMIC RESOLUTION ON ZOOM ---
    # Get the current width of the axes in display pixels
    pixel_width = int(ax.get_window_extent().width)
    # -------------------------------------------

    new_img = compute_fractal(x_min, x_max, y_min, y_max,
                              pixel_width, BASE_ITER)
    im.set_data(new_img)
    im.set_extent([x_min, x_max, y_min, y_max])
    ax.set_title(f"{FRACTAL.title()} Set — done")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("scroll_event", on_release)

plt.show(block=True)