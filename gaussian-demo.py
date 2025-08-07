#!/usr/bin/env python3
"""
gabor_demo.py
Demonstrates:
  1. 2-D Gaussian
  2. 2-D cosine (“stripe”) pattern
  3. Gaussian × cosine  →  Gabor filter (modulation)

Dependencies:  PyTorch  ≥ 1.13   •   Matplotlib  ≥ 3.6
"""

import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
N          = 800          # image width & height in pixels
GRID_EXT   = 4.0          # half-width of the coordinate grid (-GRID_EXT … +GRID_EXT)
SIGMA      = 1.0          # standard deviation of the Gaussian
FREQ_X     = 10.0         # horizontal spatial frequency (cycles across the width)
FREQ_Y     = 5.0          # vertical   spatial frequency (cycles down the height)
USE_COSINE = True         # False → use sine instead
CMAP       = "viridis"    # change to "gray" for BW visuals
# ------------------------------------------------------------------


def make_grid(n=N, ext=GRID_EXT):
    """Return mesh-grids X, Y spanning [-ext, +ext] × [-ext, +ext]."""
    coords = torch.linspace(-ext, ext, n)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")  # Y first for plotting
    return X, Y


def gaussian_2d(X, Y, sigma=SIGMA):
    """Isotropic 2-D Gaussian centred at (0,0)."""
    return torch.exp(-(X**2 + Y**2) / (2 * sigma**2))


def trig_2d(X, Y, fx=FREQ_X, fy=FREQ_Y, use_cosine=USE_COSINE):
    """2-D sine or cosine field: sin/cos(2π (fx·x + fy·y))."""
    phase = 2.0 * torch.pi * (fx * X + fy * Y)
    return torch.cos(phase) if use_cosine else torch.sin(phase)


def show(img, title):
    plt.figure(figsize=(4, 4), dpi=100)
    plt.imshow(img.numpy(), cmap=CMAP, origin="lower", extent=[0, N, 0, N], interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # 1. build coordinate grid
    X, Y = make_grid()

    # 2. compute images
    gauss  = gaussian_2d(X, Y)
    stripes = trig_2d(X, Y)
    gabor  = gauss * stripes  # modulation

    # 3. display
    show(gauss,   "2-D Gaussian")
    show(stripes, "2-D " + ("Cosine" if USE_COSINE else "Sine"))
    show(gabor,   "Gaussian × " + ("Cosine" if USE_COSINE else "Sine") + "  →  Gabor Filter")


if __name__ == "__main__":
    main()
