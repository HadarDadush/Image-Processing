# laplacian_sharpen_file_dialog.py
# Image sharpening in the frequency domain using Laplacian filter
# - Opens a file dialog to select an image
# - Saves original and sharpened images
# - Automatically opens a comparison window (before/after)

from pathlib import Path
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog
import os, subprocess, sys
import matplotlib.pyplot as plt

def read_gray(path: Path):
    """Read image as grayscale float64 in [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0

def save_gray(path: Path, arr: np.ndarray):
    """Save [0,1] float array as 8-bit PNG."""
    arr8 = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr8*255+0.5).astype(np.uint8)).save(path)

def laplacian_sharpen_freq(f: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Frequency domain Laplacian sharpening.
    g = f - k * normalized(Laplacian(f)).
    """
    P, Q = f.shape
    F = np.fft.fft2(f)

    # Frequency grids using fftfreq
    u = np.fft.fftfreq(Q) * Q
    v = np.fft.fftfreq(P) * P
    U, V = np.meshgrid(u, v)

    # Laplacian transfer function
    H = -4.0 * (np.pi ** 2) * (U ** 2 + V ** 2)

    # Apply filter and inverse FFT
    Lap = np.real(np.fft.ifft2(H * F))

    # Normalize Laplacian and sharpen
    den = np.max(np.abs(Lap)) + np.finfo(np.float64).eps
    lap_n = Lap / den
    g = f - k * lap_n

    return np.clip(g, 0.0, 1.0)

def open_file(path: Path):
    """Open file with default image viewer."""
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])

if __name__ == "__main__":
    
    # Open file dialog
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
    )
    if not file_path:
        raise SystemExit("No file selected.")

    in_path = Path(file_path)

    # Read and process
    f = read_gray(in_path)
    g = laplacian_sharpen_freq(f, k=1.0)

    # Save outputs
    orig_path = in_path.with_name(in_path.stem + "_orig.png")
    sharp_path = in_path.with_name(in_path.stem + "_sharp_lap_freq.png")
    save_gray(orig_path, f)
    save_gray(sharp_path, g)

    print("Saved files:")
    print(orig_path)
    print(sharp_path)

    # Show comparison window
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(g, cmap="gray")
    plt.title("Sharpened (Laplacian)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
