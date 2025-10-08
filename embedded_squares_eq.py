import numpy as np
import matplotlib.pyplot as plt
from imageio import v2 as imageio
from tkinter import Tk, filedialog

# ------------- Helpers -------------

def to_uint8(img):
    """Ensure uint8 [0..255]."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    img = img - img.min()
    den = img.max() if img.max() > 0 else 1.0
    img = 255.0 * (img / den)
    return img.astype(np.uint8)

def rgb_to_gray(img):
    """Convert RGB/RGBA to grayscale luminance."""
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA -> RGB
            img = img[..., :3]
        if img.shape[2] == 3:
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(img.dtype)
    return img

def global_hist_eq(img_u8):
    """Global histogram equalization (8-bit)."""
    hist = np.bincount(img_u8.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)][0] if np.any(cdf) else 0
    denom = (cdf[-1] - cdf_min) if (cdf[-1] - cdf_min) != 0 else 1
    cdf_norm = (cdf - cdf_min) / denom
    lut = np.floor(255 * np.clip(cdf_norm, 0, 1)).astype(np.uint8)
    return lut[img_u8]

def local_hist_eq(img_u8, win_size=41, pad_mode='reflect'):
    """
    Local histogram equalization with a large sliding window.
    Larger windows (e.g., 31/41/51) yield smoother background while revealing embedded details.
    """
    assert win_size % 2 == 1 and win_size >= 3, "win_size must be odd >= 3"
    r = win_size // 2

    pad_img = np.pad(img_u8, r, mode=pad_mode)
    out = np.empty_like(img_u8)

    H, W = img_u8.shape
    for i in range(H):
        for j in range(W):
            wi0, wi1 = i, i + 2 * r + 1
            wj0, wj1 = j, j + 2 * r + 1
            patch = pad_img[wi0:wi1, wj0:wj1]

            # histogram & CDF inside the local patch
            hist = np.bincount(patch.ravel(), minlength=256)
            cdf = hist.cumsum()
            if cdf[-1] == 0:
                out[i, j] = pad_img[i + r, j + r]
                continue

            # Normalize CDF to [0,255]
            cdf_min = cdf[np.nonzero(cdf)][0]
            denom = (cdf[-1] - cdf_min) if (cdf[-1] - cdf_min) != 0 else 1
            cdf_norm = (cdf - cdf_min) / denom
            lut = np.floor(255 * np.clip(cdf_norm, 0, 1)).astype(np.uint8)

            center_val = pad_img[i + r, j + r]
            out[i, j] = lut[center_val]

    return out

# ------------- Main -------------

def main():
    # Hide root Tk window and open file chooser
    Tk().withdraw()
    img_path = filedialog.askopenfilename(
        title="Choose the embedded_squares image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
    )
    if not img_path:
        print("No file selected. Exiting.")
        return

    # Read and prepare image (grayscale uint8)
    img = imageio.imread(img_path)
    img = rgb_to_gray(img)
    img = to_uint8(img)

    # Global histogram equalization
    img_global = global_hist_eq(img)

    # Local histogram equalization with a large window for a cleaner result
    win_size = 41
    img_local = local_hist_eq(img, win_size=win_size, pad_mode='reflect')

    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray', interpolation='nearest'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(img_global, cmap='gray', interpolation='nearest'); plt.title('Global HE'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(img_local, cmap='gray', interpolation='nearest'); plt.title(f'Local HE (win={win_size}x{win_size})'); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
