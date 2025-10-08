# Image Processing Projects

This repository contains three independent image processing exercises implemented in Python.  
Each script demonstrates a fundamental technique commonly used in digital image enhancement.

---

## 1. `embedded_squares_eq.py`
**Topic:** Global and Local Histogram Equalization  
**Description:**  
Enhances image contrast using two methods:
- **Global Histogram Equalization**: redistributes pixel intensity values across the entire image.
- **Local Histogram Equalization**: improves local details using a sliding window technique.

**Usage:**  
Run the script, select an image, and view the comparison of original, global, and local equalized images.

---

## 2. `laplacian_sharpen.py`
**Topic:** Frequency-Domain Laplacian Sharpening  
**Description:**  
Sharpens an image by emphasizing high-frequency components using the Laplacian operator in the frequency domain.  
The script:
- Reads a grayscale image.
- Applies Laplacian sharpening with adjustable strength.
- Displays and saves the results for comparison.

**Usage:**  
Run the script and select an image file from the dialog window.

---

## 3. `error_diffusion_gray6.py`
**Topic:** Error Diffusion Quantization (Floyd–Steinberg)  
**Description:**  
Performs gray-level quantization to 6 levels (`[0, 51, 102, 153, 204, 255]`) using the **Floyd–Steinberg** error diffusion method.  
The algorithm diffuses quantization errors to neighboring pixels for smoother tonal transitions.

**Usage:**  
Run the script, enter the filename of a grayscale image, and the program will display and save the quantized result.

---

##  Requirements
- Python 3.x  
- `numpy`, `matplotlib`, `Pillow`, `imageio`, `tkinter`

Install dependencies:
```bash
pip install numpy matplotlib pillow imageio
