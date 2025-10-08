import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os

def map_to_grayscale_levels(pixel):
    # Map to closest of 6 levels: [0, 51, 102, 153, 204, 255]
    levels = np.array([0, 51, 102, 153, 204, 255])
    nearest = np.argmin(np.abs(levels - pixel))
    return levels[nearest]

def apply_error_diffusion(input_img):
    img = input_img.astype(np.float32)
    output = np.zeros_like(img)

    height, width = img.shape
    for y in range(height):
        for x in range(width):
            original = img[y, x]
            quantized = map_to_grayscale_levels(original)
            output[y, x] = quantized
            error = original - quantized

            # Floyd–Steinberg error diffusion
            if x + 1 < width:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < height and x - 1 >= 0:
                img[y + 1, x - 1] += error * 3 / 16
            if y + 1 < height:
                img[y + 1, x] += error * 5 / 16
            if y + 1 < height and x + 1 < width:
                img[y + 1, x + 1] += error * 1 / 16

    return output

def run():
    file_name = input("Enter grayscale image filename (e.g. input.png): ").strip()
    if not os.path.exists(file_name):
        print("File not found.")
        return

    original_img = imageio.imread(file_name, pilmode='L')
    quantized_img = apply_error_diffusion(original_img)

    # Display
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(quantized_img, cmap='gray')
    plt.title("Quantized")
    plt.axis('off')

    plt.show()

    # Save result
    imageio.imwrite("quantized_output.png", quantized_img.astype(np.uint8))
    print("Saved: quantized_output.png")

if __name__ == "__main__":
    run()
