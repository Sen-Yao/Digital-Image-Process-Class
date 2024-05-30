import cv2
import numpy as np
import os


def gaussian_highpass_filter(image, file_path, output_path):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform the Fourier transform
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Get the image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a Gaussian Highpass filter mask
    D0 = 30  # Cutoff frequency
    mask = np.ones((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 - np.exp(- (distance ** 2) / (2 * (D0 ** 2)))

    # Apply the mask to the shifted DFT
    fshift = dft_shift * mask

    # Perform the inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the image to 0-255 range
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    gaussian_output_path = os.path.join(output_path, 'gaussian_highpass')
    os.makedirs(gaussian_output_path, exist_ok=True)
    # Save the result
    result_path = os.path.join(gaussian_output_path, 'gaussian_highpass_' + os.path.basename(file_path))
    cv2.imwrite(result_path, img_back)