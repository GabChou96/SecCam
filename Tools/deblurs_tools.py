from numpy.fft import fft2, ifft2
import numpy as np
import cv2

def calc_d_p(f, T, P, v, Z, w):
    return f * T / P * (v / Z + w)

def wiener_filter(blurred, kernel, K):
    # Normalize the kernel
    kernel /= np.sum(kernel)
    # Perform Fourier transforms
    blurred_fft = fft2(blurred)
    kernel_fft = fft2(kernel, s=blurred.shape)
    # Apply the Wiener filter
    kernel_fft_conj = np.conj(kernel_fft)
    denominator = (np.abs(kernel_fft) ** 2 + K)
    wiener_filter = kernel_fft_conj / denominator
    deblurred_fft = blurred_fft * wiener_filter
    # Perform the inverse Fourier transform
    deblurred = ifft2(deblurred_fft)
    # Return the real part of the result
    return np.abs(deblurred)


def deblur_image(image, size, d_p, kernel_type='two_dim', K_value=0.005):
    if kernel_type == 'one_dim_full':
        # option 1: x is positive
        kernel = 1 / d_p * np.exp(-np.arange(1, size + 1) / d_p)
    elif kernel_type == 'one_dim_line':
        # option 2 : having 1 dim only
        kernel = 1 / d_p * np.exp(-np.arange(1, size + 1) / d_p)
        kernel[1:, :] = 0  # todo understand whats better
    elif kernel_type == 'two_dim':
        # Option 3: Create a 2D  kernel changing on x and y
        x = np.arange(size).reshape(size, 1)  # Column vector of shape (n, 1)
        y = np.arange(size).reshape(1, size)  # Row vector of shape (1, m)
        # Compute the matrix using broadcasting
        kernel = (10 / d_p ** 2) * np.exp(-x / d_p + y / (0.1 * d_p))
    elif kernel_type == 'both':
        # Option 3: Create a 2D  kernel changing on x and y
        x = np.arange(size).reshape(size, 1)  # Column vector of shape (n, 1)
        y = np.arange(size).reshape(1, size)  # Row vector of shape (1, m)
        # Compute the matrix using broadcasting
        kernel = (1 / d_p ) * np.exp(-x/d_p - y/d_p)
    else:
        print("Kernel type does not exist")
        return
    deblurred_image = wiener_filter(image, kernel, K_value)
    # convert img to temperature [°C]
    tmp = deblurred_image / 100 - 273
    mT = np.median(tmp)
    sT = np.std(tmp)
    vmin = mT - 2.5 * sT
    vmax = mT + 2.5 * sT
    # Clip the pixel values to the range [vmin, vmax]
    clipped_image = np.clip(tmp, vmin, vmax)
    # Normalize the image to the range [0, 255]
    normalized_image = ((clipped_image - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized_image


def viz_image(image):
    tmp = image / 100 - 273
    mT = np.median(tmp)
    sT = np.std(tmp)
    vmin = mT - 2.5 * sT
    vmax = mT + 2.5 * sT
    # Clip the pixel values to the range [vmin, vmax]
    clipped_image = np.clip(tmp, vmin, vmax)
    # Normalize the image to the range [0, 255]
    normalized_image = ((clipped_image - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized_image


def feature_orb(left_image, right_image):
    # Detect features using ORB
    orb = cv2.ORB_create(nfeatures=10000)
    orb2 = cv2.ORB_create(nfeatures=10000)
    keypoints_left, descriptors_left = orb.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = orb2.detectAndCompute(right_image, None)
    # Visualize keypoints
    left_with_keypoints = cv2.drawKeypoints(left_image, keypoints_left, None, color=(0, 255, 0))
    right_with_keypoints = cv2.drawKeypoints(right_image, keypoints_right, None, color=(0, 255, 0))
    # cv2.imshow("mage Keypoints", np.hstack([left_with_keypoints, right_with_keypoints]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return left_with_keypoints, right_with_keypoints, len(keypoints_left), len(keypoints_right)
