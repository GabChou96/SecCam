import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2
import os
import time

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

def calc_d_p(f, T, P, v, Z, w):
    return f * T / P * (v / Z + w)

def find_and_draw_edges(image):
    # Find edges using Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Draw edges on the original image (in color)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_image[edges != 0] = [0, 0, 255]  # Red color for edges

    # Display the original image and edges
    # cv2.imshow('Image with Edges', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return color_image

    # f = 18e-3 #TODO: find the value
    # T = 1/60 #TODO: find the value
    # P = 15e-6#TODO: find the value
    # v = 6 #TODO: find the value
    # Z = 6 #TODO: find the value
    # K_value = 0.005 #0.01
    # w=0

def deblur_image(image, size, d_p, kernel_type = 'two_dim', K_value= 0.005):
    if kernel_type == 'one_dim_full':
        # option 1: x is positive
        kernel = 1/d_p * np.exp(-np.arange(1,size+1)/d_p)
    elif kernel_type == 'one_dim_line':
        # option 2 : having 1 dim only
        kernel = 1/d_p * np.exp(-np.arange(1,size+1)/d_p)
        kernel[1:,:]=  0 #todo understand whats better
    elif kernel_type == 'two_dim':
        # Option 3: Create a 2D  kernel changing on x and y
        x = np.arange(size).reshape(size, 1)  # Column vector of shape (n, 1)
        y = np.arange(size).reshape(1, size)  # Row vector of shape (1, m)
        # Compute the matrix using broadcasting
        kernel = (10 / d_p**2) * np.exp(-x / d_p + y / (0.1 * d_p))
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



def deblur(image_path,num):

    # # Load the blurred thermal image
    # image_path = r'C:\Users\User\PycharmProjects\PythonProject\Deblur\img_0.png'  # Change this to the path of your image
    # image_path = r'C:\Users\User\Downloads\image1.png'
    # blurred_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    print(blurred_image.shape)
    start = time.time_ns()
    # flip image if required
    blurred_image = np.fliplr(blurred_image)
    blurred_image = np.flipud(blurred_image)
    f = 18e-3 #TODO: find the value
    T = 1/60 #TODO: find the value
    P = 15e-6#TODO: find the value
    v = 6 #TODO: find the value
    Z = 6 #TODO: find the value
    K_value = 0.005 #0.01

    w=0
    # Calculate the pixel displacement
    d_p = calc_d_p(f, T, P, v, Z, w)
    print("d_p: ", d_p)
    d_p = 5
    # Create a convolution kernel
    size = int(d_p + 2) #TODO: find the value
    # size = 20

    # K(x,y) = 1/d_p * math.exp(-x/d_p)
    # k(x,y) = 10/d_p**2 *np.exp(-(x/d_p + y/(0.1*d_p)))

    #option 1: x is positive
    # kernel = 1/d_p * np.exp(-np.arange(1,size+1)/d_p)


    # option 2: having 2 dim repeating lines
    # kernel = 1/d_p * np.exp(-np.linspace(-size,size, size)/d_p)
    # kernel = np.full((size, size), kernel)


    # Option 3: Create a 2D  kernel changing on x and y
    x = np.arange(size).reshape(size, 1)  # Column vector of shape (n, 1)
    y = np.arange(size).reshape(1, size)  # Row vector of shape (1, m)
    # Compute the matrix using broadcasting
    kernel = (10 / d_p**2) * np.exp(-x / d_p + y / (0.1 * d_p))

    # option 4: having 1 dim only
    # kernel[1:,:]=  0 #todo understand whats better


    # print(kernel)
    # Deblur the image
    deblurred_image = wiener_filter(blurred_image, kernel, K_value)
    print(deblurred_image.min(), deblurred_image.max())

    # Normalize the 16-bit image to the range [0, 1]
    #
    # print(normalized_image.min(), normalized_image.max())
    # Convert the normalized image to 8-bit format by scaling to the range [0, 255]
    # image_8bit = cv2.convertScaleAbs(normalized_image, alpha=(255.0))
    # Save and display the deblurred image
    # convert img to temperature [°C]
    tmp = deblurred_image / 100 - 273
    # normalized_image = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    # # calculate median and std values for image

    # norm_img = cv2.normalize(normalized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # norm_img = norm_img.astype(np.uint8)

    # cv2.imwrite(f'deblurred_image_{num}_check1.jpg',norm_img)
    # print(normalized_image, normalized_image.min(), normalized_image.max())
    mT = np.median(tmp)
    sT = np.std(tmp)
    vmin = mT - 2.5 * sT
    vmax = mT + 2.5 * sT
    # Clip the pixel values to the range [vmin, vmax]
    clipped_image = np.clip(tmp, vmin, vmax)
    # Normalize the image to the range [0, 255]
    normalized_image = ((clipped_image - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    print("time", (time.time_ns() - start)/1000000)

    tmp_b = blurred_image / 100 - 273
    clipped_blurred = np.clip(tmp_b, vmin, vmax)
    # Normalize the image to the range [0, 255]
    normalized_blurred = ((clipped_blurred - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    cv2.imshow("img", np.hstack([normalized_blurred, normalized_image]))
    cv2.waitKey(0)
    # cv2.imshow("img", np.hstack([find_and_draw_edges(normalized_blurred), find_and_draw_edges(normalized_image)]))
    # cv2.waitKey(1000)
    cv2.imwrite(f'image_{num}_movingcar.jpg', np.hstack([normalized_image, normalized_blurred]))

    cv2.destroyAllWindows()

    # Display the normalized image
    # cv2.imshow('Normalized Image', normalized_image)

    # plt.imshow(blurred_image, cmap='gray')
    # plt.show()
    #
    # plt.imshow(tmp, cmap='gray', vmin=mT - 2.5 * sT, vmax=mT + 2.5 * sT)
    # plt.show()






if __name__ == '__main__':
    # for num in range(9):
    # for num in range(40,41):
    i=0
    # for file in os.listdir(r'D:\ThermalMotion\\'):
    # filename = os.path.join(r'D:\ThermalMotion\\', file)
    # for filename in os.listdir(r'D:\thermi\\'):
    #     if filename.endswith(".tiff"):
    # filename = os.path.join(r'D:\thermi\\', filename)
    filename = r"F:\ThermalMotion\Boson_Capture_318.tiff"
    print(filename)
    num = filename.split('_')[-1].split('.')[0]
    deblur(filename,num)
    # deblur(r'D:\ThermalMotion\\Boson_Capture_318.tiff', 318)
