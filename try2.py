import time

import cv2
import os
import numpy as np
# Callback function to capture mouse click
# def get_pixel(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
#         pixel_value = tmp[y, x]
#         print(f'Pixel at ({x},{y}): {pixel_value}')
#
# # Load the image
# image_dir = r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\'
# for fname in os.listdir(image_dir):
#     print(fname)
#     if fname.endswith('.tiff') :
#         image_path = os.path.join(image_dir, fname)
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         image = np.fliplr(image)
#         image = np.flipud(image)
#         tmp = image / 100 - 273
#         normalized = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         _, binary = cv2.threshold(tmp, 28, 255, cv2.THRESH_BINARY)
#         print(binary, binary.shape)
#         cv2.imshow("binary", binary)
#         # Create a window and set the mouse callback function
#         cv2.imshow('Image', normalized)
#         cv2.setMouseCallback('Image', get_pixel)
#
#         cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
# from scipy.ndimage import maximum_filter
#
# # Load the image (convert to grayscale for intensity-based maxima)
# image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture66.tiff', cv2.IMREAD_UNCHANGED)
# image = np.fliplr(image)
# image = np.flipud(image)
# tmp = image / 100 - 273
# # Apply a maximum filter to detect local peaks
# size = 5  # Window size for detecting peaks
# filtered = maximum_filter(tmp, size=size)
#
# # Identify local maxima (where the filtered result equals original image)
# local_maxima =  (tmp == filtered) &(tmp > np.percentile(tmp, 99))  # Thresholding to remove weak peaks
#
# # Mark local maxima points on the image
# normalized = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# output = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
# output[local_maxima] = [0, 0, 255]  # Red color for maxima points
# # output[filtered > 0] = [0, 255, 0]  # Green color for filtered points
# print(local_maxima)
# filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
# # Show results
# cv2.imshow('Local Maxima', output)
# cv2.imshow('Filtered Image', filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################
# image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture66.tiff', cv2.IMREAD_UNCHANGED)
# image = np.fliplr(image)
# image = np.flipud(image)
# tmp = image / 100 - 273
# normalized = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 1
# params.maxThreshold = 200
#
# # Filter by Area.
# params.filterByArea = False
# params.minArea = 3
#
# # Filter by Circularity
# params.filterByCircularity = False
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = False
# params.minConvexity = 0.87
#
# # Filter by Inertia
# params.filterByInertia = False
# params.minInertiaRatio = 0.01
# detector = cv2.SimpleBlobDetector_create(params)
# # Detect blobs.
# keypoints = detector.detect(normalized)
#
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(normalized, keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
###############################################

# import cv2
# import numpy as np
# dir_path = r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\'
# # Load image in grayscale for edge detection
# for fname in os.listdir(dir_path):
#     image = cv2.imread(dir_path+ fname,cv2.IMREAD_UNCHANGED)
#     image = np.fliplr(image)
#     image = np.flipud(image)
#     tmp = image / 100 - 273
#     image = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
#     # # Step 1: Enhance contrast (optional but often helpful)
#     # enhanced = cv2.equalizeHist(image)
#     #
#     # # Step 2: Apply Gaussian Blur to reduce noise
#     # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#
#     # Step 3: Apply the Laplacian operator
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#
#     # Step 4: Convert the result to 8-bit (for visualization)
#     laplacian_abs = cv2.convertScaleAbs(laplacian)
#     # laplacian_abs = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
#
#     # Optional Step 5: Thresholding to pick up the edges definitively
#     print(np.percentile(laplacian_abs, 99))
#     _, thresh = cv2.threshold(laplacian_abs, np.percentile(laplacian_abs, 99), 255, cv2.THRESH_BINARY)
#
#     # Display the processed images
#     cv2.imshow('Original', image)
#     cv2.imshow('Laplacian', laplacian_abs)
#     cv2.imshow('Thresholded edges', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 1
# params.maxThreshold = 200
# params.filterByArea = True
# params.minArea = 1
# params.filterByCircularity = False
# params.filterByConvexity = True
# params.filterByInertia = False
#
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(thresh)
# img_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imshow('Blob Detection', img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
# def detect_bright_spots(thermal_image):
#     """
#     Detects bright spots in a thermal image based on local temperature difference.
#
#     Args:
#         thermal_image (numpy.ndarray): The thermal image data (e.g., in Celsius).
#
#     Returns:
#         list: A list of coordinates (x, y) of detected bright spots.
#     """
#     bright_spots =[]
#     height, width = thermal_image.shape
#
#     for y in range(1, height - 1):
#         for x in range(1, width - 1):
#             center_temp = thermal_image[y, x]
#             neighborhood = thermal_image[y-1:y+2, x-1:x+2]
#             neighborhood_avg = np.mean(neighborhood)
#
#             if center_temp > neighborhood_avg + 3:
#                 # Further checks for size and roundness can be added here
#                 bright_spots.append((x, y))
#
#     return bright_spots
#
# # Example usage:
# # Assuming 'thermal_image.tiff' is your thermal image file
# # Make sure to load the temperature data correctly based on the file format
# thermal_data = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture66.tiff', cv2.IMREAD_ANYDEPTH)
# thermal_data = np.fliplr(thermal_data)
# thermal_data = np.flipud(thermal_data)
# # Convert to appropriate temperature scale if needed
# image = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# print(image.shape)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# cv2.imshow("origin", image)
# # cv2.waitKey(0)
# if thermal_data is not None:
#     detected_spots = detect_bright_spots(thermal_data)
#     print(len(detected_spots))
#     for y, x in detected_spots:
#         print(y, x)
#         image[x, y] = [0, 0, 255]
#     #     cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
#     # image[detected_spots] = [0,0,255]
#     cv2.imshow("Detected Bright Spots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Detected bright spots:", detected_spots)
# else:
#     print("Could not load the thermal image.")

#############################################

# from math import sqrt
# from skimage import data
# from skimage.feature import blob_dog, blob_log, blob_doh
# from skimage.color import rgb2gray, gray2rgb
# from skimage import io, draw, filters
# from scipy import ndimage, datasets
#
# import matplotlib.pyplot as plt
#
#
# dir_path = r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\'
# # Load image in grayscale for edge detection
# for fname in os.listdir(dir_path):
#     image =io.imread(dir_path+ fname)
#     print(dir_path+ fname)
#     image = np.fliplr(image)
#     image = np.flipud(image)
#     tmp = image / 100 - 273
#     image_gray = tmp
#     print(tmp)
#     result = ndimage.gaussian_laplace(image_gray, sigma = 2)
#
#     plt.imshow(result)
#     plt.show()
#     # Compute threshold using Otsu's method
#     # threshold_value = filters.threshold_otsu(image)
#
#     # Apply thresholding
#     binary_image = result > np.percentile(result, 90)
#     plt.imshow(binary_image)
#     plt.show()
    # # Define the kernel size (n*n area)
    # kernel_size = 15  # Adjust this to your desired area size
    #
    # # Calculate the local mean and standard deviation for each pixel
    # mean = cv2.blur(result.astype(np.float32), (kernel_size, kernel_size))
    # difference_from_mean = np.maximum(result.astype(np.float32) - mean, 0)
    # plt.imshow(difference_from_mean)
    # plt.show()



# blobs_log = blob_log(image_gray, max_sigma=5, num_sigma=10, threshold=5)
#
# # Compute radii in the 3rd column.
# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
#
# blobs_dog = blob_dog(image_gray, max_sigma=5, threshold=2)
# blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
#
# blobs_doh = blob_doh(image_gray, max_sigma=5, threshold=5)
#
# blobs_list = [blobs_log, blobs_dog, blobs_doh]
# print(blobs_list)
# colors = ['yellow', 'lime', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
# sequence = zip(blobs_list, colors, titles)
#
# fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()
#
# for idx, (blobs, color, title) in enumerate(sequence):
#     ax[idx].set_title(title)
#     ax[idx].imshow(image)
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#         ax[idx].add_patch(c)
#     ax[idx].set_axis_off()
#
# plt.tight_layout()
# plt.show()
# image = gray2rgb(image)
# for blob in blobs_log:
#     y, x, r = blob
#     rr, cc = draw.circle_perimeter(int(y), int(x), int(r))
#     image[rr, cc] = [255, 0, 0]  # Red color
# # plt.imshow(image)
# #
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.feature import peak_local_max
from skimage.morphology import label, area_opening, closing, square
from scipy import ndimage as ndi


def detect_local_maxima(image, min_size=3, max_size=200, min_distance=1, threshold_rel=0.2):
    """
    Detect areas that are local maxima in grayscale images with specific size constraints.

    Parameters:
    -----------
    image_path : str
        Path to the grayscale image
    min_size : int
        Minimum size of local maximum region in pixels
    max_size : int
        Maximum size of local maximum region in pixels
    min_distance : int
        Minimum distance between peaks in peak_local_max
    threshold_rel : float
        Minimum intensity of peaks relative to the image's max intensity

    Returns:
    --------
    labeled_maxima : ndarray
        Image with labeled regions that are local maxima
    num_regions : int
        Number of detected regions
    """
    # # Load and normalize the image
    # image = img_as_float(io.imread(image_path, as_gray=True))

    # Find local maxima coordinates
    coordinates = peak_local_max(image, min_distance=min_distance,
                                 threshold_rel=threshold_rel)

    # Create a mask of local maxima
    mask = np.zeros_like(image, dtype=bool)
    mask[tuple(coordinates.T)] = True

    # Label connected regions
    labeled_maxima, num_features = ndi.label(mask)

    # Create a component property dictionary
    props = {}
    for i in range(1, num_features + 1):
        props[i] = np.sum(labeled_maxima == i)

    # Filter regions by size
    filtered_labels = np.zeros_like(labeled_maxima)
    regions_kept = 0

    for label_id, size in props.items():
        if min_size <= size <= max_size:
            filtered_labels[labeled_maxima == label_id] = regions_kept + 1
            regions_kept += 1

    return filtered_labels, regions_kept


def visualize_results(image, labeled_maxima, num_regions):
    """
    Visualize the original image and detected local maxima regions.

    Parameters:
    -----------
    image_path : str
        Path to the original image
    labeled_maxima : ndarray
        Image with labeled regions that are local maxima
    num_regions : int
        Number of detected regions
    """
    # Load the original image
    # original = img_as_float(io.imread(image_path, as_gray=True))
    original = image
    # Create a visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Use a colormap that makes it easy to distinguish different regions
    axes[1].imshow(labeled_maxima > 0, cmap='viridis')
    axes[1].set_title(f'Local Maxima Regions ({num_regions} regions)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture76.tiff', cv2.IMREAD_ANYDEPTH)
    image = np.fliplr(image)
    image = np.flipud(image)
    # # Convert to appropriate temperature scale if needed
    # image = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Detect local maxima regions
    labeled_maxima, num_regions = detect_local_maxima(
        image,
        min_size=3,
        max_size=200,
        min_distance=1,
        threshold_rel=0.2
    )

    print(f"Found {num_regions} local maxima regions with size between 3-200 pixels")

    # Visualize the results
    visualize_results(image, labeled_maxima, num_regions)

    # To analyze the size distribution of detected regions
    unique_labels = np.unique(labeled_maxima)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)

    sizes = [np.sum(labeled_maxima == label_id) for label_id in unique_labels]

    plt.figure(figsize=(10, 5))
    plt.hist(sizes, bins=20)
    plt.xlabel('Region Size (pixels)')
    plt.ylabel('Count')
    plt.title('Size Distribution of Detected Local Maxima Regions')
    plt.show()