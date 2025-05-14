import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, filters, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.measure import regionprops
import cv2

def detect_local_maxima_watershed(image_path, min_size=3, max_size=200, sigma=1.0, h_min=0.1):
    """
    Detect areas that are local maxima in grayscale images using watershed segmentation.

    Parameters:
    -----------
    image_path : str
        Path to the grayscale image
    min_size : int
        Minimum size of local maximum region in pixels
    max_size : int
        Maximum size of local maximum region in pixels
    sigma : float
        Standard deviation for Gaussian filter (controls smoothing)
    h_min : float
        Minimum height between local maximum and background (0-1 range)

    Returns:
    --------
    watershed_labels : ndarray
        Image with labeled regions that are local maxima
    filtered_props : list
        Properties of detected regions
    """
    # Load and normalize the image
    image = img_as_float(io.imread(image_path, as_gray=True))

    # Apply Gaussian smoothing to reduce noise
    smoothed = filters.gaussian(image, sigma=sigma)

    # Invert the image because watershed finds basins, not peaks
    inverted = -smoothed

    # Find markers for watershed
    # h_min controls the minimum depth/height a basin must have
    markers = peak_local_max(
        -inverted,
        min_distance=3,
        threshold_abs=h_min,
        labels=None
    )

    # Label the markers
    # marker_labels, num_features = ndi.label(markers)
    # Convert markers to a labeled image
    marker_labels = np.zeros_like(inverted, dtype=np.int32)
    for i, coord in enumerate(markers):
        marker_labels[coord[0], coord[1]] = i + 1  # Assign unique labels

    # Perform watershed segmentation
    watershed_labels = watershed(inverted, marker_labels)

    # Get properties of each region
    props = regionprops(watershed_labels, intensity_image=image)

    # Filter regions by size and create new labeled image
    filtered_labels = np.zeros_like(watershed_labels)
    filtered_props = []

    for idx, prop in enumerate(props):
        if min_size <= prop.area <= max_size:
            filtered_labels[watershed_labels == prop.label] = len(filtered_props) + 1
            filtered_props.append(prop)

    return filtered_labels, filtered_props


def analyze_local_maxima(props, image):
    """
    Analyze the properties of detected local maxima regions.

    Parameters:
    -----------
    props : list
        List of region properties from regionprops
    image : ndarray
        Original image

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    results = {
        'count': len(props),
        'sizes': [prop.area for prop in props],
        'mean_intensities': [prop.mean_intensity for prop in props],
        'max_intensities': [np.max(image[prop.coords[:, 0], prop.coords[:, 1]]) for prop in props],
        'contrast': [
            np.max(image[prop.coords[:, 0], prop.coords[:, 1]]) -
            np.mean(image[
                        np.clip(prop.coords[:, 0] + dx, 0, image.shape[0] - 1),
                        np.clip(prop.coords[:, 1] + dy, 0, image.shape[1] - 1)
                    ])
            for prop in props
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
    }

    return results


def visualize_watershed_results(image_path, watershed_labels, props):
    """
    Visualize the original image, detected local maxima regions, and their properties.

    Parameters:
    -----------
    image_path : str
        Path to the original image
    watershed_labels : ndarray
        Image with labeled regions from watershed segmentation
    props : list
        Properties of detected regions
    """
    # Load the original image
    original = img_as_float(io.imread(image_path, as_gray=True))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Watershed regions
    axes[0, 1].imshow(watershed_labels > 0, cmap='nipy_spectral')
    axes[0, 1].set_title(f'Local Maxima Regions ({len(props)} regions)')
    axes[0, 1].axis('off')

    # Size distribution
    if props:
        sizes = [prop.area for prop in props]
        axes[1, 0].hist(sizes, bins=min(20, len(set(sizes))))
        axes[1, 0].set_xlabel('Region Size (pixels)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Size Distribution of Regions')

        # Intensity distribution
        intensities = [prop.mean_intensity for prop in props]
        axes[1, 1].hist(intensities, bins=min(20, len(set(intensities))))
        axes[1, 1].set_xlabel('Mean Intensity')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Intensity Distribution of Regions')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = r"D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture70.tiff"
    image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture66.tiff',cv2.IMREAD_UNCHANGED)
    image = np.fliplr(image)
    image = np.flipud(image)
    tmp = image / 100 - 273
    image = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Load the image for analysis
    # image = img_as_float(io.imread(image_path, as_gray=True))

    # Detect local maxima regions using watershed
    watershed_labels, props = detect_local_maxima_watershed(
        image_path,
        min_size=3,
        max_size=200,
        sigma=1.0,
        h_min=0.1
    )

    print(f"Found {len(props)} local maxima regions with size between 3-200 pixels")

    # Analyze the regions
    analysis = analyze_local_maxima(props, image)

    # Print some statistics
    if props:
        print(f"Average region size: {np.mean(analysis['sizes']):.2f} pixels")
        print(f"Average mean intensity: {np.mean(analysis['mean_intensities']):.4f}")

    # Visualize the results
    visualize_watershed_results(image_path, watershed_labels, props)













# import os
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# # Load the image
# image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture66.tiff', cv2.IMREAD_UNCHANGED)
# image = np.fliplr(image)
# image = np.flipud(image)
# tmp = image / 100 - 273
# normalized = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# print(tmp.min(), tmp.max(), tmp.mean(), tmp.std())
# # Reshape the image into a 2D array of pixels
# pixels = normalized.reshape((-1))  # Flatten image to list of RGB values
# pixels = np.float32(pixels)  # Convert to float for k-means
#
# # Define criteria and apply K-means clustering
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# num_clusters = 10  # Number of segmented regions
# _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
# # Convert back to image format
# centers = np.uint8(centers)  # Convert cluster centers to uint8
# segmented_image = centers[labels.flatten()].reshape(image.shape)  # Map pixels to cluster centers
# binary = np.where(labels.reshape(image.shape) == 2, 255, 0).astype(np.uint8)  # Convert to binary image
# print(segmented_image.min(), segmented_image.max(), segmented_image.mean(), segmented_image.std())
# # Show result
# cv2.imshow('binary', binary)
# cv2.imshow('Image', normalized)
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
#     # calculate median and std values for image
#     mT = np.median(tmp)
#     sT = np.std(tmp)
#
#     # plot thermal image
#     plt.figure()
#     plt.imshow(tmp, cmap='gray', vmin=mT - 2.5 * sT, vmax=mT + 2.5 * sT)
#     # coords=plt.ginput(3, 30, True)
#     # print(coords)
#     plt.title('Temp. ($\epsilon$ = 0.95) [°C] : ' + filename)
#     plt.show()
#
#     return img, tmp