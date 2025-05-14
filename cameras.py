import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter,  gaussian_filter
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb
from scipy.ndimage import median_filter
from skimage import io, exposure
from skimage import img_as_float
from skimage import io, filters, feature


def aggregate_close_contours(contours, max_distance=2):
    """
    Aggregates contours that are closer than `max_distance`.

    Args:
        contours (list): List of contours.
        max_distance (int): Maximum allowed distance to merge contours.

    Returns:
        list: New list of aggregated contours.
    """
    if not contours:
        return []
    merged_contours = []
    processed = np.zeros(len(contours), dtype=bool)

    for i, cnt1 in enumerate(contours):
        if processed[i]:
            continue
        merged = [cnt1]  # Initialize group with the current contour
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)

        for j, cnt2 in enumerate(contours):
            if i == j or processed[j]:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(cnt2)

            # Compute distance between bounding boxes
            dx = min(abs(x2 - (x1 + w1)), abs(x1 - (x2 + w2)))
            dy = min(abs(y2 - (y1 + h1)), abs(y1 - (y2 + h2)))

            if dx <= max_distance and dy <= max_distance:
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                processed[j] = True
        # Merge contours using convex hull or union
        merged_contours.append([new_x, new_y, new_w, new_h])  # Stack contours together
        processed[i] = True

    return merged_contours
# Callback function to capture mouse click
def get_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        pixel_value = thermal_image[y, x]
        print(f'Pixel at ({x},{y}): {pixel_value}')


# dir_path = r"D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\"
dir_path = r"C:\Users\User\Desktop\SecCamera_Thermal\\"
flip= ['v', 'h']  # flip image vertically and horizontally

for fname in os.listdir(dir_path):
    if fname.endswith('.tiff') :
        image_path = os.path.join(dir_path, fname)
        print(image_path)
        # thermal_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        thermal_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if 'h' in flip or 'H' in flip:
            thermal_image = np.fliplr(thermal_image)
        if 'v' in flip or 'V' in flip:
            thermal_image = np.flipud(thermal_image)
        thermal_image = thermal_image/100 - 273
        print(thermal_image.min(), thermal_image.max())
        original_thermal_image = thermal_image

        # Apply histogram equalization
        # TODO check which is better if any
        # median_brightness = np.median(cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        # print(median_brightness)
        # if median_brightness> 50:
        #     thermal_image = exposure.equalize_hist(thermal_image)
        # Apply CLAHE

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # thermal_image = clahe.apply(thermal_image_normalized)

        # Convert image to float and multiply to Increase contrast
        # image_float = img_as_float(thermal_image)
        # thermal_image = image_float * 1.5  # Scale factor of 1.5

        # Normalize the image to the range [0, 255]
        thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mod_thermal_image = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)
        print(thermal_image_normalized.shape)
        edges = cv2.Canny(thermal_image_normalized, threshold1=100, threshold2=200)
        cv2.imshow('sdges', edges)

        # (Optional) Smooth the image to reduce noise
        # thermal_image = gaussian_filter(thermal_image, sigma=0.5)

        # Find local maxima
        # neighborhood_size = 30  # 3x3 neighborhood
        # local_max = (img_smooth == maximum_filter(img_smooth, size=neighborhood_size))

        # Create a binary mask of the local maxima
        # coordinates = np.argwhere(local_max)
        # Display result
        # for y, x in coordinates:
        #     cv2.circle(mod_thermal_image, (x, y), 2, (0, 0, 255), -1)  # Draw maxima on the original image


        # deblur_img = deblur_image(thermal_image, 20, 10, kernel_type='two_dim', K_value=0.005)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhance = clahe.apply(thermal_image)
        # edge_output = cv2.Laplacian(thermal_image, cv2.CV_8U, ksize=1)

        # Define the kernel size (n*n area)
        kernel_size = 20  # Adjust this to your desired area size

        # Calculate the local mean and standard deviation for each pixel
        mean = cv2.blur(thermal_image.astype(np.float32), (kernel_size, kernel_size))
        squared_img = cv2.blur(thermal_image.astype(np.float32) ** 2, (kernel_size, kernel_size))
        std = np.sqrt(squared_img - mean ** 2)
        # start= time.time_ns()/1000
        # window_size = 30
        # median = median_filter(thermal_image, size=window_size)
        # print(time.time_ns()/1000-start)

        # Calculate how far each pixel is from the local mean and standard deviation
        difference_from_mean = np.maximum(thermal_image.astype(np.float32) - mean, 0)
        # difference_from_median = np.maximum(thermal_image.astype(np.float32) - median, 0)
        # median = cv2.normalize(median, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # mean = cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # print(difference_from_median)
        # cv2.imshow("median", np.hstack((median, mean)))

        # print("dif:", difference_from_mean, "std:", std)
        normalized_difference = difference_from_mean / (std + 1e-7)  # Avoid divide-by-zero
        binary_image = np.where(normalized_difference >  np.percentile(normalized_difference, 99), 127, 0).astype(np.uint8)
        binary_image_diff = np.where(difference_from_mean > np.percentile(difference_from_mean, 98), 127, 0).astype(np.uint8)

        norm_diff_see = cv2.normalize(normalized_difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_see = cv2.normalize(difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("norm diff",np.hstack((norm_diff_see, binary_image)))
        cv2.imshow(" diff", np.hstack((diff_see, binary_image_diff)))
        binary_image = np.where(normalized_difference + difference_from_mean >  np.percentile(normalized_difference + difference_from_mean, 99), 255, 0).astype(np.uint8)

        # TODo check if this is better
        local_maxima = feature.peak_local_max(normalized_difference+ difference_from_mean, min_distance=30)
        both_img = cv2.normalize(normalized_difference+ difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        both_img = cv2.cvtColor(both_img, cv2.COLOR_GRAY2BGR)
        for point in local_maxima:
            cv2.circle(both_img, (point[1], point[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("both",both_img)


        # binary_image_median = np.where(difference_from_median > np.percentile(difference_from_median, 95), 255, 0).astype(np.uint8)
        # local_max = np.where(local_max, 255, 0).astype(np.uint8)
        # both = np.where(local_max & binary_image==255, 255, 0).astype(np.uint8)
        # binary_image =np.where(difference_from_mean>np.percentile(difference_from_mean, 99), 255, 0).astype(np.uint8)
        # cv2.imshow("binary median", binary_image_median)
        # Display the results
        # mod_thermal_image[np.where(normalized_difference + difference_from_mean > np.max(normalized_difference + difference_from_mean)-1)] = [0, 0, 255]  # Mark detected points in red
        cv2.imshow("Original Image", mod_thermal_image)
        cv2.setMouseCallback("Original Image", get_pixel)
        # cv2.imshow("Difference from Mean", np.hstack((cv2.convertScaleAbs(difference_from_mean), cv2.convertScaleAbs(normalized_difference))))
        # cv2.imshow("binary", both)
        # print("norm:", normalized_difference.shape, normalized_difference)
        # plt.imshow(normalized_difference)
        # plt.show()
#         kernel = np.ones((10,10), np.float32) / 100
#         local_mean = cv2.filter2D(deblur_img, -1, kernel)
#         difference = cv2.absdiff(deblur_im , local_mean)
#         cv2.imshow("dif", difference)
#         cv2.waitKey(0)
#         _, threshold = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
        # Define a structuring element (kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # You can use (3, 3), (5, 5), etc.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # You can use (3, 3), (5, 5), etc.
        # Apply opening
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_open)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        highlights = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)
        # highlights = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2BGR)
        # new_cnts= aggregate_close_contours(contours, max_distance=2)
        # check = highlights.copy()
        # cv2.drawContours(check, contours, -1, (0, 0, 255), 2)
        # cv2.drawContours(check, new_cnts, -1, (255, 0, 0), 2)

        # cv2.imshow("contours", check)
        hight, width = thermal_image_normalized.shape
        all_diff = []
        for c in contours:
            if cv2.contourArea(c)>2 and cv2.contourArea(c)<500:
                x, y, w, h = cv2.boundingRect(c)
                ilu = original_thermal_image[y:y + h, x:x + w].mean()
                big_rec = original_thermal_image[max(y - h, 0):min(y + 2*h, hight),
                          max(x - w, 0):min(x + 2*w,  width)].mean()
                out_mean = (9*big_rec - ilu)/8
                all_diff.append(ilu - out_mean)

        for c in contours:
            if cv2.contourArea(c)>2 and cv2.contourArea(c)<500:
                x, y, w, h = cv2.boundingRect(c)
                ratio = h / w if h > w else w / h
                # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.imshow("mask", mask)
                # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if ratio < 3: #w*h>9 and
                    ilu = original_thermal_image[y:y + h, x:x + w].mean()
                    big_rec = original_thermal_image[max(y - h, 0):min(y + 2 * h, hight),
                              max(x - w, 0):min(x + 2 * w, width)].mean()
                    out_mean = (9*big_rec - ilu) / 8
                    # print('out_mean', out_mean, big_rec)
                    #todo bring back
                    # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                    # all_diff.append( ilu-out_mean)
                    # print(ilu-out_mean, np.percentile(np.array(all_diff), 75))
                    if ilu-out_mean> np.percentile(np.array(all_diff), 50): #2.5: # todo: has to be according to image illumination
                        # print("area:", cv2.contourArea(c), "ratio:", w, h, "mean ilu", ilu, "mean all", thermal_image.mean(), f"{int(ilu)/int(thermal_image.mean()):.1f}")
                        cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                        # print(x, y, w, h, normalized_difference[y:y + h, x:x + w])
                        # cv2.putText(highlights,f"{difference_from_mean[y:y + h, x:x + w].max():.1f}",
                        #             (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if (difference_from_mean[y:y + h, x:x + w]>1).sum()>1: #sum(local_max[y:y + h, x:x + w].flatten())>0:
                            cv2.rectangle(highlights, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(highlights,f"{ilu- out_mean:.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        print("all diff", np.mean(np.array(all_diff)),np.median(np.array(all_diff)), np.percentile(np.array(all_diff), 50), np.percentile(np.array(all_diff), 75), len(all_diff) )
                        # cv2.putText(highlights,f"{int(ilu)/int(thermal_image.mean()):.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # highlights[threshold == 255] = [0, 0, 255]

        cv2.imshow("image",highlights)
        # cv2.imwrite(f"highlights_{fname}", highlights)
        cv2.imshow("close", np.hstack((opening, closing)))
        k = cv2.waitKey(0)
        if k == ord("k"):
            print([cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>2 and cv2.contourArea(c)<100])
#         cv2.imshow("image2", threshold)
#         cv2.waitKey(0)
# cv2.destroyAllWindows()