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
from skimage.filters import threshold_otsu


def evaluate_metrics(ground_truths, predictions, iou_threshold=0.5):
    """
    Compute Precision, Recall, and F1 Score.
    ground_truths: List of ground-truth bounding boxes.
    predictions: List of predicted bounding boxes.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    matched_gt = set()

    for pred in predictions:
        best_iou = 0
        best_gt = None

        for gt in ground_truths:
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = tuple(gt)

        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(ground_truths) - tp  # Missed ground truths

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


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


import json
import cv2
import numpy as np


# Function to compute Intersection over Union (IoU)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea)
    return iou


# Load ground truth bounding boxes from JSON
def load_ground_truth(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)


def detect_cameras():

    return

dir_path = r"C:\Users\User\Desktop\SecCamera_Thermal\\"
flip= ['v', 'h']  # flip image vertically and horizontally
all_accs = []
all_falses = []
problem_names =[]
for fname in os.listdir(dir_path + r"jsons\\"):
# for fname in 'Boson_Capture119.json', 'Boson_Capture130.json':
    if fname.endswith('.json') :
        image_path = os.path.join(dir_path, fname[:-5]+".tiff")
        print(image_path)
        thermal_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        if 'h' in flip or 'H' in flip:
            thermal_image = np.fliplr(thermal_image)
        if 'v' in flip or 'V' in flip:
            thermal_image = np.flipud(thermal_image)
        thermal_image = thermal_image/100 - 273
        # Normalize the image to the range [0, 255]
        thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mod_thermal_image = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)

        # (Optional) Smooth the image to reduce noise
        # thermal_image = gaussian_filter(thermal_image, sigma=0.5)
        # deblur_img = deblur_image(thermal_image, 20, 10, kernel_type='two_dim', K_value=0.005)

        # Define the kernel size (n*n area)
        kernel_size = 30  # Adjust this to your desired area size

        # Calculate the local mean and standard deviation for each pixel
        mean = cv2.blur(thermal_image.astype(np.float32), (kernel_size, kernel_size))
        squared_img = cv2.blur(thermal_image.astype(np.float32) ** 2, (kernel_size, kernel_size))
        std = np.sqrt(squared_img - mean ** 2)

        # Calculate how far each pixel is from the local mean and standard deviation
        difference_from_mean = np.maximum(thermal_image.astype(np.float32) - mean, 0)
        normalized_difference = difference_from_mean / (std + 1e-7)  # Avoid divide-by-zero
        binary_image = np.where(normalized_difference >  np.percentile(normalized_difference, 99), 127, 0).astype(np.uint8)
        binary_image_diff = np.where(difference_from_mean > np.percentile(difference_from_mean, 98), 127, 0).astype(np.uint8)

        norm_diff_see = cv2.normalize(normalized_difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_see = cv2.normalize(difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("norm diff",np.hstack((norm_diff_see, binary_image)))
        cv2.imshow(" diff", np.hstack((diff_see, binary_image_diff)))
        # cv2.setMouseCallback("norm diff", get_pixel)
        binary_image = np.where(normalized_difference + difference_from_mean >  np.percentile(normalized_difference + difference_from_mean, 98), 255, 0).astype(np.uint8)

        # TODo check if this is better
        # local_maxima = feature.peak_local_max(normalized_difference+ difference_from_mean, min_distance=30)
        # both_img = cv2.normalize(normalized_difference+ difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # both_img = cv2.cvtColor(both_img, cv2.COLOR_GRAY2BGR)
        # for point in local_maxima:
        #     cv2.circle(both_img, (point[1], point[0]), radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.imshow("both",both_img)

        # calculate median and std values for image
        mT = np.median(thermal_image)
        sT = np.std(thermal_image)
        vmin = mT - 2.5 * sT
        vmax = mT + 2.5 * sT
        scaled_img = np.clip(thermal_image, vmin, vmax)
        scaled_img = cv2.normalize(scaled_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #
        # # Normalize to full 8-bit range (0-255)
        # scaled_img = ((scaled_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        cv2.imshow("Original Image1", scaled_img)
        cv2.imshow("Original Image", mod_thermal_image)
        cv2.setMouseCallback("Original Image", get_pixel)


        # blurred = cv2.GaussianBlur(thermal_image, (3, 3), 0)
        # # Apply Laplacian operator
        # laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        # # Convert the result to 8-bit for visualization
        # laplacian = cv2.convertScaleAbs(laplacian)
        # laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Display the images
        # cv2.imshow('LoG Result', np.hstack((laplacian, cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)[1])))
        # cv2.setMouseCallback("Original Image", get_pixel)
        # unique, counts = np.unique(difference_from_mean, return_counts=True)
        # print( unique, counts)
        # plt.scatter(unique[np.where(counts<10000)], counts[np.where(counts<10000)])
        # plt.show()

        # Define a structuring element (kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # You can use (3, 3), (5, 5), etc.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # You can use (3, 3), (5, 5), etc.
        # Apply opening
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_open)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
        cv2.imshow("close", np.hstack((opening, closing)))

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Apply connected components analysis
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening_check)
        # # Display labeled image
        # colored_labels = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imshow('Connected Components', opening_check)

        highlights = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)

        height, width = thermal_image_normalized.shape
        all_diff = []
        for c in contours:
            if cv2.contourArea(c)>3 and cv2.contourArea(c)<350:
                x, y, w, h = cv2.boundingRect(c)
                ilu = thermal_image[y:y + h, x:x + w].mean()
                big_rec = thermal_image[max(y - h, 0):min(y + 2*h, height),
                          max(x - w, 0):min(x + 2*w,  width)].mean()
                out_mean = (9*big_rec - ilu)/8
                all_diff.append(ilu - out_mean)
        predicted_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (cv2.contourArea(c)>3 and cv2.contourArea(c)<350): # or w*h>5:

                ratio = h / w if h > w else w / h
                # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # predicted_boxes.append([x, y, w, h])

                if ratio < 4: #w*h>9 and
                    ilu = thermal_image[y:y + h, x:x + w].mean()
                    big_rec = thermal_image[max(y - h, 0):min(y + 2 * h, height),
                              max(x - w, 0):min(x + 2 * w, width)].mean()
                    rec = thermal_image[max(y - h, 0):min(y + 2 * h, height),
                              max(x - w, 0):min(x + 2 * w, width)].copy()
                    rec[h:2*h, w:2*w] = 0
                    diff_in_out = thermal_image[y:y + h, x:x + w].max() - rec.max()
                    out_mean = (9*big_rec - ilu) / 8
                    # predicted_boxes.append([x, y, w, h])

                    #todo bring back
                    # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    if thermal_image[y:y + h, x:x + w].max()- out_mean>1:
                        # predicted_boxes.append([x, y, w, h])

                        # cv2.rectangle(highlights, (x, y), (x + w, y + h), (127, 127, 0), 2)
                        # cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.3, (0, 255, 0), 1)
                        # cv2.putText(highlights, f"{diff_in_out:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        #             (255, 0, 0), 1)

                        if diff_in_out >0  and  ((difference_from_mean[y:y + h, x:x + w]>2).sum()>1 or (normalized_difference[y:y + h, x:x + w]>2).sum()>1): #2.5: # todo: has to be according to image illumination
                            # print("area:", cv2.contourArea(c), "ratio:", w, h, "mean ilu", ilu, "mean all", thermal_image.mean(), f"{int(ilu)/int(thermal_image.mean()):.1f}")
                            # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                            # cv2.putText(highlights, f"{diff_in_out:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            #             0.3, (255, 0, 0), 1)
                            # predicted_boxes.append([x, y, w, h])

                            if diff_in_out >1: #ilu-out_mean> np.percentile(np.array(all_diff), 75): #sum(local_max[y:y + h, x:x + w].flatten())>0:
                                cv2.rectangle(highlights, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.putText(highlights,f"{diff_in_out:.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                                predicted_boxes.append([x, y, w, h])
                                # cv2.putText(highlights,f"{ilu- out_mean:.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        print("all diff", np.mean(np.array(all_diff)),np.median(np.array(all_diff)), np.percentile(np.array(all_diff), 50), np.percentile(np.array(all_diff), 75), len(all_diff) )

        ground_truth_boxes = load_ground_truth(fr"C:\Users\User\Desktop\SecCamera_Thermal\jsons\\{fname[:-5]}.json")
        print("read json ", f"{fname[:-5]}.json")
        # Compare predictions against ground truth
        good_detection = 0
        for gt in ground_truth_boxes:
            best_iou = 0
            for pred in predicted_boxes:
                iou = compute_iou(pred, [gt["x"], gt["y"], gt["w"], gt["h"]])
                best_iou = max(best_iou, iou)
            # print(f" Box {gt} - Best IoU: {best_iou:.2f}")
            if best_iou > 0.5:
                good_detection+=1
        if len(ground_truth_boxes):
            print(f"number of objects: {len(ground_truth_boxes)}, number of good predictions: {good_detection}, percentage: {good_detection/len(ground_truth_boxes):.2f}")
            all_accs.append(good_detection / len(ground_truth_boxes))
            if good_detection / len(ground_truth_boxes) <1:
                problem_names.append((fname,good_detection / len(ground_truth_boxes)) )
        else:
            print("no objects in the image")
            # all_accs.append(-1)
        print(f"number of false alarm {len(predicted_boxes) - good_detection}")

        all_falses.append(len(predicted_boxes) - good_detection)


        cv2.imshow("image",highlights)
        k = cv2.waitKey(10)
        if k == ord("k"):
            print([cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>2 and cv2.contourArea(c)<100])


print("all accuracy : ", all_accs)
print("final accuracy : ", np.mean(abs(np.array(all_accs))))
print("all false alarms : ", all_falses)
print("final false alarms : ", np.mean(np.array(all_falses)))
print(problem_names)
cv2.destroyAllWindows()