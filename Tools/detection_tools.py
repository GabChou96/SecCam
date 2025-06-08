import json
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter,  gaussian_filter
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb
from scipy.ndimage import median_filter
from skimage import io, exposure
from skimage import io, filters, feature
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max




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


def detect_cameras(thermal_image, diff_param =2, diff_in_out_param = 1):
    def get_pixel(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            pixel_value = thermal_image[y, x]
            print(f'Pixel at ({x},{y}): {pixel_value}')
        return
    # Normalize the image to the range [0, 255]
    thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mod_thermal_image = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)

    # Define the kernel size (n*n area)
    kernel_size = 30  # Adjust this to your desired area size

    # Calculate the local mean and standard deviation for each pixel
    mean = cv2.blur(thermal_image.astype(np.float32), (kernel_size, kernel_size))
    squared_img = cv2.blur(thermal_image.astype(np.float32) ** 2, (kernel_size, kernel_size))
    std = np.sqrt(squared_img - mean ** 2)

    # Calculate how far each pixel is from the local mean and standard deviation
    difference_from_mean = np.maximum(thermal_image.astype(np.float32) - mean, 0)
    normalized_difference = difference_from_mean / (std + 1e-7)  # Avoid divide-by-zero
    binary_image = np.where(normalized_difference > np.percentile(normalized_difference, 99), 127, 0).astype(np.uint8)
    binary_image_diff = np.where(difference_from_mean > np.percentile(difference_from_mean, 98), 127, 0).astype(
        np.uint8)

    norm_diff_see = cv2.normalize(normalized_difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_see = cv2.normalize(difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("norm diff", np.hstack((norm_diff_see, binary_image)))
    cv2.imshow(" diff", np.hstack((diff_see, binary_image_diff)))
    # cv2.setMouseCallback("norm diff", get_pixel)
    # binary_image = np.where(
    #     normalized_difference + difference_from_mean > np.percentile(normalized_difference + difference_from_mean, 98),
    #     255, 0).astype(np.uint8)

    # binary_image1 = np.where(((norm_diff_see > 1) ), 255, 0).astype(np.uint8)
    # binary_image2 = np.where(((diff_see > 0.5)), 255, 0).astype(np.uint8)
    # binary_image = np.where((binary_image1 & binary_image2), 255, 0).astype(np.uint8)


    difference_from_mean_127norm =  cv2.normalize(difference_from_mean, None, 0, 127, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_difference_127norm =  cv2.normalize(normalized_difference, None, 0, 127, cv2.NORM_MINMAX).astype(np.uint8)
    binary_image = np.where(normalized_difference_127norm + difference_from_mean_127norm > np.percentile(normalized_difference_127norm + difference_from_mean_127norm, 97), 255, 0).astype(np.uint8)
    cv2.imshow("sum images", np.hstack((normalized_difference_127norm + difference_from_mean_127norm, binary_image)))
    sum_both = normalized_difference_127norm + difference_from_mean_127norm

    # # Apply Watershed Algorithm
    # _, binary = cv2.threshold(sum_both, 150, 255, cv2.THRESH_BINARY)
    # markers = cv2.connectedComponents(binary)[1]
    # markers = cv2.watershed(cv2.cvtColor(sum_both, cv2.COLOR_GRAY2BGR), markers)
    # # Highlight peaks and lakes
    # sum_both = cv2.cvtColor(sum_both, cv2.COLOR_GRAY2BGR)
    # sum_both[markers == -1] = (0,0,255)  # High peaks (white)
    # sum_both[markers == 1] = (255,0,0)  # Low points (black)
    # cv2.imshow("Terrain Analysis", sum_both)


    # cv2.imwrite("norm_diff.png", norm_diff_see)
    # cv2.imwrite("diff.png", diff_see)
    # cv2.imwrite("sum_diffs.png", cv2.normalize(normalized_difference + difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # cv2.imwrite("binary_image.png", binary_image)


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
    # cv2.imwrite("OriginalImage.png", scaled_img)


    # Define a structuring element (kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # You can use (3, 3), (5, 5), etc.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # You can use (3, 3), (5, 5), etc.
    # Apply opening
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_open)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    cv2.imshow("close", np.hstack((opening, closing)))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlights = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)
    cnt_image = highlights.copy()
    cv2.drawContours(cnt_image, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("Contours.png", cnt_image)

    height, width = thermal_image_normalized.shape
    predicted_boxes = []
    means_pred = []
    maxs_pred = []
    bbox_and_max = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (cv2.contourArea(c) > 2 and cv2.contourArea(c) < 700):  # or w*h>5:
            ratio = h / w if h > w else w / h
            # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # predicted_boxes.append([x, y, w, h])

            if ratio < 4:  #3?:  # w*h>9 and
                ilu = thermal_image[y:y + h, x:x + w].mean()
                big_rec = thermal_image[max(y - h, 0):min(y + 2 * h, height),
                          max(x - w, 0):min(x + 2 * w, width)].mean()
                rec = thermal_image[max(y - h, 0):min(y + 2 * h, height),
                      max(x - w, 0):min(x + 2 * w, width)].copy()
                rec[h:2 * h, w:2 * w] = 0
                if len(np.where(rec <0)[0]) / rec.size >0.7:
                    continue
                diff_in_out = thermal_image[y:y + h, x:x + w].max() - rec.max()
                out_mean = (9 * big_rec - ilu) / 8
                # predicted_boxes.append([x, y, w, h])

                # # todo bring back
                # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.putText(highlights, f"{(thermal_image[y:y + h, x:x + w].max() - out_mean)/cv2.contourArea(c):.3f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                # cv2.putText(highlights, f"{cv2.contourArea(c)}, {w}, {h}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                if thermal_image[y:y + h, x:x + w].max() - out_mean > 1: # and  (thermal_image[y:y + h, x:x + w].max() - out_mean)/cv2.contourArea(c) > 0.1:
                    # predicted_boxes.append([x, y, w, h])
                    # cv2.rectangle(highlights, (x, y), (x + w, y + h), (127, 127, 0), 2)
                    # cv2.putText(highlights,f"{(thermal_image[y:y + h, x:x + w].max() - out_mean)/min(max(w, 1),max(h, 1)):.1f}",
                    #             (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    # cv2.putText(highlights, f"{diff_in_out:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255, 0, 0), 1)
                    if  (difference_from_mean[y:y + h, x:x + w] > 1).sum() > 2 or (normalized_difference[y:y + h,x:x + w] > 1).sum() > 2: # or (difference_from_mean[y:y + h, x:x + w] > 2).sum() > 0 or (normalized_difference[y:y + h,x:x + w] > 2).sum() > 0:
                        # todo: has to be according to image illumination  ## diff_in_out > 0
                            # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # cv2.putText(highlights, f"{diff_in_out:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255, 0, 0), 1)
                        # predicted_boxes.append([x, y, w, h])

                        # if diff_in_out > diff_in_out_param:  # sorted(all_diff, reverse=True)[5]: #diff_in_out_param:  # and ilu-out_mean> np.percentile(np.array(all_diff), 75): #sum(local_max[y:y + h, x:x + w].flatten())>0:
                        # (thermal_image[y:y + h, x:x + w].max() - out_mean)/min(max(w, 1),max(h, 1)) > 0.1 and
                        if  diff_in_out > diff_in_out_param:
                            cv2.rectangle(highlights, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            # predicted_boxes.append([x, y, w, h])


                            means_pred.append(thermal_image[y:y + h, x:x + w].mean())
                            maxs_pred.append(thermal_image[y:y + h, x:x + w].max())
                            bbox_and_max.append([thermal_image[y:y + h, x:x + w].max(), [x, y, w, h]])
                            big_100 = thermal_image[y + h // 2 - 50:y + h // 2 + 50, x + w // 2 - 50:x + w // 2 + 50].copy()
                            big_100[50-h//2:5+h//2, 5-w//2:50+w//2] = 0
                            inside_max = thermal_image[y:y + h, x:x + w].max()
                            cv2.putText(highlights, f"{len(np.where((big_100> inside_max-0.1) & (big_100< inside_max+0.1))[0])}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                            if len(np.where((big_100> inside_max-0.1) & (big_100< inside_max+0.1))[0])<50:
                                cv2.rectangle(highlights, (x, y), (x + w, y + h), (127, 0, 127), 2)
                                predicted_boxes.append([x, y, w, h])


                            # cv2.putText(highlights,f"{ilu- out_mean:.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    #       np.percentile(np.array(all_diff), 75), len(all_diff))
    # bbox_and_max = sorted(bbox_and_max, key=lambda x: x[0])
    # print(len(bbox_and_max))
    # for i in range(len(bbox_and_max)):
    #     # print(i, bbox_and_max[i][0], bbox_and_max[i][1])
    #     # print(i>0 and bbox_and_max[i][0] -bbox_and_max[i -1][0] <0.5,bbox_and_max[i][0] , bbox_and_max[i -1][0] )
    #     # print(i < len(bbox_and_max) - 2 and bbox_and_max[i + 1][0] - bbox_and_max[i][0] < 0.5, bbox_and_max[i + 1][0],  bbox_and_max[i][0])
    #     if ((i>0 and bbox_and_max[i][0] -bbox_and_max[i -1][0] <0.3) and  ) or  (i<len(bbox_and_max)-2 and bbox_and_max[i+1][0] -bbox_and_max[i][0] <0.3):
    #         continue
    #     else:
    #         cv2.rectangle(highlights, (bbox_and_max[i][1][0], bbox_and_max[i][1][1]), (bbox_and_max[i][1][0] + bbox_and_max[i][1][2], bbox_and_max[i][1][1] + bbox_and_max[i][1][3]), (127, 0, 127), 2)
    #         predicted_boxes.append([bbox_and_max[i][1][0], bbox_and_max[i][1][1], bbox_and_max[i][1][2], bbox_and_max[i][1][3]])
            # cv2.putText(highlights, f"{bbox_and_max[i][0]:.1f}", (bbox_and_max[i][1][0] - 10, bbox_and_max[i][1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    # for i in range(len(bbox_and_max)):
    #     out_i = True
    #     xi, yi, wi, hi = bbox_and_max[i][1]
    #     rec_i = thermal_image[max(yi - hi, 0):min(yi + 2 * hi, height),
    #           max(xi - wi, 0):min(xi + 2 * wi, width)].copy()
    #     rec_i[hi:2 * hi, wi:2 * wi] = 0
    #     diff_in_out_i = thermal_image[yi:yi + hi, xi:xi + wi].max() - rec_i.max()
    #     cv2.putText(highlights,
    #                 f"{bbox_and_max[i][0]:.1f}, {diff_in_out_i:.1f}",
    #                 (xi - 10, yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        # for j in range(len(bbox_and_max)):
        #     xj, yj, wj, hj = bbox_and_max[j][1]
        #     rec_j = thermal_image[max(yj - hj, 0):min(yj + 2 * hj, height),
        #             max(xj - wj, 0):min(xj + 2 * wj, width)].copy()
        #     rec_j[hj:2 * hj, wj:2 * wj] = 0
        #     diff_in_out_j = thermal_image[yj:yj + hj, xj:xj + wj].max() - rec_j.max()
        #     if i != j  and abs(bbox_and_max[i][0] - bbox_and_max[j][0]) < 0.2  and abs(diff_in_out_i - diff_in_out_j)  < 0.2:
        #         # print(i, j, bbox_and_max[i][0], bbox_and_max[j][0], diff_in_out_i, diff_in_out_j)
        #         out_i  = False
        #         break
        #
        # # print(i, out_i)
        # if out_i:
        #     cv2.rectangle(highlights, (xi, yi), (xi + wi, yi + hi), (127, 0, 127), 2)
        #     predicted_boxes.append([xi, yi, wi, hi])



    # plt.hist(means_pred, bins=100)
    # plt.title("Means")
    # plt.show()
    # plt.hist(sub_means_pred, bins=100)
    # plt.title("sub Means")
    # plt.show()
    # plt.hist(maxs_pred, bins=100)
    # plt.title("Maxs")
    # plt.show()
    return highlights , predicted_boxes


def run_check_detections(data_path, diff_in_out_param = 1, diff_param =2, display_image_time = 1):
    dir_path = r"C:\Users\User\Desktop\SecCamera_Thermal\\"
    flip = ['v', 'h']  # flip image vertically and horizontally
    problem_names = []
    all_tp = []
    all_fp = []
    all_fn = []
    for fname in data_path:
        if fname.endswith('.json'):
            image_path = os.path.join(dir_path, fname[:-5] + ".tiff")
            print(image_path)
            thermal_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            if 'h' in flip or 'H' in flip:
                thermal_image = np.fliplr(thermal_image)
            if 'v' in flip or 'V' in flip:
                thermal_image = np.flipud(thermal_image)
            thermal_image = thermal_image/100 - 273


    #######################################
            highlights , predicted_boxes = detect_cameras(thermal_image, diff_param, diff_in_out_param)
    #######################################

            ground_truth_boxes = load_ground_truth(fr"C:\Users\User\Desktop\SecCamera_Thermal\jsons\\{fname[:-5]}.json")
            # Compare predictions against ground truth
            tp = 0
            not_count = 0
            for gt in ground_truth_boxes:
                best_iou = 0
                for pred in predicted_boxes:
                    iou = compute_iou(pred, [gt["x"], gt["y"], gt["w"], gt["h"]])
                    if iou > 0.5 and gt["label"] == "ignore":
                        not_count += 1
                    best_iou = max(best_iou, iou)
                if best_iou > 0.5 and gt["label"] == "detection":
                    tp+=1


            # ground_truth_boxes[ground_truth_boxes[:]['label'] == "detection"]

            not_count = min(not_count,len(predicted_boxes) - tp)
            len_detections = len([obj for obj in ground_truth_boxes if obj["label"] == "detection"])

            if tp<len_detections:
                problem_names.append(fname)
            fp = len(predicted_boxes) - tp - not_count
            fn = len_detections - tp
            print(f"{fname} - TP: {tp}, FP: {fp}, FN: {fn}, Total Detections: {len(predicted_boxes)}, Ground Truth Detections: {len_detections}")
            all_fp.append(fp)
            all_tp.append(tp)
            all_fn.append(fn)

            cv2.imshow("image",highlights)
            cv2.waitKey(display_image_time)



    precision = sum(all_tp) / (sum(all_tp) + sum(all_fp)) if (sum(all_tp) + sum(all_fp)) > 0 else 0
    recall = sum(all_tp) / (sum(all_tp) + sum(all_fn)) if (sum(all_tp) + sum(all_fn)) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")
    print("mean false alarms : ", np.mean(np.array(all_fp)))
    print(problem_names)
    cv2.destroyAllWindows()
    return recall , precision, f1, np.mean(np.array(all_fp))
