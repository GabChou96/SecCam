import os
import time
import cv2
import numpy as np
from skimage import img_as_float
from skimage import io, filters, feature, exposure
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator



def get_pixel(event, x, y, flags, param):
    global clicked_x, clicked_y
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        pixel_value = image[y, x]
        print(f'Pixel at ({x},{y}): {pixel_value}')
        clicked_x, clicked_y = x, y


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


clicked_x, clicked_y = None, None
sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\User\PycharmProjects\PythonProject\Thermi\model\sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
dir_path = r"D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\"
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
        median_brightness = np.median(cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print(median_brightness)
        if median_brightness> 50:
            thermal_image = exposure.equalize_hist(thermal_image)

        thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mod_thermal_image = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)
        # Define the kernel size (n*n area)
        kernel_size = 20  # Adjust this to your desired area size

        # Calculate the local mean and standard deviation for each pixel
        mean = cv2.blur(thermal_image.astype(np.float32), (kernel_size, kernel_size))
        squared_img = cv2.blur(thermal_image.astype(np.float32) ** 2, (kernel_size, kernel_size))
        std = np.sqrt(squared_img - mean ** 2)

        difference_from_mean = np.maximum(thermal_image.astype(np.float32) - mean, 0)

        normalized_difference = difference_from_mean / (std + 1e-7)  # Avoid divide-by-zero
        binary_image = np.where(normalized_difference >  np.percentile(normalized_difference, 99), 127, 0).astype(np.uint8)
        binary_image_diff = np.where(difference_from_mean > np.percentile(difference_from_mean, 98), 127, 0).astype(np.uint8)

        norm_diff_see = cv2.normalize(normalized_difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_see = cv2.normalize(difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow("norm diff",np.hstack((norm_diff_see, binary_image)))
        # cv2.imshow(" diff", np.hstack((diff_see, binary_image_diff)))
        binary_image = np.where(normalized_difference + difference_from_mean >  np.percentile(normalized_difference + difference_from_mean, 99), 255, 0).astype(np.uint8)

        # TODo check if this is better
        local_maxima = feature.peak_local_max(normalized_difference+ difference_from_mean, min_distance=30)
        both_img = cv2.normalize(normalized_difference+ difference_from_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        both_img = cv2.cvtColor(both_img, cv2.COLOR_GRAY2BGR)
        for point in local_maxima:
            cv2.circle(both_img, (point[1], point[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow("both",both_img)

        cv2.imshow("Original Image", mod_thermal_image)
        cv2.setMouseCallback("Original Image", get_pixel)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # You can use (3, 3), (5, 5), etc.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # You can use (3, 3), (5, 5), etc.
        # Apply opening
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_open)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        highlights = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)

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
        image = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)
        image = np.array(image, dtype=np.uint8)
        predictor.set_image(image)
        for c in contours:
            if cv2.contourArea(c)>2 and cv2.contourArea(c)<500:
                x, y, w, h = cv2.boundingRect(c)
                ratio = h / w if h > w else w / h

                if ratio < 3: #w*h>9 and
                    ilu = original_thermal_image[y:y + h, x:x + w].mean()
                    big_rec = original_thermal_image[max(y - h, 0):min(y + 2 * h, hight),
                              max(x - w, 0):min(x + 2 * w, width)].mean()
                    out_mean = (9*big_rec - ilu) / 8
                    #todo bring back
                    # cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                    if ilu-out_mean> np.percentile(np.array(all_diff), 50): #2.5: # todo: has to be according to image illumination


                        point_coords = np.array([[x+ w//2, y+h//2]])
                        masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=np.array([1]))
                        mask = masks[np.argmax(scores)]
                        mask_image = show_mask(mask)
                        if sum(mask.flatten())<500:
                            cv2.imshow(f"mask_{x, y}", mask_image)
                            cv2.rectangle(highlights, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(highlights, f"{ilu - out_mean:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.3, (255, 0, 0), 1)

                        if (difference_from_mean[y:y + h, x:x + w]>1).sum()>1: #sum(local_max[y:y + h, x:x + w].flatten())>0:
                            cv2.rectangle(highlights, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(highlights,f"{ilu- out_mean:.1f}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        print("all diff", np.mean(np.array(all_diff)),np.median(np.array(all_diff)), np.percentile(np.array(all_diff), 50), np.percentile(np.array(all_diff), 75), len(all_diff) )


        cv2.imshow("image",highlights)
        cv2.imshow("close", np.hstack((opening, closing)))

        k = cv2.waitKey(0)
        if k == ord("k"):
            print([cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>2 and cv2.contourArea(c)<100])
