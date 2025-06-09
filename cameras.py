from Tools.detection_tools import run_check_detections, detect_cameras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

flip = ['h', 'v']  # 'h' for horizontal flip, 'v' for vertical flip, can be combined like ['h', 'v'] for both flips
def main():
    dir_path = r"C:\Users\User\Desktop\SecCamera_Thermal\8.6\\"
    for image_path in os.listdir(dir_path):
        if image_path.endswith(".tiff"):
            print("Processing:", image_path)
            # Load the thermal image
            thermal_image = cv2.imread(dir_path + image_path, cv2.IMREAD_ANYDEPTH)
            if 'h' in flip or 'H' in flip:
                thermal_image = np.fliplr(thermal_image)
            if 'v' in flip or 'V' in flip:
                thermal_image = np.flipud(thermal_image)
            thermal_image = thermal_image/100 - 273
            # Run detection on the thermal image
            highlights , predicted_boxes = detect_cameras(thermal_image, diff_param =2, diff_in_out_param = 1)
            cv2.imshow("image", highlights)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()