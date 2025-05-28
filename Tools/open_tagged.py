import cv2
import json
import os
from tkinter import Tk, filedialog
import numpy as np
# ['Boson_Capture115.json', 'Boson_Capture118.json', 'Boson_Capture119.json', 'Boson_Capture122.json', 'Boson_Capture130.json', 'Boson_Capture93.json']

# ['Boson_Capture118.json', 'Boson_Capture119.json', 'Boson_Capture122.json', 'Boson_Capture130.json']
# Function to open file dialog and select an image
def select_image():
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.tif;*.tiff")])
    return file_path


# Function to load bounding boxes from the JSON file
def load_bounding_boxes(image_path):
    # Get the directory of the image
    image_dir = os.path.dirname(image_path)

    # Construct the JSON file path (inside the "jsons" folder)
    json_path = os.path.join(dir_path, image_dir, "jsons", os.path.splitext(os.path.basename(image_path))[0] + ".json")
    print(json_path)

    # Load bounding boxes if the file exists
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            return json.load(json_file)

    return []  # Return empty if no JSON file is found

dir_path =r"C:\Users\User\Desktop\SecCamera_Thermal\\"
colors = {'detection':(0,255,0), 'ignore':(0, 255, 255)}
# Select the image file
# image_path = select_image()
# for image_path in ['Boson_Capture119.json', 'Boson_Capture130.json', 'Boson_Capture168.json']:
# for image_path in ['Boson_Capture119.json', 'Boson_Capture130.json', 'Boson_Capture144.json', 'Boson_Capture149.json', 'Boson_Capture150.json', 'Boson_Capture168.json', 'Boson_Capture170.json']:
for image_path in os.listdir(dir_path + r"jsons\\"):
    if image_path:
        # Load image
        img = cv2.imread(dir_path + image_path[:-5]+".tiff", cv2.IMREAD_ANYDEPTH)
        # img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

        if img is None:
            print("Error: Cannot load image.")
        else:
            img = np.fliplr(img)
            img = np.flipud(img)
            img = img / 100 - 273
            # Normalize the image to the range [0, 255]
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
            # Load bounding boxes from the JSON folder
            bounding_boxes = load_bounding_boxes(image_path)
            cv2.imshow(" Image", img_normalized)
            # k = cv2.waitKey(0)
            # if k==ord('s'):
            #     cv2.imwrite(image_path[:-5]+".jpeg", img_normalized)

            # Draw bounding boxes on the image
            for box in bounding_boxes:
                try:
                    x, y, w, h, label = box["x"], box["y"], box["w"], box["h"], box['label']
                    cv2.rectangle(img_normalized, (x, y), (x + w, y + h),colors[label], 2)
                except:
                    print("no labels")
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    cv2.rectangle(img_normalized, (x, y), (x + w, y + h), colors['detection'], 2)

            # Show the image with bounding boxes
            cv2.imshow("Annotated Image", img_normalized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No image selected.")