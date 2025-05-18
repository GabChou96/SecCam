import cv2
import json
import numpy as np
from tkinter import Tk, filedialog
import os
# Initialize global variables
drawing = False
start_x, start_y = -1, -1
dir_path =r"C:\Users\User\Desktop\SecCamera_Thermal\\"

# Function to open file dialog and select an image
def select_image():
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(initialdir=dir_path ,filetypes=[("TIFF Images", "*.tif;*.tiff")])
    return file_path


# Mouse callback function
def draw_bounding_box(event, x, y, flags, param):
    global start_x, start_y, drawing, bounding_boxes, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bounding_boxes.append({"x": start_x, "y": start_y, "w": x - start_x, "h": y - start_y})
        cv2.rectangle(img_normalized, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("Bounding Box Editor", img_normalized)


# Function to save bounding boxes to a JSON file
def save_to_json(bounding_boxes, file_path="bounding_boxes"):
    with open(dir_path + r"jsons\\" + file_path + ".json", "w") as json_file:
        json.dump(bounding_boxes, json_file, indent=4)
    print(f"Bounding boxes saved to {dir_path + file_path}.json")


# Load the selected image
# image_path = select_image()
for image_path in os.listdir(dir_path):
    bounding_boxes = []
    if image_path.endswith(".tiff"):
        if image_path[:-5]+'.json' in os.listdir(dir_path + r"jsons\\"):
            continue
        print(f"Selected image: {image_path}")
        img = cv2.imread(dir_path + image_path, cv2.IMREAD_ANYDEPTH)

        if img is None:
            print("Error: Cannot load image.")
        else:
            img = np.fliplr(img)
            img = np.flipud(img)
            img = img / 100 - 273
            # Normalize the image to the range [0, 255]
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
            original = img_normalized.copy()
            cv2.namedWindow("Bounding Box Editor")
            cv2.setMouseCallback("Bounding Box Editor", draw_bounding_box)

            while True:
                cv2.imshow("Bounding Box Editor", img_normalized)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):  # Save bounding boxes
                    save_to_json(bounding_boxes, os.path.splitext(os.path.basename(image_path))[0]
    )
                elif key == ord('r'):  # Reset bounding boxes
                    print("cleared")
                    bounding_boxes.clear()
                    print(bounding_boxes)
                    img_normalized = original.copy()
                    cv2.imshow("Bounding Box Editor", img_normalized)
                    cv2.waitKey(1)
                elif key == ord('q'):  # Quit application
                    break

            cv2.destroyAllWindows()
    else:
        print("No image selected.")