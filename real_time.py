import sys
import cv2
import numpy as np
import serial
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from flirpy.camera.boson import Boson

# # Attempt to open serial connection
# try:
#     ser = serial.Serial('COM3', baudrate=921600, timeout=1)  # Adjust 'COM3' as needed
#     # # Initialize FLIR Boson camera
#
#     camera_connected = True
# except serial.SerialException:
#     print("Warning: No camera detected. Using synthetic image.")
#     camera_connected = False


def detect_cameras(thermal_image, diff_param =2, diff_in_out_param = 1):

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


    difference_from_mean_127norm =  cv2.normalize(difference_from_mean, None, 0, 127, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_difference_127norm =  cv2.normalize(normalized_difference, None, 0, 127, cv2.NORM_MINMAX).astype(np.uint8)
    binary_image = np.where(normalized_difference_127norm + difference_from_mean_127norm > np.percentile(normalized_difference_127norm + difference_from_mean_127norm, 98), 255, 0).astype(np.uint8)


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

    # Define a structuring element (kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # You can use (3, 3), (5, 5), etc.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # You can use (3, 3), (5, 5), etc.
    # Apply opening
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_open)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlights = cv2.cvtColor(thermal_image_normalized, cv2.COLOR_GRAY2BGR)

    height, width = thermal_image_normalized.shape
    predicted_boxes = []
    means_pred = []
    maxs_pred = []
    bbox_and_max = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (cv2.contourArea(c) > 2 and cv2.contourArea(c) < 350):  # or w*h>5:
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
                            # cv2.rectangle(highlights, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            # predicted_boxes.append([x, y, w, h])


                            means_pred.append(thermal_image[y:y + h, x:x + w].mean())
                            maxs_pred.append(thermal_image[y:y + h, x:x + w].max())
                            bbox_and_max.append([thermal_image[y:y + h, x:x + w].max(), [x, y, w, h]])
                            big_100 = thermal_image[y + h // 2 - 50:y + h // 2 + 50, x + w // 2 - 50:x + w // 2 + 50].copy()
                            big_100[50-h//2:5+h//2, 5-w//2:50+w//2] = 0
                            inside_max = thermal_image[y:y + h, x:x + w].max()

                            if len(np.where((big_100> inside_max-0.1) & (big_100< inside_max+0.1))[0])<50:
                                cv2.rectangle(highlights, (x, y), (x + w, y + h), (127, 0, 127), 2)
                                cv2.putText(highlights,
                                            f"{len(np.where((big_100 > inside_max - 0.1) & (big_100 < inside_max + 0.1))[0])}",
                                            (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                                predicted_boxes.append([x, y, w, h])
    return highlights , predicted_boxes



class ThermalCameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FLIR Boson Thermal Camera")
        self.setGeometry(100, 100, 680, 600)

        # Layout
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 512)

        self.frame_counter = 0

        self.freeze_button = QPushButton("Freeze & Detect")
        self.freeze_button.setFixedSize(150, 50)
        self.freeze_button.clicked.connect(self.freeze_image)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setFixedSize(150, 50)
        self.continue_button.clicked.connect(self.resume_live_feed)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.freeze_button)
        button_layout.addWidget(self.continue_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Timer for continuous frame update
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        self.frozen_image = None
        try:
            # Attempt to open serial connection
            self.camera = Boson(port="COM4")
            self.camera_connected = True
        except:
            print("Warning: No camera detected. Using synthetic image.")
            self.camera_connected = False

    def generate_synthetic_frame(self, frame_counter):
        """ Generate a dynamic synthetic thermal-like image with flowing lines that move over time """
        img = np.zeros((512, 640), dtype=np.uint8)
        for i in range(512):
            img[i] = np.sin(np.linspace(0, np.pi * 3, 640) + (i / 20) + (frame_counter / 5)) * 127 + 128
        return abs(img)



    def capture_frame(self):
        """ Capture a frame from the FLIR Boson camera or generate synthetic data """
        if self.camera_connected:
            try:
                img_array = self.camera.grab()
                if img_array is None:
                    raise ValueError("No image data received")
                img_array = np.fliplr(img_array)
                img_array = np.flipud(img_array)
                img_normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception as e:
                print(f"Warning: {e}. Using synthetic image instead.")
        else:
            # Generate synthetic dynamic image with shifting waves
            img_normalized = self.generate_synthetic_frame(self.frame_counter)
            self.frame_counter += 1  # Increment the frame counter for continuous motion

        return img_normalized

    # def capture_frame(self):
    #     """ Capture a frame from the FLIR Boson camera or generate synthetic moving thermal data """
    #     if camera_connected:
    #         ser.write(b'CAPTURE_COMMAND')  # Replace with actual command
    #         data = ser.read(640 * 512 * 2)
    #
    #         if len(data) != 640 * 512 * 2:
    #             print("Error: Incomplete image data received")
    #             return None
    #
    #         img_array = np.frombuffer(data, dtype=np.uint16).reshape((512, 640))
    #         img_normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     else:
    #         # Generate synthetic dynamic image with shifting waves
    #         img_normalized = self.generate_synthetic_frame(self.frame_counter)
    #         self.frame_counter += 1  # Increment the frame counter for continuous motion
    #
    #     return img_normalized

    def update_frame(self):
        if self.frozen_image is None:
            frame = self.capture_frame()
            if frame is not None:
                self.display_image(frame)

    def freeze_image(self):
        self.frozen_image = self.capture_frame()
        if self.frozen_image is not None:
            processed_image, detections = detect_cameras(self.frozen_image)
            self.display_image(processed_image)
            print("Detections:", detections)

    def resume_live_feed(self):
        """ Resume real-time image display """
        self.frozen_image = None

    def display_image(self, image):
        """ Convert and display image in QLabel """
        height, width= image.shape[0], image.shape[1]
        bytes_per_line = width
        if len(image.shape) == 3:
            bytes_per_line *= 3
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThermalCameraApp()
    window.show()
    sys.exit(app.exec_())