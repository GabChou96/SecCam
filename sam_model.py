# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)
import cv2
import numpy as np
import os
import time
import onnxruntime
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
image = cv2.imread(r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\Boson_Capture70.tiff' , cv2.IMREAD_ANYDEPTH)
image = np.fliplr(image)
image = np.flipud(image)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image, dtype=np.uint8)
predictor.set_image(image)


cv2.imshow("image", image)
cv2.setMouseCallback("image", get_pixel)


for i in range(100):
    # Perform your operations here
    time.sleep(1)  # Simulate some processing
    if clicked_x is not None and clicked_y is not None:
        point_coords = np.array([[clicked_x, clicked_y]])
        print(point_coords)
        masks,scores, logits = predictor.predict(point_coords=point_coords, point_labels=np.array([1]))
        mask = masks[np.argmax(scores)]
        mask_image = show_mask(mask)
        cv2.imshow("mask", mask_image)
        # Reset clicked coordinates
        clicked_x, clicked_y = None, None

    # Keep the image window responsive
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break


cv2.destroyAllWindows()


# sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\User\PycharmProjects\PythonProject\Thermi\model\sam_vit_h_4b8939.pth")
# dir_path = r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\\'
# onnx_model_path = "sam_onnx_example.onnx"
# ort_session = onnxruntime.InferenceSession(onnx_model_path)
# sam.to(device='cuda')
#
# mask_generator = SamAutomaticMaskGenerator(sam)
#
# # Load image in grayscale for edge detection
# for fname in os.listdir(dir_path):
#     if fname.endswith('.jpeg'):
#         image = cv2.imread(dir_path + fname , cv2.IMREAD_ANYDEPTH)  # Load as origin
#         # TODO tiff
#         # image = np.fliplr(image)
#         # image = np.flipud(image)
#         # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         print(image.shape)
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         image = np.array(image, dtype=np.uint8)
#         start = time.time()
#         masks = mask_generator.generate(image)
#         print("time", time.time()-start)
#         # print(masks)
#
#         show_img = image.copy()
#
#         for mask in masks:
#             x,y,w,h = mask['bbox']
#             cv2.rectangle(image,(x,y), (x+w, y+h),(255, 0, 0), 2 )
#         cv2.imshow("image", image)
#         cv2.imwrite(f"image_{fname[:-5]}.png", image)
#         cv2.waitKey(0)



