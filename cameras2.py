from Tools.detection_tools import run_check_detections
import numpy as np
import matplotlib.pyplot as plt
import os
def main():
    dir_path = r"C:\Users\User\Desktop\SecCamera_Thermal\8.6\\"
    data_path = os.listdir(dir_path + r"jsons\\")
    # data_path = ['Boson_Capture115.json', 'Boson_Capture116.json', 'Boson_Capture117.json', 'Boson_Capture118.json', 'Boson_Capture119.json', 'Boson_Capture120.json', 'Boson_Capture121.json', 'Boson_Capture130.json', 'Boson_Capture144.json', 'Boson_Capture150.json', 'Boson_Capture163.json', 'Boson_Capture168.json', 'Boson_Capture169.json', 'Boson_Capture86.json']
    run_check_detections(data_path, diff_in_out_param = 1, diff_param =2,  display_image_time =0)


if __name__ == "__main__":
    main()