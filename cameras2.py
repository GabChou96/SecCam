from Tools.detection_tools import run_check_detections
import numpy as np
import matplotlib.pyplot as plt

def main():
    run_check_detections(diff_in_out_param = 0.5, diff_param =2,  disply_image_time = 1)


if __name__ == "__main__":
    main()