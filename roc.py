from Tools.detection_tools import run_check_detections
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(fpr, all_recall):
    # plot curve
    plt.plot(fpr, all_recall)
    plt.scatter(fpr, all_recall, color='red', marker='o', label="Specific Points")  # Highlight points
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("Recall  - True Positive Rate")
    plt.title("ROC curve")
    plt.show()


def calc_auc(fpr, all_recall):
    # calc AUC
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    recall = np.array(all_recall)[sorted_indices]
    print(np.trapezoid(recall, fpr))
    return np.trapezoid(recall, fpr)


def main():
    all_recall = []
    all_precision = []
    all_false = []
    param=2
    param2 = 1
    for param in [0.5, 1, 1.5]: #np.arange(-2, 3, 0.5):
        print("param", param)
        recall , precision, f1, mean_false = run_check_detections(diff_in_out_param = param2,diff_param =param,
                                                                   disply_image_time=1)
        all_recall.append(recall)
        all_precision.append(precision)
        all_false.append(mean_false)
    fpr = all_false/max(all_false)
    print(list(zip(all_recall, all_false)))
    # plt.suptitle(perc)
    plot_curve(fpr, all_recall)
    auc = calc_auc(fpr, all_recall)




if __name__ == "__main__":
    main()