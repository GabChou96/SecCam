import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import cv2
import os
# Generate similar synthetic data.
# X_normal = np.random.randn(100, 2) * 0.05 + 0.5
# X_anomalies = np.array([[0.1, 0.9], [0.9, 0.1]])
# X = np.concatenate((X_normal, X_anomalies))
dir_path = r'D:\ThermalMotion\thermi\BosonCaptures\BosonCaptures\Experiment_10_3_2025\tiff\\'
# Load image in grayscale for edge detection
for fname in os.listdir(dir_path):
    X = cv2.imread(dir_path+ fname,cv2.IMREAD_UNCHANGED).flatten().reshape(-1, 1)
    X = np.fliplr(X)
    X = np.flipud(X)
    print(X.shape)

    # Fit the Isolation Forest model on normal data (or all data if anomalies are rare).
    rng = np.random.RandomState(42)
    clf = IsolationForest(contamination=0.1, random_state=rng)
    clf.fit(X)
    y_pred = clf.predict(X)
    print(y_pred)
    X = X.reshape(512, 640)
    y_pred = y_pred.reshape(512, 640)
    # cv2.imshow("Original Image", X)
    with_detec = X.copy()
    with_detec = with_detec/100 - 273
    with_detec = cv2.normalize(with_detec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(with_detec.shape)

    with_detec = cv2.cvtColor(with_detec, cv2.COLOR_GRAY2BGR)

    with_detec[y_pred == -1] = [0,0,255]
    cv2.imshow("Predicted Image", with_detec)
    cv2.waitKey(0)

    # Visualize: -1 indicates an anomaly; 1 indicates a normal point.
    # plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
    # plt.title("Isolation Forest: Anomaly Detection")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.show()

#     # Apply LOF: Here, n_neighbors is tuned based on your data distribution.
#     lof = LocalOutlierFactor(n_neighbors=20)
#     y_pred = lof.fit_predict(X)
#     print(len(y_pred))
#
#     X= X.reshape(512, 640)
#     y_pred = y_pred.reshape(512, 640)
#     # cv2.imshow("Original Image", X)
#     with_detec = X.copy()
#     with_detec = with_detec/100 - 273
#     with_detec = cv2.normalize(with_detec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     print(with_detec.shape)
#
#     with_detec = cv2.cvtColor(with_detec, cv2.COLOR_GRAY2BGR)
#     with_detec[y_pred == -1] = [0,0,255]
#     cv2.imshow("Predicted Image", with_detec)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(X.reshape(512, 640), cmap='gray')
# plt.show()
# plt.imshow(y_pred.reshape(512, 640), cmap='gray')
# plt.show()
# Outliers are labeled as -1.
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
# plt.title("Local Outlier Factor (LOF) for Anomaly Detection")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.show()
