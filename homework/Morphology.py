import os
import numpy as np
import cv2


def morphology_operation(image, file_path, output_path):
    morphology_output_path = os.path.join(output_path, 'morphology')
    os.makedirs(morphology_output_path, exist_ok=True)
    morphology_output_path = os.path.join(morphology_output_path, os.path.basename(file_path))

    # 膨胀操作
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    dilation_output_path = os.path.join(morphology_output_path, 'dilation_' + os.path.basename(file_path))
    cv2.imwrite(dilation_output_path, dilation)

    # 腐蚀操作
    erosion = cv2.erode(image, kernel, iterations=1)
    erosion_output_path = os.path.join(morphology_output_path, 'erosion_' + os.path.basename(file_path))
    cv2.imwrite(erosion_output_path, erosion)
