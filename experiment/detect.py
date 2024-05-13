import cv2
import numpy as np


def find_peaks(gradient, threshold, interval_length):
    peaks = []
    for i in range(interval_length, len(gradient) - interval_length):
        window = gradient[i - interval_length:i + interval_length]
        if gradient[i] == max(window) and gradient[i] > threshold:
            peaks.append(i)
        elif gradient[i] == min(window) and gradient[i] < -threshold:
            peaks.append(i)
    return peaks


def segment_image(img, mask, interval_length, peak_threshold):
    # 确保img是灰度图
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用掩膜来限定区域
    img = cv2.bitwise_and(img, img, mask=mask)

    # 计算梯度
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # 寻找梯度峰值
    peaks_x = find_peaks(grad_x.flatten(), peak_threshold, interval_length)
    peaks_y = find_peaks(grad_y.flatten(), peak_threshold, interval_length)

    # 创建一个新的掩膜用于结果
    markers = np.zeros_like(img, dtype=np.int32)

    # 标记找到的峰值
    for i in peaks_x:
        markers[i % img.shape[0], i // img.shape[0]] = 255
    for i in peaks_y:
        markers[i % img.shape[0], i // img.shape[0]] = 255

    # 使用分水岭算法进行分割
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)

    # 将标记转换为掩膜
    mask_out = np.zeros_like(img, dtype=np.uint8)
    mask_out[markers == -1] = 255  # 分水岭边界为-1

    return mask_out