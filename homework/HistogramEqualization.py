import os
import cv2
from matplotlib import pyplot as plt


def histogram_equalization(input_image, file_path, output_path):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    histogram_output_path = os.path.join(output_path, 'histogram_equalization')
    histogram_output_path = os.path.join(histogram_output_path, os.path.basename(file_path))
    os.makedirs(histogram_output_path, exist_ok=True)

    # 保存原始灰度图像和均衡化后的灰度图像
    gray_image_path = os.path.join(histogram_output_path, 'gray_image' + os.path.basename(file_path))
    equalized_image_path = os.path.join(histogram_output_path, 'equalized_image' + os.path.basename(file_path))

    cv2.imwrite(gray_image_path, gray_image)
    cv2.imwrite(equalized_image_path, equalized_image)

    # 使用 matplotlib 绘制并保存直方图
    plt.figure(figsize=(12, 6))

    # 灰度图像的直方图
    plt.subplot(2, 2, 1)
    plt.title('Gray Image Histogram')
    plt.hist(gray_image.ravel(), 256, [0, 256], color='blue')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # 均衡化后灰度图像的直方图
    plt.subplot(2, 2, 2)
    plt.title('Equalized Gray Image Histogram')
    plt.hist(equalized_image.ravel(), 256, [0, 256], color='red')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # 显示原始彩色图像
    plt.subplot(2, 2, 3)
    plt.title('Original Color Image')
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 显示均衡化后的灰度图像
    plt.subplot(2, 2, 4)
    plt.title('Equalized Gray Image')
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    # 保存直方图
    histogram_path = os.path.join(histogram_output_path, 'histograms_' + os.path.basename(file_path))

    plt.tight_layout()
    plt.savefig(histogram_path)

    # 关闭绘图窗口
    plt.close()