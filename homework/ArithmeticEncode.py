import numpy as np
import cv2
import os


def arithmetic_encode(image, file_path, output_path):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像的像素值范围
    unique_values, counts = np.unique(gray_image, return_counts=True)

    # 计算频率表
    total_pixels = gray_image.size
    frequencies = counts / total_pixels

    # 构建累积分布函数(CDF)
    cdf = np.cumsum(frequencies)

    # 初始化区间
    low = 0.0
    high = 1.0

    # 创建文件夹
    arithmetic_encode_output_path = os.path.join(output_path, 'coded_image')
    os.makedirs(arithmetic_encode_output_path, exist_ok=True)
    text_file_path = os.path.join(arithmetic_encode_output_path, os.path.splitext(os.path.basename(file_path))[0] + '_arithmetic_encode.txt')

    # 打开文件写入编码过程和结果
    with open(text_file_path, 'w') as f:
        f.write(f"Arithmetic Encoding Process for {os.path.basename(file_path)}\n\n")
        f.write(f"Pixel\tLow\tHigh\tRange\n")

        # 算术编码
        for pixel in gray_image.flatten():
            index = np.where(unique_values == pixel)[0][0]
            range_width = high - low
            high = low + range_width * cdf[index]
            low = low + range_width * cdf[index - 1] if index > 0 else low

            # 写入每一步的详细信息
            f.write(f"{pixel}\t{low:.6f}\t{high:.6f}\t{range_width:.6f}\n")

        # 写入最终的编码结果
        f.write(f"\nFinal encoded value: {low}")



