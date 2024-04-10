import os
import cv2
import numpy as np


def find_bmp_files(directory):
    bmp_files = []
    # 遍历当前目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为'.bmp'
            if file.endswith('.bmp'):
                # 构建完整的文件路径并添加到列表中
                bmp_files.append(os.path.join(root, file))
    return bmp_files


def ROI_extract(pic_path):
    image = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = 1850, 3750, 7150, 3300  # 这里是左上角点的坐标和宽度、高度
    roi = image[y:y + h, x:x + w]
    cv2.imwrite(os.path.join('output', '1. roi_img.jpg'), roi)
    return roi


def pixel_texture_suppression(image):

    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)  # 将零频率移到频谱中心
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)  # 计算幅度谱
    # 将频谱结果保存为图像文件
    cv2.imwrite(os.path.join('output', '2. pixel_texture_suppression.jpg'), magnitude_spectrum)

    # 设计滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 20:crow + 20, 0:ccol - 300] = 0
    mask[crow - 20:crow + 20, ccol + 300:] = 0

    mask[0:crow - 200, ccol-20:ccol+20] = 0
    mask[crow + 200:, ccol-20:ccol+20] = 0

    mask[crow - 300:crow + 300, 300:600] = 0
    mask[crow - 300:crow + 300, 6550:6850] = 0
    mask[100:300, ccol - 300:ccol + 300] = 0
    mask[3000:3200, ccol - 300:ccol + 300] = 0
    mask[100:300, 300:600] = 0
    mask[3000:3200, 300:600] = 0
    mask[100:300, 6550:6850] = 0
    mask[3000:3200, 6550:6850] = 0

    mask[:, 1950:2050] = 0
    mask[:, 5100:5200] = 0
    mask[:, 400:450] = 0
    mask[:, 6725:6775] = 0
    mask[:, 1100:1150] = 0
    mask[:, 6000:6100] = 0

    mask[175:225, :] = 0
    mask[510:530, :] = 0
    mask[900:925, :] = 0
    mask[2350:2375, :] = 0
    mask[2750:2770, :] = 0
    mask[3050:3150, :] = 0

    # 应用滤波器
    f_shift_filtered = f_shift * mask
    magnitude_spectrum = 20 * np.log(np.abs(f_shift_filtered) + 1e-9)  # 计算幅度谱
    # 将结果转换为0到255之间的整数值
    # 将频谱结果保存为图像文件
    cv2.imwrite(os.path.join('output', '3. pixel_texture_suppression_filtered.jpg'), magnitude_spectrum)

    # 逆向傅立叶变换
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 转换结果为0到255的整数
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join('output', '4. img_back_normalized.jpg'), img_back_normalized)
    return img_back_normalized


def contrast(image):
    # 简单线性变换增强对比度
    alpha = 10  # 控制对比度
    beta = -500  # 控制亮度
    contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imwrite(os.path.join('output', '5. contrasted_image.jpg'), contrasted_image)
    return contrasted_image

def adjust_gamma(image, gamma=1):
    # 构建映射表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    cv2.imwrite(os.path.join('output', '5. contrasted_image.jpg'), cv2.LUT(image, table))

    # 应用伽马校正使用查找表
    return cv2.LUT(image, table)

def threshold(image):
    ret, thresholded_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join('output', '6. threshold_image.jpg'), thresholded_image)
    return thresholded_image

def opening(image):
    # 创建一个核，用于形态学操作，大小和形状可以根据需要调整
    kernel = np.ones((5, 5), np.uint8)

    # 应用开运算，去除小的噪点
    opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join('output', '7. opening_image.jpg'), opening_image)

def main():
    bmp_files = find_bmp_files('data')
    roi_img = ROI_extract(bmp_files[0])
    pts = pixel_texture_suppression(roi_img)
    contrast_img = adjust_gamma(pts)
    threshold_img = threshold(contrast_img)
    opening_img = opening(threshold_img)

if __name__ == "__main__":
    main()