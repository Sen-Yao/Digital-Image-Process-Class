import os
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


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


def extract_screen_roi(image_path, padding=5):
    # 加载图像
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (7096, 5320))
    cv2.imwrite(os.path.join('output', '0.0 resized_image.jpg'), resized_image)
    # 转换为灰度图像
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join('output', '0. gray.jpg'), gray)

    # 阈值处理，将灰度图像二值化
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width, height = rect[1][1], rect[1][0]

    src_pts = box.astype("float32")
    dst_pts = np.array([
        [0 + padding, 0 + padding],
        [width - 1 - padding, 0 + padding],
        [width - 1 - padding, height - 1 - padding],
        [0 + padding, height - 1 - padding]], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换
    warped = cv2.warpPerspective(resized_image, M, (int(width), int(height)))
    height, width = warped.shape[:2]
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join('output', '1. roi.jpg'), warped)
    return warped

def correct_illumination(image):
    # 分离颜色通道
    blue, green, red = cv2.split(image)

    # 对每个通道应用光照拟合
    corrected_blue = fit_illumination_model(blue, 'blue')
    corrected_green = fit_illumination_model(green, 'green')
    corrected_red = fit_illumination_model(red, 'red')

    # 合并通道
    corrected_image = cv2.merge([corrected_blue, corrected_green, corrected_red])
    cv2.imwrite(os.path.join('output', '3. corrected_image.jpg'), corrected_image)
    return corrected_image

def fit_illumination_model(channel, channel_name):
    rows, cols = channel.shape
    # 生成坐标网格
    y, x = np.indices((rows, cols))
    # 展平图像和坐标
    x = x.ravel()
    y = y.ravel()
    z = channel.ravel()

    # 构建矩阵A，包含x, y, xy, x^2, y^2项
    A = np.c_[x, y, x * y, x ** 2, y ** 2, np.ones_like(x)]
    # 使用最小二乘法拟合光照模型
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # 使用拟合的模型重建光照分布
    fitted = coeffs[0] * x + coeffs[1] * y + coeffs[2] * x * y + coeffs[3] * x ** 2 + coeffs[4] * y ** 2 + coeffs[5]
    fitted_image = fitted.reshape(rows, cols)

    # 从原图中减去拟合的背景模型
    corrected_image = channel - fitted_image + np.mean(fitted_image)
    cv2.imwrite(os.path.join('output', '3. enhanced_image.jpg'), corrected_image)
    return np.clip(corrected_image, 0, 255).astype(np.uint8)


def pixel_texture_suppression(image, sigma):
    # 分离彩色图像的三个通道
    b, g, r = cv2.split(image)

    # 对每个通道应用傅里叶变换并进行处理
    def process_channel(channel, sigma):
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)  # 将零频率移到频谱中心
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)  # 计算幅度谱

        # 将频谱结果保存为图像文件
        cv2.imwrite(os.path.join('output', '1.1 pixel_texture_suppression.jpg'), magnitude_spectrum)

        # 设计高斯低通滤波器
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        f_shift[crow, ccol] *= 0.3

        # 创建高斯掩模，中心为1，边缘向0平滑过渡
        x = np.linspace(-ccol, ccol, cols)
        y = np.linspace(-crow, crow, rows)
        x, y = np.meshgrid(x, y)
        gauss_mask = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

        # 应用滤波器
        f_shift_filtered = f_shift * gauss_mask
        magnitude_spectrum_filtered = 20 * np.log(np.abs(f_shift_filtered) + 1e-9)  # 计算幅度谱

        # 将频谱结果保存为图像文件
        cv2.imwrite(os.path.join('output', '1.2 pixel_texture_suppression_filtered.jpg'), magnitude_spectrum_filtered)

        # 逆向傅里叶变换
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 转换结果为0到255的整数
        img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        return img_back_normalized

    # 分别处理每个通道
    b_processed = process_channel(b, sigma)
    g_processed = process_channel(g, sigma)
    r_processed = process_channel(r, sigma)

    # 合并处理后的通道
    result = cv2.merge([b_processed, g_processed, r_processed])
    cv2.imwrite(os.path.join('output', '2. img_back_normalized.jpg'), result)
    return result


def contrast(image):
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(100, 100))
    enhanced_image = clahe.apply(image)
    cv2.imwrite(os.path.join('output', '5.2 contrasted_image.jpg'), enhanced_image)
    # enhanced_image = clahe.apply(image)
    # enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=3, beta=-500)
    # enhanced_image = cv2.equalizeHist(enhanced_image)
    # cv2.imwrite(os.path.join('output', '5.3 re-enhanced_image.jpg'), enhanced_image)
    return enhanced_image


def illumination_correction(image):
    # 使用高斯模糊来估计图像的背景
    blurred = cv2.GaussianBlur(image, (151, 151), 0, borderType=cv2.BORDER_REFLECT)
    # 从原始图像中减去背景
    corrected_image = cv2.addWeighted(image, 1, blurred, -1, 128)  # 增加128是为了避免负数的产生
    cv2.imwrite(os.path.join('output', '2. illumination_correction.jpg'), corrected_image)
    return corrected_image


def find_screen_mask(img, threshold=50, epoch=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_not(thresh)
    # 查找连通区域（轮廓）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的连通区域（假设最大的区域是边框）
    max_contour = max(contours, key=cv2.contourArea)

    # 创建一个mask，初值为1（黑色）
    mask = np.ones_like(img) * 0

    # 在mask中填充最大的连通区域，内部为0（白色）
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=epoch)
    # 将原图中mask对应白色部分（非最大连通区域）设为黑色
    return mask


def sobel(img, mask=None, kenel=9, threshold=80):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kenel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kenel)

    # 计算梯度的大小
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 应用阈值
    _, edge_image = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    if mask is not None:
        edge_image = cv2.bitwise_and(edge_image, edge_image, mask=mask)
    return edge_image


def canny_edge_detection(img, mask=None, low_threshold=50, high_threshold=150):
    # 使用Canny算子进行边缘检测
    edge_image = cv2.Canny(img, low_threshold, high_threshold)

    # 如果提供了掩膜，应用掩膜以限制边缘检测的区域
    if mask is not None:
        edge_image = cv2.bitwise_and(edge_image, edge_image, mask=mask)

    return edge_image


def average_gray_value(image, mask):
    # 应用掩膜到图像上
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # 计算掩膜内的灰度值总和
    total_gray_value = np.sum(masked_image)

    # 统计掩膜内的像素数量
    pixel_count = np.count_nonzero(mask)

    # 计算平均灰度值
    if pixel_count != 0:
        average_gray = total_gray_value / pixel_count
    else:
        average_gray = 0

    return average_gray


def watershed_segmentation(gray_image, mask):
    # 将灰度图像转换为彩色图像
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 创建一个与原始图像大小相同的标记图像
    markers = np.zeros_like(gray_image, dtype=np.int32)

    # 对掩膜mask中为255的区域进行标记
    markers[mask == 255] = 255
    gray_image[mask == 0] = 255

    avg = average_gray_value(gray_image, mask)

    # 对图像进行阈值化处理
    ret, thresh = cv2.threshold(gray_image, avg, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 对阈值化图像进行距离变换
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)

    # 标记分水岭算法的种子点
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 执行分水岭算法
    sure_fg = np.uint8(sure_fg)
    markers = np.uint8(markers)  # 将类型转换为CV_8U
    unknown = cv2.subtract(sure_fg, markers)
    ret, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_image, markers)
    color_image[markers == -1] = [255, 0, 0]  # 标记未知区域

    return color_image


def opening(image):
    # 创建一个核，用于形态学操作，大小和形状可以根据需要调整
    kernel = np.ones((25, 25), np.uint8)

    # 应用开运算，去除小的噪点
    opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join('output', '7. opening_image.jpg'), opening_image)


def close(img, kernel_size_1=9, kernel_size_2=5):
    # 定义闭操作的内核大小
    kernel_size = (kernel_size_1, kernel_size_2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # 执行膨胀操作
    dilated_image = cv2.dilate(img, kernel)
    # 执行腐蚀操作
    eroded_image = cv2.erode(dilated_image, kernel)

    return eroded_image


def filter_small_white_regions(image, min_area_threshold):
    # 二值化图像
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_image = np.uint8(binary_image)
    # 进行连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 创建一个新的图像，用于存储过滤后的结果
    filtered_image = np.zeros_like(image)

    # 遍历所有连通区域
    for label in range(1, num_labels):
        # 计算连通区域的面积
        area = stats[label, cv2.CC_STAT_AREA]

        # 如果连通区域面积大于阈值，则保留该区域
        if area >= min_area_threshold:
            filtered_image[labels == label] = 255

    return filtered_image


def open(img, kernel_size_1=9, kernel_size_2=5):
    kernel_size = (kernel_size_1, kernel_size_2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    eroded_image = cv2.erode(img, kernel)
    dilated_image = cv2.dilate(eroded_image, kernel)
    return dilated_image


def highlight_defects(original_image, defect_image):
    # 转换成灰度图像（如果defect_image不是灰度图）
    if len(defect_image.shape) > 2:
        defect_image = cv2.cvtColor(defect_image, cv2.COLOR_BGR2GRAY)

    # 应用阈值以确保图像是二值的
    _, binary_image = cv2.threshold(defect_image, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上画出轮廓
    highlighted_image = original_image.copy()
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # 红色轮廓

    # 创建一个新图像用于展示“before”和“after”
    result_height = original_image.shape[0] * 2 + 10  # 加10像素间隔
    result_image = np.zeros((result_height, original_image.shape[1], 3), dtype=np.uint8)

    # 将原始图像和高亮图像放在一起
    result_image[:original_image.shape[0], :] = original_image
    result_image[original_image.shape[0] + 10:, :] = highlighted_image

    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_image, 'Before', (50, 30), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result_image, 'After', (50, original_image.shape[0] + 60), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    # 保存结果图像
    return result_image


def segment_image_by_edges_and_tolerance(original_img, input_mask, ksize=3, edge_threshold=10, gray_tolerance=5,
                                         min_area=100, proximity=50):
    inpainted_img = cv2.inpaint(original_img, cv2.bitwise_not(input_mask), 3, cv2.INPAINT_TELEA)
    colored = is_color_image(inpainted_img)
    print('color:', colored)
    sobel_x = cv2.Sobel(inpainted_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(inpainted_img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_x = cv2.convertScaleAbs(sobel_x)  # 转换为8位图像
    sobel_y = cv2.convertScaleAbs(sobel_y)  # 转换为8位图像

    # 合并梯度（获得更全面的边缘信息）
    sobel_img = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    # 将 Sobel 图像转换为二值图像，标记出强烈的边缘
    sobel_img = set_channels_to_max(sobel_img)
    sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_BGR2GRAY)
    _, edge_mask = cv2.threshold(sobel_img, edge_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join('output', '3.1 edge_mask.jpg'), sobel_img)
    # 应用输入掩膜确保边缘仅在感兴趣区域内被识别

    edge_mask = cv2.bitwise_and(edge_mask, edge_mask, mask=input_mask)
    cv2.imwrite(os.path.join('output', '3.2 edge_mask.jpg'), edge_mask)

    closed_img = close(edge_mask, 3, 3)
    cv2.imwrite(os.path.join('output', '3.3 closed_img.jpg'), closed_img)
    closed_img = filter_small_white_regions(closed_img, min_area)
    cv2.imwrite(os.path.join('output', '3.4 filtered.jpg'), closed_img)
    # Find all strong edge contours
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask of the same size as the original image, initially all black
    mask = np.zeros_like(original_img, dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Process each found contour
    # valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    valid_contours = contours
    print('Edge_contour found:', len(valid_contours))
    # Find adjacent contours and apply gray tolerance
    if len(valid_contours) == 1:
        mid_point = np.mean([np.mean(valid_contours[0], axis=0), np.mean(valid_contours[0], axis=0)],
                            axis=0).astype(int)
        x, y = mid_point[0][0], mid_point[0][1]

        draw_circle_on_image(original_img, 10, (x, y))

        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        flood_mask = np.zeros((original_img.shape[0] + 2, original_img.shape[1] + 2), dtype=np.uint8)
        if not colored:
            num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255,
                                                (gray_tolerance, gray_tolerance, gray_tolerance),
                                                (gray_tolerance, gray_tolerance, gray_tolerance), flags)
        else:
            num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255,
                                                (3 * gray_tolerance, 3 * gray_tolerance, 3 * gray_tolerance),
                                                (3 * gray_tolerance, 3 * gray_tolerance, 3 * gray_tolerance), flags)
        print(
            f'FloodFill operation filled {num_filled} pixels starting from ({x}, {y}).')

        flood_mask = flood_mask[1:-1, 1:-1]
        if num_filled > 0:
            mask = cv2.bitwise_or(mask, flood_mask)
    else:
        for i in range(len(valid_contours)):
            if cv2.contourArea(valid_contours[i]) > 1000:
                print('大边缘')
                mid_point = np.mean([np.mean(valid_contours[i], axis=0), np.mean(valid_contours[i], axis=0)],
                                    axis=0).astype(int)
                x, y = mid_point[0][0], mid_point[0][1]
                draw_circle_on_image(original_img, 10, (x, y))
                # Applying gray tolerance segmentation at the midpoint

                flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
                flood_mask = np.zeros((original_img.shape[0] + 2, original_img.shape[1] + 2), dtype=np.uint8)
                if not colored:
                    num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255,
                                                        (gray_tolerance, gray_tolerance, gray_tolerance),
                                                        (gray_tolerance, gray_tolerance, gray_tolerance), flags)
                else:
                    num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255,
                                                        (3 * gray_tolerance, 3 * gray_tolerance, 3 * gray_tolerance),
                                                        (3 * gray_tolerance, 3 * gray_tolerance, 3 * gray_tolerance),
                                                        flags)
                print(
                    f'FloodFill operation filled {num_filled} pixels starting from ({x}, {y})')

                flood_mask = flood_mask[1:-1, 1:-1]
                if num_filled > 0:
                    mask = cv2.bitwise_or(mask, flood_mask)
            for j in range(i + 1, len(valid_contours)):
                # print(np.mean(valid_contours[i], axis=0), np.mean(valid_contours[j], axis=0))
                if np.linalg.norm(np.mean(valid_contours[i], axis=0) - np.mean(valid_contours[j], axis=0)) < proximity:

                    mid_point = np.mean([np.mean(valid_contours[i], axis=0), np.mean(valid_contours[j], axis=0)],
                                        axis=0).astype(int)
                    x, y = mid_point[0][0], mid_point[0][1]
                    draw_circle_on_image(original_img, 10, (x, y))
                    # Applying gray tolerance segmentation at the midpoint

                    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
                    flood_mask = np.zeros((original_img.shape[0] + 2, original_img.shape[1] + 2), dtype=np.uint8)
                    if not colored:
                        num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255, (gray_tolerance,gray_tolerance, gray_tolerance),
                                                            (gray_tolerance,gray_tolerance, gray_tolerance), flags)
                    else:
                        num_filled, _, _, _ = cv2.floodFill(inpainted_img, flood_mask, (x, y), 255,
                                                            (3 * gray_tolerance,3 * gray_tolerance,3 * gray_tolerance),
                                                            (3 * gray_tolerance,3 * gray_tolerance,3 * gray_tolerance), flags)

                    print(
                        f'FloodFill operation filled {num_filled} pixels starting from ({x}, {y})')

                    flood_mask = flood_mask[1:-1, 1:-1]
                    if num_filled > 0:
                        mask = cv2.bitwise_or(mask, flood_mask)
    cv2.imwrite(os.path.join('output', '3.9 mask.jpg'), mask)
    mask = fill_large_white_areas(mask, 200000)
    return mask


def segment_by_histogram_or_edges(original_img, input_mask, ksize=3, edge_threshold=10, gray_tolerance=5, min_area=100,
                                  proximity=50):
    """# 提取掩膜区域内的灰度值
    masked_img = cv2.bitwise_and(original_img, original_img, mask=input_mask)
    gray_values = masked_img[input_mask > 0]

    # 计算灰度直方图
    hist = cv2.calcHist([gray_values], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # 查找直方图中的峰值
    peaks, _ = find_peaks(hist, height=0)
    plt.figure()
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('output', '3.1 histogram.png'))"""
    # print('divide by edge')
    # 否则调用 segment_image_by_edges_and_tolerance 进行进一步处理
    return segment_image_by_edges_and_tolerance(original_img, input_mask, ksize, edge_threshold, gray_tolerance,
                                                min_area, proximity)

'''    adaptive_thresh = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 25, 2)
    adaptive_thresh = cv2.bitwise_and(adaptive_thresh, input_mask)
    adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
    adaptive_thresh = cv2.bitwise_and(adaptive_thresh, input_mask)
    adaptive_thresh = filter_small_white_regions(adaptive_thresh, 10)
    cv2.imwrite(os.path.join('output', '3.2 adaptive_thresh.jpg'), adaptive_thresh)'''

"""    if np.sum(adaptive_thresh) < 0:
        print('divide by threshold')
        return adaptive_thresh
    else:"""



def draw_circle_on_image(img, radius, center, thickness=5, output_dir='output'):

    # 中心坐标
    x, y = center

    # 画一个红色的圆
    color = (0, 0, 255)  # 红色 (BGR格式)
    cv2.circle(img, (x, y), radius, color, thickness)

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果图像
    output_path = os.path.join(output_dir, '3.4_point.jpg')
    cv2.imwrite(output_path, img)

    print(f"Result image saved to: {output_path}")


def fill_large_white_areas(image, area_threshold):
    """
    识别所有白色连续块。如果检测到连续的、面积大于指定阈值的白色块，则将其填充为黑色后返回。

    :param image: 输入黑白图像 (numpy array)
    :param area_threshold: 面积阈值 (int)
    :return: 填充后的图像 (numpy array)
    """
    # 复制输入图像
    # 找到所有的连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # 创建一个与输入图像相同大小的黑色图像
    output = np.zeros_like(image)
    print(num_labels)
    # 遍历所有连通域
    for i in range(1, num_labels):  # 忽略背景连通域
        # 获取当前连通域的面积
        area = stats[i, cv2.CC_STAT_AREA]
        print(area)
        # 如果面积大于阈值，则将该连通域填充为黑色
        if area > area_threshold:
            output[labels == i] = 0
        else:
            output[labels == i] = 255

    return output


def is_color_image(image, threshold=10):
    """
    判断图像是彩色图像还是黑白图像
    :param image: 输入图像
    :param threshold: 用于判断的阈值
    :return: 如果是彩色图像返回True，否则返回False
    """
    # 如果图像是灰度图像，直接返回False
    if len(image.shape) == 2 or image.shape[2] == 1:
        return False

    # 计算每个像素的颜色通道差异
    diff_b_g = np.abs(image[:, :, 0] - image[:, :, 1])
    diff_b_r = np.abs(image[:, :, 0] - image[:, :, 2])
    diff_g_r = np.abs(image[:, :, 1] - image[:, :, 2])

    # 计算差异的平均值
    mean_diff_b_g = np.mean(diff_b_g)
    mean_diff_b_r = np.mean(diff_b_r)
    mean_diff_g_r = np.mean(diff_g_r)

    # 如果颜色通道差异的平均值都小于阈值，认为是黑白图像
    if mean_diff_b_g < threshold and mean_diff_b_r < threshold and mean_diff_g_r < threshold:
        return False
    else:
        return True

def set_channels_to_max(image):
    # 获取图像的形状
    h, w, c = image.shape

    # 确保图像是彩色图像
    assert c == 3, "输入图像必须是彩色图像（具有3个通道）"

    # 获取每个像素的最大通道值
    max_channel = np.max(image, axis=2)

    # 将所有通道设置为最大通道值
    max_image = np.stack([max_channel] * 3, axis=-1)

    return max_image