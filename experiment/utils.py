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


def extract_screen_roi(image_path, padding=5):
    # 加载图像
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (7096, 5320))
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


def fit_illumination_model(channel):
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
    corrected_channel = channel - fitted_image + np.mean(fitted_image)
    cv2.imwrite(os.path.join('output', '5.6 enhanced_image.jpg'), corrected_channel)
    return np.clip(corrected_channel, 0, 255).astype(np.uint8)


def correct_illumination(image):
    # 分离颜色通道
    blue, green, red = cv2.split(image)

    # 对每个通道应用光照拟合
    corrected_blue = fit_illumination_model(blue)
    corrected_green = fit_illumination_model(green)
    corrected_red = fit_illumination_model(red)

    # 合并通道
    corrected_image = cv2.merge([corrected_blue, corrected_green, corrected_red])
    cv2.imwrite(os.path.join('output', '2. corrected_image.jpg'), corrected_image)
    return corrected_image


def fit_illumination_model(image):
    rows, cols = image.shape
    # 生成坐标网格
    y, x = np.indices((rows, cols))
    # 展平图像和坐标
    x = x.ravel()
    y = y.ravel()
    z = image.ravel()

    # 构建矩阵A，包含x, y, xy, x^2, y^2项
    A = np.c_[x, y, x*y, x**2, y**2, np.ones_like(x)]
    # 使用最小二乘法拟合光照模型
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # 使用拟合的模型重建光照分布
    fitted = coeffs[0]*x + coeffs[1]*y + coeffs[2]*x*y + coeffs[3]*x**2 + coeffs[4]*y**2 + coeffs[5]
    fitted_image = fitted.reshape(rows, cols)

    # 从原图中减去拟合的背景模型
    corrected_image = image - fitted_image + np.mean(fitted_image)
    cv2.imwrite(os.path.join('output', '5. corrected_image.jpg'), corrected_image)
    return np.clip(corrected_image, 0, 255).astype(np.uint8)



def pixel_texture_suppression(image, sigma):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(image)

    f_shift = np.fft.fftshift(f_transform)  # 将零频率移到频谱中心
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9)  # 计算幅度谱
    # 将频谱结果保存为图像文件
    cv2.imwrite(os.path.join('output', '2. pixel_texture_suppression.jpg'), magnitude_spectrum)
    # 设计高斯低通滤波器
    rows, cols = image.shape
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
    cv2.imwrite(os.path.join('output', '3. pixel_texture_suppression_filtered.jpg'), magnitude_spectrum_filtered)

    # 逆向傅立叶变换
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 转换结果为0到255的整数
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join('output', '4. img_back_normalized.jpg'), img_back_normalized)

    return img_back_normalized


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
    corrected_image = cv2.addWeighted(image, 1, blurred, -1, 128) # 增加128是为了避免负数的产生
    cv2.imwrite(os.path.join('output', '2. illumination_correction.jpg'), corrected_image)
    return corrected_image


def find_screen_mask(img, threshold=50, epoch=20):
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

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


def draw_all_contours_on_intervals(image, interval=5):
    # 创建一个与原图同大小的黑色图像，用于绘制等高线
    contour_image = np.zeros_like(image)

    # 遍历不同的灰度级阈值
    for threshold in range(0, 256, interval):
        # 应用阈值，生成二值图像
        _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # 找出二值图像的所有等高线
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 在黑色图像上绘制等高线
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    return contour_image