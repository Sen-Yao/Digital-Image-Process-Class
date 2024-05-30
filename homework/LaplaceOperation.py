import cv2
import os


def laplace_operation(input_image, file_path, output_path):
    laplace_output_path = os.path.join(output_path, 'laplace_operation')
    os.makedirs(laplace_output_path, exist_ok=True)
    laplace_output_path = os.path.join(laplace_output_path, 'laplace_operation' + os.path.basename(file_path))
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行拉普拉斯运算
    laplace_image = cv2.Laplacian(gray_image, cv2.CV_64F)

    # 归一化图像
    laplace_image = cv2.convertScaleAbs(laplace_image)

    # 保存处理后的图像
    cv2.imwrite(laplace_output_path, laplace_image)

