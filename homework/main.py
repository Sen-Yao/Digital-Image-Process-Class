import argparse
import cv2
import os

from HistogramEqualization import histogram_equalization
from LaplaceOperation import laplace_operation
from GaussianHighpass import gaussian_highpass_filter
from ArithmeticEncode import arithmetic_encode
from Morphology import morphology_operation

def main(args):
    functions = [histogram_equalization,
                 laplace_operation,
                 gaussian_highpass_filter,
                 arithmetic_encode,
                 morphology_operation]
    input_path = args.input_path
    images = []
    file_names = os.listdir(input_path)
    for file_name in file_names:
        file_path = os.path.join(input_path, file_name)
        # Check pics
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            # 读取图像
            image = cv2.imread(file_path)
            print('Processing', file_name)
            # 检查图像是否正确加载
            if image is not None:
                images.append(image)
            else:
                print(f"Could not open or find the image: {file_path}")

            # 图像处理
            if args.function == 0:
                for function in functions:
                    print(f"Applying function: {function.__name__}")
                    function(image, file_path, args.output_path)
            else:
                functions[args.function - 1](image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Image Process Homework")

    parser.add_argument('--function', '-f', type=int, default=0, help='Function to display:\n0: All\n'
                                                                      '1: Histogram Equalization\n'
                                                                      '2. Laplace Operation\n'
                                                                      '3. Gaussian High Pass Filter\n'
                                                                      '4. Arithmetic Encode\n'
                                                                      '5. Morphology Operation')
    parser.add_argument('--input_path', '-ip', type=str, default='data/input', help='path to input data')
    parser.add_argument('--output_path', '-op', type=str, default='data/output', help='path to output data')

    args = parser.parse_args()

    main(args)
