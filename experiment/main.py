import os
import cv2
import numpy as np
import utils
import argparse
import detect


def step_test(args):
    print('Loading Data')
    bmp_files = utils.find_bmp_files('data')
    print('ROI extracting')
    roi_img = utils.extract_screen_roi(bmp_files[7], padding=args.roi_padding)
    print('Fourier transforming')
    pts = utils.pixel_texture_suppression(roi_img, args.sigma)
    print('Illumination correcting')
    illumination_img = utils.correct_illumination(pts)
    blurred_image = cv2.GaussianBlur(illumination_img, (args.gaussian_blur_kernel_size, args.gaussian_blur_kernel_size), 0)
    screen_mask = utils.find_screen_mask(roi_img, 70, epoch=40)
    print('Dividing')
    sobel_img = utils.segment_by_histogram_or_edges(blurred_image, screen_mask, edge_threshold=30, gray_tolerance=3, ksize=5, min_area=200, proximity=50)

    cv2.imwrite(os.path.join('output', '6 edge_image.jpg'), sobel_img)
    sobel_img = utils.filter_small_white_regions(sobel_img, 20)
    cv2.imwrite(os.path.join('output', '7 filtered.jpg'), sobel_img)
    result_img = utils.highlight_defects(roi_img, sobel_img)
    cv2.imwrite(os.path.join('output', '8 result_img.jpg'), result_img)


def main(args):
    bmp_files = utils.find_bmp_files('data')
    counter = 0
    for bmp_file in bmp_files:
        counter += 1
        roi_img = utils.extract_screen_roi(bmp_file, padding=args.roi_padding)
        img = roi_img.copy()
        filename = 'output/1. roi/roi_{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        img = utils.pixel_texture_suppression(img, args.sigma)
        filename = 'output/2. low_pass/low_pass{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        img = utils.fit_illumination_model(img)
        filename = 'output/3. corrected_image/corrected_image{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        screen_mask = utils.find_screen_mask(img, 50, epoch=40)

        img = cv2.GaussianBlur(img, (args.gaussian_blur_kernel_size, args.gaussian_blur_kernel_size), 0)
        sobel_img = utils.segment_by_histogram_or_edges(img, screen_mask, edge_threshold=35, gray_tolerance=2, ksize=5, min_area=100, proximity=50)
        filename = 'output/4. edge/edge{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        img = utils.highlight_defects(roi_img, sobel_img)
        filename = 'output/7. output/output{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        print(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="program")

    parser.add_argument('--data_path', '-p', type=str, default='data', help='path to data')

    parser.add_argument('--roi_padding', '-padding', type=int, default=25, help='path to GloVo model')
    parser.add_argument('--sigma', '-s', type=int, default=50, help='path to GloVo model')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='path to GloVo model')
    parser.add_argument('--gaussian_blur_kernel_size', '-gb', type=int, default=15, help='path to GloVo model')
    parser.add_argument('--sobel_kernel_size', '-sks', type=int, default=7, help='path to GloVo model')
    parser.add_argument('--sobel_threshold', '-skh', type=int, default=700, help='path to GloVo model')
    parser.add_argument('--open_kernel', '-ok', type=int, default=5, help='path to GloVo model')
    parser.add_argument('--close_kernel', '-ck', type=int, default=25, help='path to GloVo model')
    parser.add_argument('--minimum_size_1', '-ms1', type=int, default=200, help='path to GloVo model')
    parser.add_argument('--minimum_size_2', '-ms2', type=int, default=500, help='path to GloVo model')

    args = parser.parse_args()
    step_test(args)
    # main(args)
