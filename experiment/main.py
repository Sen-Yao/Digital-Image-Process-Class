import os
import cv2
import numpy as np
import utils
import argparse
import detect


def step_test(args):
    bmp_files = utils.find_bmp_files('data')
    roi_img = utils.extract_screen_roi(bmp_files[0], padding=args.roi_padding)
    pts = utils.pixel_texture_suppression(roi_img, args.sigma)
    illumination_img = utils.fit_illumination_model(pts)
    cv2.imwrite(os.path.join('output', '5.1 fix_background_img.jpg'), illumination_img)
    blurred_image = cv2.GaussianBlur(illumination_img, (args.gaussian_blur_kernel_size, args.gaussian_blur_kernel_size), 0)
    screen_mask = utils.find_screen_mask(illumination_img, 50, epoch=args.epoch)
    # sobel_img = utils.watershed_segmentation(blurred_image, screen_mask)
    # sobel_img = utils.sobel(blurred_image, screen_mask, kenel=args.sobel_kernel_size, threshold=args.sobel_threshold)
    # sobel_img = utils.canny_edge_detection(blurred_image, screen_mask, 2, 6)
    sobel_img = detect.segment_image(blurred_image, screen_mask, 20, 2)
    contour_img = utils.draw_all_contours_on_intervals(blurred_image, 16)
    cv2.imwrite(os.path.join('output', '5.6 contour_img.jpg'), contour_img)
    cv2.imwrite(os.path.join('output', '6 edge_image.jpg'), sobel_img)
    sobel_img = utils.filter_small_white_regions(sobel_img, 20)
    closed_img = utils.close(sobel_img, args.close_kernel, args.close_kernel)

    cv2.imwrite(os.path.join('output', '7.1 closed_img.jpg'), closed_img)
    opened_img = utils.open(closed_img, args.open_kernel, args.open_kernel)
    opened_img = utils.filter_small_white_regions(opened_img, args.minimum_size_1)
    cv2.imwrite(os.path.join('output', '7.2 opened_img.jpg'), opened_img)
    filtered_img = utils.filter_small_white_regions(opened_img, args.minimum_size_2)
    cv2.imwrite(os.path.join('output', '7.3 filtered.jpg'), filtered_img)
    result_img = utils.highlight_defects(roi_img, filtered_img)
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

        screen_mask = utils.find_screen_mask(img)
        img = cv2.GaussianBlur(img, (args.gaussian_blur_kernel_size, args.gaussian_blur_kernel_size), 0)
        img = utils.sobel(img, screen_mask, kenel=args.sobel_kernel_size, threshold=args.sobel_threshold)

        filename = 'output/4. edge/edge{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        opened_img = utils.open(img, args.open_kernel, args.open_kernel)
        img = utils.close(opened_img, args.close_kernel, args.close_kernel)
        filename = 'output/5. open_close/open_close{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        img = utils.filter_small_white_regions(img, args.minimum_size)
        filename = 'output/6. filter/filter{}.jpg'.format(counter)
        cv2.imwrite(filename, img)
        filename = 'output/7. output/output{}.jpg'.format(counter)
        img = utils.highlight_defects(roi_img, img)
        cv2.imwrite(filename, img)
        print(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="program")

    parser.add_argument('--data_path', '-p', type=str, default='data', help='path to data')

    parser.add_argument('--roi_padding', '-padding', type=int, default=15, help='path to GloVo model')
    parser.add_argument('--sigma', '-s', type=int, default=50, help='path to GloVo model')
    parser.add_argument('--epoch', '-e', type=int, default=25, help='path to GloVo model')
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
