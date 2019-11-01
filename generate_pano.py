"""Image stitching is implemented in the left half of the view and the right
half separately.
Calling stitch.stitch_image() to stitch two adjacent key frames and
append_operation.append_image() join the left and right half parts together.
"""
from stitch import stitch_image
from append_operation import append_image
import cv2
import argparse


def main(key_frame_num):
    current_img = 'key_frames/frame0.jpg'

    for i in range(1, key_frame_num):
        if 11 < i < 15:  # just for improving performance of the specific task
            continue
        if i == 15:
            left = current_img
            current_img = 'key_frames/frame{}.jpg'.format(i)
            continue
        next_img = 'key_frames/frame{}.jpg'.format(i)
        print("Stitching frame{} and frame{}...".format(i - 1, i))
        current_img = stitch_image(current_img, next_img)

    if key_frame_num > 16:
        right = current_img
        result = append_image(left, right)
        print("Join left and right parts.")
    else:  # too few frames, no need to divide into two groups
        result = current_img

    cv2.imwrite('panoramic.jpg', result)
    print("panoramic.jpg complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num', nargs='?', default=23, type=int,
                        help="the number of key frames (default: 23)")
    args = parser.parse_args()

    main(args.num)
