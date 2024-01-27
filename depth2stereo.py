import os
import cv2
import numpy as np
from argparse import ArgumentParser

IPD = 6.5
MONITOR_W = 38.5
NEAR_DIST = 5
FAR_DIST = 70


def generate_stereo(left, depth):
    h, w, c = left.shape

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left)

    deviation_cm = IPD * NEAR_DIST / FAR_DIST
    deviation = deviation_cm * w / MONITOR_W

    cols = np.arange(w)
    col_r = cols - ((1 - depth) * deviation).astype(int)
    # rows, col_r_cols = np.where(col_r >= 0, col_r, 0)

    # Create a 2D grid of indices
    rows, cols = np.indices((h, w))

    # Get the corresponding c_r values
    c_r_values = col_r[rows, cols]

    # Create a mask where c_r is greater than or equal to 0
    mask = c_r_values >= 0

    # Apply the mask to the rows and c_r_values (which act as columns for 'right')
    right[rows[mask], c_r_values[mask]] = left[rows[mask], cols[mask]]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    cnt = 0
    for row, col in zip(rows, cols):
        cnt = cnt + 1
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break
    result = np.hstack([left, right_fix])
    return result


def process(in_file, depth_file, out_dir):
    left = cv2.imread(in_file)
    depth_src = cv2.imread(depth_file)
    depth = cv2.cvtColor(depth_src, cv2.COLOR_BGR2GRAY)
    result = generate_stereo(left, depth, out_dir)
    filename = os.path.basename(in_file).split(".")[0]
    cv2.imwrite(os.path.join(out_dir, "stereo_" + filename + ".jpg"), result)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--in-dir", help="Input directory for source images", default="example")
    arg_parser.add_argument("-d", "--depth-dir", help="Directory of depth maps", default="depth")
    arg_parser.add_argument("-p", "--depth-prefix", help="Prefix of file name for depth maps", default="MiDaS_")
    arg_parser.add_argument("-o", "--out-dir", help="Output directory for stereo images", default="stereo")
    args = arg_parser.parse_args()

    in_dir = args.in_dir
    depth_dir = args.depth_dir
    depth_prefix = args.depth_prefix
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for f in os.listdir(in_dir):
        filename = f.split(".")[0]
        in_file = os.path.join(in_dir, filename + ".jpg")
        depth_file = os.path.join(depth_dir, depth_prefix + filename + ".png")
        process(in_file, depth_file, out_dir)


if __name__ == "__main__":
    # main()
    process("../Depth-Anything/bison.jpg", "../Depth-Anything/output/bison_depth.png", "stereo")
