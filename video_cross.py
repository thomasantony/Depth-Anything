import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


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
    return np.hstack([left, right_fix])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()

    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)

    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)

        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 # + margin_width

        filename = os.path.basename(filename)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_stereo_cross.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        num_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth = depth.cpu().numpy().astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            # combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])

            stereo_cross = generate_stereo(raw_frame, depth)
            out.write(stereo_cross)
            # store frame to jpg
            # filename = 'frame_{:d}.jpg'.format(int(raw_video.get(cv2.CAP_PROP_POS_FRAMES)))
            # cv2.imwrite(os.path.join(args.outdir, filename), stereo_cross)
            print('Progress {:}/{:},'.format(i, num_frames))
            i += 1

        raw_video.release()
        out.release()
