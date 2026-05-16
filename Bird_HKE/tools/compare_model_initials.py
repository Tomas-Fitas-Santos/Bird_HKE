"""
Create a side-by-side comparison video for initial pose estimates
from multiple trained models.

Input videos are expected at:
Bird_HKE/videos_experiments/<model_name>/<video_title>/head_pose_estimation.mp4
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


MODEL_NAMES = [
    "HRNET_W32_BirdGaze",
    "VHR_BirdPose",
    "VHR_MAMBA_VISION_T",
    "VHR_MAMBA_D1",
]

DEFAULT_INPUT_FILENAME = "head_pose_estimation.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare initial pose estimates across 4 models."
    )
    parser.add_argument(
        "--video_title",
        type=str,
        required=True,
        help="Video title folder name under each model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: videos_experiments/compare_<video_title>.mp4)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Padding (px) between videos",
    )
    parser.add_argument(
        "--label_height",
        type=int,
        default=40,
        help="Height (px) reserved for label under each video",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.8,
        help="Font scale for labels",
    )
    parser.add_argument(
        "--font_thickness",
        type=int,
        default=2,
        help="Font thickness for labels",
    )
    return parser.parse_args()


def build_input_paths(root_dir: Path, video_title: str):
    paths = []
    for model_name in MODEL_NAMES:
        video_path = (
            root_dir
            / "videos_experiments"
            / model_name
            / video_title
            / DEFAULT_INPUT_FILENAME
        )
        paths.append(video_path)
    return paths


def open_captures(paths):
    caps = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        caps.append(cap)
    return caps


def get_common_size(caps):
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    min_w = min(widths)
    min_h = min(heights)
    return min_w, min_h


def add_label(frame, label, label_height, font_scale, thickness):
    h, w = frame.shape[:2]
    label_area = np.zeros((label_height, w, 3), dtype=np.uint8)
    label_area[:] = (15, 15, 15)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(0, (w - text_size[0]) // 2)
    text_y = max(text_size[1] + 5, label_height - 10)

    cv2.putText(
        label_area,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return np.vstack([frame, label_area])


def assemble_grid(frames, padding):
    """
    Assemble 4 frames into a 2x2 grid with padding.
    frames: list of 4 images (H x W x 3)
    """
    if len(frames) != 4:
        raise ValueError("Expected exactly 4 frames")

    h, w = frames[0].shape[:2]
    pad_col = np.zeros((h, padding, 3), dtype=np.uint8)
    pad_row = np.zeros((padding, w * 2 + padding, 3), dtype=np.uint8)

    top = np.hstack([frames[0], pad_col, frames[1]])
    bottom = np.hstack([frames[2], pad_col, frames[3]])
    grid = np.vstack([top, pad_row, bottom])

    return grid


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    input_paths = build_input_paths(root_dir, args.video_title)
    caps = open_captures(input_paths)

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    width, height = get_common_size(caps)

    label_height = args.label_height
    padded_height = height + label_height
    padded_width = width

    output_width = padded_width * 2 + args.padding
    output_height = padded_height * 2 + args.padding

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = root_dir / "videos_experiments" / f"compare_{args.video_title}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

    try:
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    return
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(frame)

            labeled_frames = []
            for frame, label in zip(frames, MODEL_NAMES):
                labeled = add_label(
                    frame,
                    label,
                    label_height,
                    args.font_scale,
                    args.font_thickness,
                )
                labeled_frames.append(labeled)

            grid = assemble_grid(labeled_frames, args.padding)
            writer.write(grid)
    finally:
        for cap in caps:
            cap.release()
        writer.release()


if __name__ == "__main__":
    main()

