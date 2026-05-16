"""
Utility Functions
Common helper functions for geometry, video I/O, and pose visualization.
"""

import cv2
import numpy as np
import math


# ==================== Video I/O ====================

def read_video_frames(video_path):
    """Read all frames from video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if fps == 0:
        raise ValueError("Invalid video file or FPS is 0")
    
    total_time = frame_count / fps
    return frames, fps, total_time


def write_video_from_frames(frames, output_path, fps=30):
    """Write frames to video file."""
    if len(frames) == 0:
        raise ValueError("No frames to write")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


# ==================== Geometry Functions ====================

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_angular_displacement(x_head_prev, y_head_prev, x_beak_prev, y_beak_prev,
                                   x_head_curr, y_head_curr, x_beak_curr, y_beak_curr):
    """
    Calculate angular displacement between two head-beak vectors.
    
    Returns:
        theta_radians, theta_degrees, cross_product
    """
    # Previous vector
    vec_prev = np.array([
        x_beak_prev - x_head_prev,
        y_beak_prev - y_head_prev
    ])
    mag_prev = np.linalg.norm(vec_prev)
    
    # Current vector
    vec_curr = np.array([
        x_beak_curr - x_head_curr,
        y_beak_curr - y_head_curr
    ])
    mag_curr = np.linalg.norm(vec_curr)
    
    # Avoid division by zero
    if mag_prev == 0 or mag_curr == 0:
        return 0.0, 0.0, 0.0
    
    # Dot product for angle
    dot_prod = np.dot(vec_prev, vec_curr)
    cos_theta = dot_prod / (mag_prev * mag_curr)
    
    # Clamp to valid range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    # Cross product for direction
    cross_prod = np.cross(vec_prev / mag_prev, vec_curr / mag_curr)
    
    return theta_rad, theta_deg, cross_prod


def box_to_center_scale(box, model_width, model_height, bbox_expand=1.0):
    """
    Convert bounding box to center and scale for pose model.
    
    Args:
        box: [(x1, y1), (x2, y2)]
        model_width: Model input width
        model_height: Model input height
        bbox_expand: Extra multiplier applied on top of the default 1.25×
                     padding (1.0 = no change, 1.5 = 1.875× total, etc.).
    
    Returns:
        center: (cx, cy)
        scale: (sx, sy)
    """
    center = np.zeros(2, dtype=np.float32)
    
    bottom_left = box[0]
    top_right = box[1]
    
    box_width = top_right[0] - bottom_left[0]
    box_height = top_right[1] - bottom_left[1]
    
    center[0] = bottom_left[0] + box_width * 0.5
    center[1] = bottom_left[1] + box_height * 0.5
    
    aspect_ratio = model_width * 1.0 / model_height
    pixel_std = 200
    
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    
    scale = np.array([
        box_width * 1.0 / pixel_std,
        box_height * 1.0 / pixel_std
    ], dtype=np.float32)
    
    if center[0] != -1:
        scale = scale * 1.25
    
    if bbox_expand != 1.0:
        scale = scale * bbox_expand
    
    return center, scale


# ==================== Pose Visualization ====================

# Skeleton connections for bird pose
SKELETON = [
    [0, 1], [0, 2],  # Head to Eyes
    [0, 3],          # Head to Beak
]

# Colors for keypoints: Head, LeftEye, RightEye, Beak
KEYPOINT_COLORS = [
    [255, 0, 0],      # Blue - Head
    [170, 0, 255],    # Rose - Left Eye
    [255, 255, 0],    # Cyan - Right Eye
    [0, 255, 170]     # Yellow - Beak
]

# Colors for skeleton lines
SKELETON_COLORS = [
    [60, 0, 255],
    [60, 0, 255],
    [60, 0, 255]
]


def DrawHeadPose(keypoints, img):
    """
    Draw keypoints and skeleton on image.
    
    Args:
        keypoints: Array of shape (4, 2) with [x, y] coordinates
        img: Image to draw on
    
    Returns:
        Image with drawn pose
    """
    assert keypoints.shape == (4, 2), f"Expected shape (4,2), got {keypoints.shape}"
    
    # Draw skeleton lines
    for i, (kpt_a_idx, kpt_b_idx) in enumerate(SKELETON):
        x_a, y_a = keypoints[kpt_a_idx][0], keypoints[kpt_a_idx][1]
        x_b, y_b = keypoints[kpt_b_idx][0], keypoints[kpt_b_idx][1]
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)),
                SKELETON_COLORS[i], 2)
    
    # Draw keypoints
    for i in range(4):
        x, y = keypoints[i][0], keypoints[i][1]
        cv2.circle(img, (int(x), int(y)), 6, KEYPOINT_COLORS[i], -1)
    
    return img


def DrawBoxes(image, boxes, frame_number=None, color=(0, 255, 0), thickness=5):
    """Draw bounding boxes on image."""
    image_copy = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)),
                     color, thickness)
    
    # Add frame number
    if frame_number is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(str(frame_number), font, 1.5, 2)[0]
        text_x = 10
        text_y = 10 + text_size[1]
        cv2.putText(image_copy, str(frame_number), (text_x, text_y),
                   font, 1.5, color, 2)
    
    return image_copy


# ==================== YOLO Utilities ====================

def get_yolo_model():
    """
    Load YOLO model from configured path.
    
    Returns:
        YOLO model or None if not found
    """
    from ultralytics import YOLO
    from lib.config import cfg
    import os
    
    model_path = None
    
    # Try config first
    try:
        if hasattr(cfg, 'TEST') and getattr(cfg.TEST, 'DETECT_MODEL_FILE', ''):
            model_path = cfg.TEST.DETECT_MODEL_FILE
    except:
        pass
    
    # Default path
    if not model_path:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, '..', 'models', 'trained', 'YOLOv8Bird.pt')
    
    # Make absolute
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    if not os.path.exists(model_path):
        print(f'YOLO model not found at {model_path}')
        return None
    
    try:
        print(f'Loading YOLO model from {model_path}')
        return YOLO(model_path)
    except Exception as e:
        print(f'Failed to load YOLO model: {e}')
        return None


# ==================== File I/O Utilities ====================

def save_pose_scores(scores_list, filepath):
    """Save pose scores to text file."""
    with open(filepath, 'w') as f:
        for scores in scores_list:
            if scores is not None:
                for item in scores:
                    try:
                        f.write(f"{item[0]}\n")
                    except:
                        f.write("0.0\n")
                f.write("\n")
            else:
                f.write("\n")


def read_pose_scores(filepath):
    """Read pose scores from text file."""
    scores_list = []
    with open(filepath, 'r') as f:
        sublist = []
        for line in f:
            if line.strip():
                sublist.append([float(line.strip())])
            else:
                scores_list.append(sublist)
                sublist = []
        if sublist:
            scores_list.append(sublist)
    return scores_list


def calculate_centers(boxes):
    """Calculate center points of bounding boxes."""
    centers = []
    for box in boxes:
        if box:
            x1, y1, x2, y2 = box[0] if isinstance(box[0], tuple) else box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        else:
            centers.append(())
    return centers