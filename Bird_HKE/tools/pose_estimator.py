"""
Pose Estimation Module
Handles bird pose estimation, refinement, filtering, and smoothing.
"""

import numpy as np
import cv2
import copy
import torch
from pathlib import Path

from lib.config import cfg
from lib.core.function import get_final_preds
from lib.core.inference import reset_score_stats, print_score_stats
from lib.utilities.transforms import get_affine_transform
from .pose_filters import Validation_Smoothing_Filter, One_Euro_Custom_Filter
from .utils import box_to_center_scale, DrawHeadPose
from .utils import calculate_angular_displacement
import torchvision.transforms as transforms


# Pre-built image transform for model input
POSE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PoseEstimator:
    """Handles complete pose estimation pipeline."""
    
    NUM_KEYPOINTS = 4  # Head, LeftEye, RightEye, Beak
    
    def __init__(self, pose_model, device, results_dir, fps, total_time,
                 filter_type='one_euro', filter_mode='offline',
                 bbox_expand=1.0):
        """
        Initialize pose estimator.
        
        Args:
            pose_model: The pose estimation model
            device: Computing device
            results_dir: Directory for results
            fps: Video frame rate
            total_time: Total video duration
            filter_type: 'val_smooth', 'one_euro', or 'none' (default: 'one_euro')
            filter_mode: 'online' or 'offline' (default: 'offline')
        """
        self.model = pose_model
        self.device = device
        self.results_dir = Path(results_dir)
        self.fps = fps
        self.total_time = total_time
        self.bbox_expand = float(bbox_expand)
        
        # Filter configuration
        self.filter_type = filter_type.lower()
        self.filter_mode = filter_mode.lower()
        if self.filter_type in ('one_euro',) and self.filter_mode != 'offline':
            print("one_euro filter only supports offline mode. Switching to offline.")
            self.filter_mode = 'offline'
        
        # Initialize filters based on configuration
        if self.filter_type == 'none':
            self.filter = None
            print("No filtering applied. Final poses are initial poses.")
        elif self.filter_type == 'val_smooth':
            self.filter = Validation_Smoothing_Filter()
            print(f"Using Validation_Smoothing_Filter in {self.filter_mode} mode")
        elif self.filter_type in ('one_euro', 'stage1', 'confidence'):
            self.filter = One_Euro_Custom_Filter(fps=self.fps)
            print(f"Using One_Euro_Custom_Filter in {self.filter_mode} mode")
        else:
            raise ValueError(
                f"Unknown filter_type: {filter_type}. Use 'val_smooth', 'one_euro', or 'none'."
            )
        
        # Keep separate reference for backward compatibility
        self.pose_filter = Validation_Smoothing_Filter() if self.filter_type == 'val_smooth' else None
        
        # Statistics tracking
        self.stats = {
            'drift_errors': 0,
            'flicker_corrections': 0,
            'filter_type': self.filter_type,
            'filter_mode': self.filter_mode
        }
        
    def process_video(self, frames, bboxes, write_initial=False):
        """
        Process entire video for pose estimation.
        
        Returns:
            Dictionary containing all pose results
        """
        print(f"\n=== Pose Estimation ({self.filter_type.upper()} filter, {self.filter_mode} mode) ===")
        
        results = {
            'poses_initial': [],
            'poses_filtered': [],
            'poses_final': [],
            'scores_initial': [],
            'scores_final': [],
            'frames_initial': [],
            'frames_final': [],
            'motion_data': {
                'angles': [],
                'angles_unsmoothed': [],
                'velocities': [],
                'cross_products': []
            }
        }
        
        reset_score_stats()

        # Process based on filter mode
        if self.filter_mode == 'online':
            self._process_online(frames, bboxes, results, write_initial)
        else:  # offline
            self._process_offline(frames, bboxes, results, write_initial)
        
        # Update statistics
        if self.filter_type == 'val_smooth' and self.pose_filter:
            self.stats['drift_errors'] = self.pose_filter.drift_count
            self.stats['flicker_corrections'] = self.pose_filter.flicker_count
            print(f"Drift errors removed: {self.stats['drift_errors']}")
            print(f"Flicker corrections: {self.stats['flicker_corrections']}")
        
        return results
    
    def _process_online(self, frames, bboxes, results, write_initial):
        """Process video with online (per-frame) filtering."""
        prev_filtered = None
        prev_final = None
        pose_count = 0
        
        for frame_idx, frame in enumerate(frames):
            print(f"Processing pose {frame_idx+1}/{len(frames)}", end='\r')
            
            # Skip if no bounding box
            if not bboxes or frame_idx >= len(bboxes) or not bboxes[frame_idx]:
                self._append_empty_frame(results, frame)
                continue
            
            try:
                # Step 1: Initial pose estimation
                pose_init, score_init, frame_init = self._estimate_initial_pose(
                    frame, frame_idx, bboxes, write_initial
                )
                if pose_init is not None:
                    pose_count += 1

                # Step 2 & 3: Apply online filtering
                if self.filter_type == 'none':
                    pose_filtered = pose_init
                    pose_final = pose_init
                    score_filtered = score_init
                    score_final = score_init
                elif self.filter_type == 'val_smooth':
                    # Validation_Smoothing_Filter: filter + smooth
                    pose_filtered, score_filtered = self.filter.filter_pose(
                        pose_init, prev_filtered, score_init, frame_idx
                    )
                    pose_final, score_final = self.filter.smooth_pose(
                        pose_filtered, prev_filtered, score_filtered
                    )
                else:  # one_euro
                    pose_final, score_final, _ = self.filter.online_filter_pose(
                        pose_init, score_init, frame, frame_idx
                    )
                    pose_filtered = pose_final
                    score_filtered = score_final
                
                frame_final = self._draw_pose(frame.copy(), pose_final, score_final, frame_idx, 'final' + self.filter_type)
                
                # Step 4: Calculate motion metrics
                self._calculate_motion(
                    pose_final, prev_final, frame_idx, results['motion_data']
                )
                
                # Store results
                self._append_results(
                    results,
                    pose_init, pose_filtered, pose_final,
                    score_init, score_final,
                    frame_init, frame_final
                )
                
                # Update previous states
                prev_filtered = copy.deepcopy(pose_filtered)
                prev_final = copy.deepcopy(pose_final)
                
            except Exception as e:
                print(f"\nError processing frame {frame_idx}: {e}")
                self._append_empty_frame(results, frame)
        
        print()  # New line after progress
        print(f"Initial pose estimates produced: {pose_count}/{len(frames)}")
        if pose_count == 0:
            print("Warning: No initial poses were estimated. Check detection output (bboxes).")
        print_score_stats(prefix='Initial pose score stats')
    
    def _process_offline(self, frames, bboxes, results, write_initial):
        """Process video with offline (full sequence) filtering."""
        # Step 1: Estimate all poses first
        print("Step 1/2: Estimating initial poses...")
        all_poses = []
        all_scores = []
        pose_count = 0
        
        for frame_idx, frame in enumerate(frames):
            print(f"Processing frame {frame_idx+1}/{len(frames)}", end='\r')
            
            if not bboxes or frame_idx >= len(bboxes) or not bboxes[frame_idx]:
                all_poses.append(None)
                all_scores.append(None)
                pose_init, score_init, frame_init = None, None, frame
            else:
                pose_init, score_init, frame_init = self._estimate_initial_pose(
                    frame, frame_idx, bboxes, write_initial
                )
                all_poses.append(pose_init)
                all_scores.append(score_init)
                if pose_init is not None:
                    pose_count += 1
            
            # Store initial results
            self._append_pose(results['poses_initial'], pose_init)
            results['scores_initial'].append(score_init)
            results['frames_initial'].append(frame_init)
        
        print()  # New line after progress
        print(f"Initial pose estimates produced: {pose_count}/{len(frames)}")
        if pose_count == 0:
            print("Warning: No initial poses were estimated. Check detection output (bboxes).")
        print_score_stats(prefix='Initial pose score stats')
        
        # Step 2: Apply offline filtering to entire sequence
        print(f"Step 2/2: Applying {self.filter_type} filter to sequence...")
        
        if self.filter_type == 'none':
            filtered_poses = all_poses
            filtered_scores = all_scores
        elif self.filter_type == 'val_smooth':
            # Apply Validation_Smoothing_Filter offline (frame by frame with full context)
            filtered_poses, filtered_scores = self._apply_pose_filter_offline(
                all_poses, all_scores
            )
        elif self.filter_type in ('one_euro', 'stage1', 'confidence'):
            # Stage-1: high-confidence smoothing
            stage1_poses = self.filter.smooth_high_confidence(
                poses=all_poses,
                scores=all_scores
            )
            # Stage-2: low-confidence correction (placeholder)
            stage2_poses = self.filter.correct_low_confidence(
                poses=stage1_poses,
                scores=all_scores
            )
            # Stage-3: final smoothing (placeholder)
            filtered_poses = self.filter.final_smooth(
                poses=stage2_poses,
                scores=all_scores
            )
            filtered_scores = all_scores
        else:
            raise ValueError(
                f"Unknown filter_type: {self.filter_type}. Use 'val_smooth', 'one_euro', or 'none'."
            )
        
        # Generate visualization frames and store results
        print("Generating visualization frames...")
        for frame_idx, (frame, pose_final, score_final) in enumerate(zip(frames, filtered_poses, filtered_scores)):
            if pose_final is not None and not np.all(np.isnan(pose_final)):
                frame_final = self._draw_pose(frame.copy(), [pose_final], score_final, frame_idx, 'final' + self.filter_type)
                
                # Calculate motion
                prev_pose = filtered_poses[frame_idx-1] if frame_idx > 0 else None
                self._calculate_motion(
                    [pose_final], [prev_pose] if prev_pose is not None else None,
                    frame_idx, results['motion_data']
                )
                
                self._append_pose(results['poses_filtered'], [pose_final])
                self._append_pose(results['poses_final'], [pose_final])
                results['scores_final'].append(score_final)
                results['frames_final'].append(frame_final)
            else:
                self._append_pose(results['poses_filtered'], None)
                self._append_pose(results['poses_final'], None)
                results['scores_final'].append(None)
                results['frames_final'].append(frame)
                
                results['motion_data']['angles'].append(0.0)
                results['motion_data']['velocities'].append(0.0)
                results['motion_data']['cross_products'].append(0.0)
        
        print(f"Offline filtering complete! Processed {len(frames)} frames.")
    
    def _apply_pose_filter_offline(self, all_poses, all_scores):
        """Apply Validation_Smoothing_Filter to entire sequence (offline mode)."""
        filtered_poses = []
        filtered_scores = []
        prev_filtered = None
        
        for frame_idx, (pose, score) in enumerate(zip(all_poses, all_scores)):
            # Filter
            pose_filtered, score_filtered = self.filter.filter_pose(
                pose, prev_filtered, score, frame_idx
            )
            # Smooth
            pose_final, score_final = self.filter.smooth_pose(
                pose_filtered, prev_filtered, score_filtered
            )
            
            filtered_poses.append(pose_final if pose_final is not None else None)
            filtered_scores.append(score_final)
            prev_filtered = copy.deepcopy(pose_filtered)
        
        return filtered_poses, filtered_scores
    
    def _estimate_initial_pose(self, frame, frame_idx, bboxes, write_images):
        """Estimate initial pose from bounding box."""
        if not bboxes[frame_idx]:
            return None, None, frame
        
        bbox = bboxes[frame_idx][0]
        box = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]

        # Run pose estimation on bbox
        pose_bbox, score_bbox = self._predict_pose(frame, box=box)

        pose, score = pose_bbox, score_bbox
        
        # Visualize if requested
        if write_images and pose is not None:
            frame_vis = self._draw_pose(frame.copy(), pose, score, frame_idx, 'initial')
        else:
            frame_vis = frame
        
        return pose[0] if pose is not None else None, score[0] if score is not None else None, frame_vis
    
    def _predict_pose(self, image, center=None, scale=None, box=None):
        """Run pose model on cropped image patch."""
        rotation = 0
        
        # Convert box to center/scale
        if box is not None:
            if isinstance(box, (list, tuple)) and len(box) == 4 and \
               not isinstance(box[0], (list, tuple)):
                box = [(box[0], box[1]), (box[2], box[3])]
            center, scale = box_to_center_scale(
                box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1],
                bbox_expand=self.bbox_expand,
            )
        
        if center is None or scale is None:
            raise ValueError('Pose prediction requires box or center+scale')
        
        # Get affine transformation
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        
        # Warp image
        model_input = cv2.warpAffine(
            image, trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Apply mask if box provided (ensure only bbox pixels used)
        if box is not None:
            try:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                x1, y1 = int(box[0][0]), int(box[0][1])
                x2, y2 = int(box[1][0]), int(box[1][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
                
                warped_mask = cv2.warpAffine(
                    mask, trans,
                    (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                
                if len(model_input.shape) == 3:
                    warped_mask_3c = np.repeat(warped_mask[:, :, None], 3, axis=2)
                    model_input = np.where(warped_mask_3c == 255, model_input, 0)
            except:
                pass
        
        # Transform for model
        model_input = POSE_TRANSFORM(model_input).unsqueeze(0)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(model_input)
            pose, score = get_final_preds(
                cfg,
                output.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale])
            )
        return pose, score
    
    def _draw_pose(self, frame, pose, score, frame_idx, stage):
        """Draw pose on frame and save to disk."""
        if pose is None:
            return frame
        
        frame_copy = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for kpt in pose:
            frame_copy = DrawHeadPose(kpt, frame_copy)
        
        # Add scores as text
        if score is not None:
            colors = [(255, 0, 0), (170, 0, 255), (255, 255, 0), (0, 255, 170)]
            positions = [
                (frame_copy.shape[1] - 200, 100),
                (frame_copy.shape[1] - 200, 200),
                (frame_copy.shape[1] - 200, 300),
                (frame_copy.shape[1] - 200, 400)
            ]
            
            for i, (pos, color) in enumerate(zip(positions, colors)):
                try:
                    cv2.putText(
                        frame_copy,
                        f"{score[i][0]:.2f}",
                        pos, font, 2, color, 2, cv2.LINE_AA
                    )
                except:
                    pass
        
        # Add frame number
        cv2.putText(
            frame_copy,
            f"Frame: {frame_idx}",
            (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Save image
        save_dir = self.results_dir / f'pose_estimation_{stage}'
        save_dir.mkdir(exist_ok=True)
        
        mean_score = np.mean(score) if score is not None else 0.0
        save_path = save_dir / f'Frame_{frame_idx}_Score_{mean_score:.3f}.jpg'
        cv2.imwrite(str(save_path), frame_copy)
        
        return frame_copy
    
    def _calculate_motion(self, pose, prev_pose, frame_idx, motion_data):
        """Calculate angular velocity and other motion metrics."""
        if pose is None or prev_pose is None:
            motion_data['angles'].append(0.0)
            motion_data['velocities'].append(0.0)
            motion_data['cross_products'].append(0.0)
            return
        
        curr_time = frame_idx * self.total_time / len(motion_data['angles'])
        prev_time = (frame_idx - 1) * self.total_time / len(motion_data['angles'])
        dt = curr_time - prev_time if curr_time > prev_time else 1.0 / self.fps
        
        for curr, prev in zip(pose, prev_pose):
            curr_beak = curr[3]
            prev_beak = prev[3]
            curr_head = curr[0]
            prev_head = prev[0]
            
            if self._is_valid_point(curr_beak) and self._is_valid_point(prev_beak) and \
               self._is_valid_point(curr_head) and self._is_valid_point(prev_head):
                
                theta_rad, theta_deg, cross_prod = calculate_angular_displacement(
                    prev_head[0], prev_head[1], prev_beak[0], prev_beak[1],
                    curr_head[0], curr_head[1], curr_beak[0], curr_beak[1]
                )
                
                ang_vel = theta_rad / dt if dt > 0 else 0.0
                
                motion_data['angles'].append(theta_deg)
                motion_data['velocities'].append(ang_vel)
                motion_data['cross_products'].append(cross_prod)
                return
        
        # Fallback if no valid points
        motion_data['angles'].append(0.0)
        motion_data['velocities'].append(0.0)
        motion_data['cross_products'].append(0.0)
    
    def _append_results(self, results, pose_init, pose_filt, pose_final,
                       score_init, score_final, frame_init, frame_final):
        """Append results to storage."""
        self._append_pose(results['poses_initial'], pose_init)
        self._append_pose(results['poses_filtered'], pose_filt)
        self._append_pose(results['poses_final'], pose_final)
        
        results['scores_initial'].append(score_init)
        results['scores_final'].append(score_final)
        
        results['frames_initial'].append(frame_init)
        results['frames_final'].append(frame_final)
    
    def _append_empty_frame(self, results, frame):
        """Append empty/null results when no pose detected."""
        self._append_pose(results['poses_initial'], None)
        self._append_pose(results['poses_filtered'], None)
        self._append_pose(results['poses_final'], None)
        
        results['scores_initial'].append(None)
        results['scores_final'].append(None)
        
        results['frames_initial'].append(frame)
        results['frames_final'].append(frame)
        
        results['motion_data']['angles'].append(0.0)
        results['motion_data']['velocities'].append(0.0)
        results['motion_data']['cross_products'].append(0.0)
    
    @staticmethod
    def _append_pose(pose_list, pose):
        """Append pose with proper formatting."""
        if pose is None:
            pose_list.append(np.full((PoseEstimator.NUM_KEYPOINTS, 2), np.nan))
            return

        if isinstance(pose, (list, tuple)):
            try:
                arr = np.asarray(pose)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    pose_list.append(arr[0])
                    return
                if arr.ndim == 2 and arr.shape == (PoseEstimator.NUM_KEYPOINTS, 2):
                    pose_list.append(arr)
                    return
            except Exception:
                pass

        if isinstance(pose, np.ndarray):
            if pose.ndim == 3 and pose.shape[0] == 1:
                pose_list.append(pose[0])
                return
            if pose.ndim == 2 and pose.shape == (PoseEstimator.NUM_KEYPOINTS, 2):
                pose_list.append(pose)
                return

        pose_list.append(np.asarray(pose))
    
    @staticmethod
    def _is_valid_point(point):
        """Check if point has valid coordinates."""
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0