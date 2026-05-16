"""
Bird Pose Estimation Pipeline
Main entry point for video processing with object detection and pose estimation.
"""

import sys
import re
import argparse
import json
import cv2
import numpy as np
import torch
from pathlib import Path

from lib.config import cfg, update_config
from models import get_pose_net
from lib.utilities.utilities import get_model_summary
from tools.utils import read_video_frames, write_video_from_frames
from tools.utils import get_yolo_model
from tools.bird_detector import BirdDetector
from tools.pose_estimator import PoseEstimator
from tools.motion_analyzer import MotionAnalyzer
from tools.visualizer import Visualizer
from tools.metrics import evaluate_poses, evaluate_jitter_only


# ── Legacy checkpoint compatibility ──────────────────────────────────
_KEY_RENAMES = {}


def _remap_legacy_keys(state_dict: dict) -> dict:
    """Rename keys in *state_dict* that were saved under old conventions."""
    new_sd = {}
    remapped = []
    for k, v in state_dict.items():
        new_k = _KEY_RENAMES.get(k, k)
        if new_k != k:
            remapped.append(f'{k} → {new_k}')
        new_sd[new_k] = v
    if remapped:
        print(f'  Remapped {len(remapped)} legacy key(s): {remapped}')
    return new_sd


class BirdPoseEstimationPipeline:
    """Main pipeline for bird pose estimation from video."""
    
    def __init__(self, args):
        self.args = args
        self.video_name = Path(args.video).stem
        self.results_dir = self._setup_results_dir()
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.pose_model = self._load_pose_model()
        self.yolo_model = get_yolo_model() if args.write_obj else None
        
        # Load ground truth if provided
        self.gt_annotations = self._load_ground_truth()

    def _setup_results_dir(self):
        """Create and return results directory path."""
        results_root = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else 'videos_experiments'
        results_dir = Path(results_root) / self.video_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def _load_pose_model(self):
        """Load and configure pose estimation model."""
        model = get_pose_net(cfg, is_train=False)
        
        if cfg.TEST.POSE_MODEL_FILE:
            print(f'Loading model from {cfg.TEST.POSE_MODEL_FILE}')
            state = torch.load(cfg.TEST.POSE_MODEL_FILE, map_location='cpu')
            state = _remap_legacy_keys(state)
            info = model.load_state_dict(state, strict=False)
            if info.missing_keys:
                print(f'  WARNING – missing keys:    {info.missing_keys}')
            if info.unexpected_keys:
                print(f'  WARNING – unexpected keys:  {info.unexpected_keys}')
        else:
            print('Warning: No pose model file specified in config')
        
        # ── compute & save model stats (params + GFLOPs) ──
        try:
            img_h, img_w = (cfg.MODEL.IMAGE_SIZE if hasattr(cfg.MODEL, 'IMAGE_SIZE')
                            else (256, 256))
            model.to(self.device)
            dummy = torch.zeros(1, 3, img_h, img_w).to(self.device)
            summary_str = get_model_summary(model, dummy, verbose=False)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            m = re.search(
                r'Total Multiply Adds.*?:\s*([0-9,\.]+)\s*GFLOPs',
                summary_str
            )
            gflops = float(m.group(1).replace(',', '')) if m else 0.0
            stats_path = self.results_dir / 'model_stats.json'
            with open(stats_path, 'w') as _f:
                json.dump({'params_M': round(params_m, 4), 'gflops': round(gflops, 4)}, _f, indent=2)
            print(f'Model stats saved: {params_m:.2f}M params, {gflops:.4f} GFLOPs')
        except Exception as _e:
            print(f'Warning: could not compute model stats: {_e}')

        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_ground_truth(self):
        """Load ground truth annotations if provided."""
        if not self.args.gt:
            return None
        
        try:
            import json
            with open(self.args.gt) as f:
                gt = json.load(f)
            print(f'Loaded GT annotations: {len(gt)} entries')
            return gt
        except Exception as e:
            print(f'Failed to load GT annotations: {e}')
            return None
    
    def run(self):
        """Execute the full pipeline."""
        # Load video
        print(f"Processing video: {self.args.video}")
        frames, fps, total_time = read_video_frames(self.args.video)
        print(f"Video info - Frames: {len(frames)}, FPS: {fps}, Duration: {total_time:.2f}s")
        
        # Step 1: Object Detection
        bboxes = None
        if self.args.write_obj:
            detector = BirdDetector(self.yolo_model, self.results_dir)
            bboxes = detector.detect_and_filter(frames, fps)
        
        # Step 2: Pose Estimation
        if self.args.write_pose or self.args.annotation:
            estimator = PoseEstimator(
                self.pose_model,
                self.device,
                self.results_dir,
                fps,
                total_time,
                filter_type=self.args.filter_type,
                filter_mode=self.args.filter_mode,
                bbox_expand=self.args.bbox_expand,
            )
            
            results = estimator.process_video(
                frames,
                bboxes,
                write_initial=self.args.write_pose
            )
            
            # Step 3: Analysis and Visualization
            self._analyze_and_visualize(results, frames, fps, total_time)

            # Step 3.5: Save annotations if requested
            if self.args.annotation:
                self._save_annotations(results, frames, bboxes)
            
            # Step 4: Evaluation
            if self.gt_annotations:
                self._evaluate(results, fps, bboxes)
            else:
                self._evaluate_jitter(results, fps)
        
        print(f"\nPipeline complete! Results saved to: {self.results_dir}")
    
    def _analyze_and_visualize(self, results, frames, fps, total_time):
        """Analyze motion and create visualizations."""
        has_final = self.args.filter_type.lower() != 'none'

        analyzer = MotionAnalyzer(fps, total_time, self.results_dir)
        motion_initial = analyzer.analyze(results['poses_initial'])
        motion_final = analyzer.analyze(results['poses_final']) if has_final else None

        visualizer = Visualizer(self.results_dir, fps)
        visualizer.create_all_visualizations(
            frames=frames,
            results=results,
            motion_initial=motion_initial,
            motion_final=motion_final,
            has_final=has_final
        )
    
    def _evaluate(self, results, fps, bboxes=None):
        """Evaluate against ground truth."""
        has_final = self.args.filter_type.lower() != 'none'
        metrics = evaluate_poses(
            results['poses_initial'],
            results['poses_final'] if has_final else None,
            self.gt_annotations,
            self.results_dir,
            fps=fps,
            has_final=has_final,
            pred_bboxes=bboxes
        )

        print("\n=== Evaluation Metrics (Initial) ===")
        for key, value in metrics.get('initial', {}).items():
            print(f"{key}: {value:.4f}")

        if has_final:
            print("\n=== Evaluation Metrics (Final) ===")
            for key, value in metrics.get('final', {}).items():
                print(f"{key}: {value:.4f}")

    def _evaluate_jitter(self, results, fps):
        """Compute GT-independent jitter metrics."""
        has_final = self.args.filter_type.lower() != 'none'
        metrics = evaluate_jitter_only(
            results['poses_initial'],
            results['poses_final'] if has_final else None,
            self.results_dir,
            fps=fps,
            has_final=has_final,
        )

        print("\n=== Jitter Metrics (Initial) ===")
        for key, value in metrics.get('initial', {}).items():
            print(f"{key}: {value:.4f}")

        if has_final:
            print("\n=== Jitter Metrics (Final) ===")
            for key, value in metrics.get('final', {}).items():
                print(f"{key}: {value:.4f}")

    def _save_annotations(self, results, frames, bboxes):
        """Save per-frame images and annotations using final poses."""
        project_root = Path(__file__).resolve().parent
        videos_root = project_root / 'videos'
        images_dir = videos_root / 'images' / self.video_name
        annot_path = videos_root / 'annot' / f'{self.video_name}.json'

        images_dir.mkdir(parents=True, exist_ok=True)
        annot_path.parent.mkdir(parents=True, exist_ok=True)

        annotations = []

        for idx, (frame, pose) in enumerate(zip(frames, results['poses_final'])):
            frame_name = f"{self.video_name}_f{idx:06d}.jpg"
            frame_path = images_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            joints = self._pose_to_joints(pose)
            center, scale = self._compute_center_scale(joints)
            bbox = self._bbox_to_points(bboxes, idx)

            annotations.append({
                "image": f"images/{self.video_name}/{frame_name}",
                "animal": "Bird",
                "animal_parent_class": "Bird",
                "animal_class": "Bird",
                "animal_subclass": "Bird",
                "joints_vis": [1, 1, 1, 1],
                "joints": joints,
                "scale": scale,
                "center": center,
                "bbox": bbox,
                "Protocol 3 bird": "validation"
            })

        with open(annot_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=4)

        print(f"Saved frames to: {images_dir}")
        print(f"Saved annotations to: {annot_path}")

    @staticmethod
    def _pose_to_joints(pose):
        """Convert pose array to list of joints with -1.0 for missing values."""
        if pose is None:
            return [[-1.0, -1.0] for _ in range(4)]

        arr = np.asarray(pose, dtype=float)
        if arr.shape != (4, 2):
            return [[-1.0, -1.0] for _ in range(4)]

        joints = []
        for x, y in arr:
            if np.isnan(x) or np.isnan(y):
                joints.append([-1.0, -1.0])
            else:
                joints.append([float(x), float(y)])
        return joints

    @staticmethod
    def _compute_center_scale(joints):
        """Compute center and scale from joints (BirdGaze format)."""
        valid = [(x, y) for x, y in joints if x >= 0 and y >= 0]
        if not valid:
            return [0.0, 0.0], 0.0

        xs, ys = zip(*valid)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center = [float((min_x + max_x) / 2.0), float((min_y + max_y) / 2.0)]
        w = max_x - min_x
        h = max_y - min_y
        scale = float(max(w, h) / 200.0) if max(w, h) > 0 else 0.0

        return center, scale

    @staticmethod
    def _bbox_to_points(bboxes, frame_idx):
        """Convert filtered detection bbox to two-point format."""
        if not bboxes or frame_idx >= len(bboxes) or not bboxes[frame_idx]:
            return [[-1.0, -1.0], [-1.0, -1.0]]

        x1, y1, x2, y2 = bboxes[frame_idx][0]
        return [[float(x1), float(y1)], [float(x2), float(y2)]]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bird Pose Estimation Pipeline')
    
    parser.add_argument('--cfg', type=str, default='experiments/inference.yaml',
                       help='Path to configuration file')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--gt', type=str, default=None,
                       help='Path to ground truth annotations (optional)')
    parser.add_argument('--write_obj', action='store_true',
                       help='Enable object detection and save results')
    parser.add_argument('--write_pose', action='store_true',
                       help='Enable pose estimation and save results')
    parser.add_argument('--annotation', action='store_true',
                       help='Save per-frame images and annotations to videos/annot/<video_title>.json')
    parser.add_argument('--filter_type', type=str, default='custom',
                       choices=['pose', 'kalman', 'custom', 'none'],
                       help='Pose filter type to use')
    parser.add_argument('--filter_mode', type=str, default='offline',
                       choices=['online', 'offline'],
                       help='Filtering mode (optimization/particle support offline only)')
    parser.add_argument('--bbox-expand', type=float, default=1.0,
                       help='Extra multiplier on the detection bounding box '
                            '(1.0 = default 1.25x padding, 1.5 = 1.875x, etc.). '
                            'Increase for birds whose head extends beyond the bbox.')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                       help='Modify config options from command line')
    
    args = parser.parse_args()
    
    # Required by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.prevModelDir = ''
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    update_config(cfg, args)
    
    pipeline = BirdPoseEstimationPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()