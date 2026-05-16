"""
Object Detection Module
Handles YOLO-based bird detection and bounding box filtering.
"""

import numpy as np
import cv2
from pathlib import Path
from lib.config import cfg


class BirdDetector:
    """Handles bird detection and bounding box filtering."""
    
    def __init__(self, yolo_model, results_dir):
        self.model = yolo_model
        self.results_dir = Path(results_dir)
        
    def detect_and_filter(self, frames, fps):
        """
        Detect birds in all frames and filter bounding boxes.
        
        Returns:
            List of filtered bounding boxes per frame
        """
        print("\n=== Object Detection ===")
        
        # Initial detection
        raw_boxes, scores = self._detect_all_frames(frames)
        
        # Filter boxes
        filtered_boxes = self._filter_boxes(raw_boxes, scores)
        
        # Visualize and save
        self._save_detection_results(frames, raw_boxes, filtered_boxes, fps)

        # Debug summary
        raw_with_box = sum(1 for b in raw_boxes if b)
        filtered_with_box = sum(1 for b in filtered_boxes if b)
        total_frames = len(frames)
        all_scores = [float(s) for frame_scores in scores for s in frame_scores]
        if all_scores:
            max_score = float(np.max(all_scores))
            mean_score = float(np.mean(all_scores))
        else:
            max_score = 0.0
            mean_score = 0.0

        print(
            f"Detection summary: frames={total_frames}, "
            f"raw_boxes={raw_with_box}, filtered_boxes={filtered_with_box}, "
            f"max_score={max_score:.3f}, mean_score={mean_score:.3f}"
        )
        
        return filtered_boxes
    
    def _detect_all_frames(self, frames):
        """Run YOLO detection on all frames."""
        boxes_per_frame = []
        scores_per_frame = []
        
        for i, frame in enumerate(frames):
            print(f"Detecting frame {i+1}/{len(frames)}", end='\r')
            
            results = self.model.predict(frame, save=False, conf=0.01)
            boxes = results[0].boxes
            
            # Convert to standard format
            box_coords = boxes.xyxy.cpu().numpy()
            converted = [(b[0], b[1], b[2], b[3]) for b in box_coords]
            
            boxes_per_frame.append(converted)
            scores_per_frame.append(boxes.conf.cpu().numpy())
        
        print()  # New line after progress
        return boxes_per_frame, scores_per_frame
    
    def _filter_boxes(self, all_boxes, all_scores):
        """
        Filter bounding boxes using IoU-based trajectory smoothing.
        
        Steps:
        1. Filter by confidence threshold
        2. Find smooth trajectories using IoU
        3. Complete trajectories with interpolation
        """
        # Step 1: Filter by score
        main_boxes = self._filter_by_score(all_boxes, all_scores)
        
        # Step 2: Find smooth trajectory
        trajectory_idx = self._find_smooth_trajectory(main_boxes)
        
        # Step 3: Complete trajectory
        completed = self._complete_trajectory(main_boxes, all_boxes, trajectory_idx)
        
        # Step 4: Fill endpoints and interpolate
        filled = self._fill_endpoints(completed)
        interpolated = self._interpolate_gaps(filled)
        
        return interpolated
    
    def _filter_by_score(self, all_boxes, all_scores, min_threshold=0.4):
        """Keep only boxes above score threshold."""
        threshold = min_threshold
        main_boxes = []
        
        for boxes, scores in zip(all_boxes, all_scores):
            frame_boxes = []
            for box, score in zip(boxes, scores):
                if score > threshold:
                    frame_boxes.append(box)
                    break
            main_boxes.append(frame_boxes)
        
        # Adjust threshold if too few boxes
        while self._box_percentage(main_boxes) < 0.4 and threshold > 0.1:
            threshold -= 0.05
            print(f"Adjusting threshold to {threshold:.2f}")
            main_boxes = []
            for boxes, scores in zip(all_boxes, all_scores):
                frame_boxes = []
                for box, score in zip(boxes, scores):
                    if score > threshold:
                        frame_boxes.append(box)
                        break
                main_boxes.append(frame_boxes)
        
        return main_boxes
    
    def _find_smooth_trajectory(self, boxes, iou_threshold=0.9):
        """Find the smoothest trajectory based on IoU between consecutive frames."""
        sequences = []
        
        i = 0
        while i < len(boxes):
            if boxes[i]:
                j = i + 1
                while j < len(boxes):
                    if boxes[j]:
                        iou = self._calculate_iou(boxes[i][0], boxes[j][0])
                        if iou > iou_threshold:
                            if not sequences or sequences[-1][1] != i:
                                sequences.append([i, j])
                            else:
                                sequences[-1][1] = j
                        break
                    j += 1
            i += 1
        
        # Find longest sequence
        max_len = 0
        best_seq = None
        for seq in sequences:
            length = seq[1] - seq[0]
            if length > max_len:
                max_len = length
                best_seq = seq
        
        return best_seq if best_seq else [0, len(boxes)-1]
    
    def _complete_trajectory(self, main_boxes, all_boxes, trajectory_idx, iou_threshold=0.7):
        """Complete trajectory by finding matching boxes before/after main sequence."""
        completed = main_boxes.copy()
        start, end = trajectory_idx
        
        # Fill backwards
        for i in range(start, 0, -1):
            if completed[i]:
                for offset in range(1, start + 1):
                    if i - offset >= 0 and all_boxes[i - offset]:
                        best_box = self._find_best_matching_box(
                            completed[i][0],
                            all_boxes[i - offset],
                            iou_threshold
                        )
                        if best_box:
                            completed[i - offset] = [best_box]
                            break
        
        # Fill forwards
        for i in range(start, len(completed) - 1):
            if completed[i]:
                for offset in range(1, len(completed) - i):
                    if i + offset < len(completed) and all_boxes[i + offset]:
                        best_box = self._find_best_matching_box(
                            completed[i][0],
                            all_boxes[i + offset],
                            iou_threshold
                        )
                        if best_box:
                            completed[i + offset] = [best_box]
                            break
        
        return completed
    
    def _fill_endpoints(self, boxes):
        """Fill missing start/end frames."""
        filled = boxes.copy()
        
        try:
            max_fill = int(cfg.DETECTION.MAX_ENDPOINT_FILL)
        except:
            max_fill = 0
        
        if max_fill <= 0:
            return filled
        
        # Fill start
        if not filled[0]:
            first_box = next((b for b in filled if b), None)
            if first_box:
                idx = filled.index(first_box)
                for i in range(max(0, idx - max_fill), idx):
                    filled[i] = first_box
        
        # Fill end
        if not filled[-1]:
            last_box = next((b for b in reversed(filled) if b), None)
            if last_box:
                idx = len(filled) - 1 - filled[::-1].index(last_box)
                for i in range(idx + 1, min(len(filled), idx + 1 + max_fill)):
                    filled[i] = last_box
        
        return filled
    
    def _interpolate_gaps(self, boxes):
        """Interpolate small gaps in trajectory."""
        try:
            max_gap = int(cfg.DETECTION.MAX_INTERPOLATION_GAP)
        except:
            max_gap = 0
        
        if max_gap <= 0:
            return boxes
        
        interpolated = boxes.copy()
        prev_box = None
        prev_idx = None
        
        for i, box in enumerate(interpolated):
            if box:
                if prev_box and prev_idx is not None:
                    gap = i - prev_idx - 1
                    if 0 < gap <= max_gap:
                        # Linear interpolation
                        for j in range(prev_idx + 1, i):
                            alpha = (j - prev_idx) / (i - prev_idx)
                            interp_box = tuple(
                                int(prev_box[0][k] + alpha * (box[0][k] - prev_box[0][k]))
                                for k in range(4)
                            )
                            interpolated[j] = [interp_box]
                
                prev_box = box
                prev_idx = i
        
        return interpolated
    
    def _save_detection_results(self, frames, raw_boxes, filtered_boxes, fps):
        """Save detection visualizations and plots."""
        from .visualizer import Visualizer
        
        viz = Visualizer(self.results_dir, fps)
        viz.save_detection_plots(raw_boxes, filtered_boxes)
        viz.save_detection_video(frames, raw_boxes, filtered_boxes)
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _find_best_matching_box(self, ref_box, candidate_boxes, iou_threshold):
        """Find the box with highest IoU above threshold."""
        best_box = None
        best_iou = iou_threshold
        
        for box in candidate_boxes:
            iou = self._calculate_iou(ref_box, box)
            if iou > best_iou:
                best_iou = iou
                best_box = box
        
        return best_box
    
    def _box_percentage(self, boxes):
        """Calculate percentage of frames with boxes."""
        count = sum(1 for b in boxes if b)
        return count / len(boxes) if boxes else 0