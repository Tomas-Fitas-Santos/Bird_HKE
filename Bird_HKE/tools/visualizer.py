"""
Visualization Module
Handles all plotting and video creation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import write_video_from_frames
from .utils import DrawBoxes, calculate_centers


class Visualizer:
    """Creates all visualizations and videos."""
    
    # Matplotlib colors for plotting
    COLORS_MPL = ['blue', 'magenta', 'cyan', 'yellowgreen']
    KEYPOINT_NAMES = ['Head', 'Left Eye', 'Right Eye', 'Beak']
    
    def __init__(self, results_dir, fps):
        self.results_dir = Path(results_dir)
        self.fps = fps
    
    def create_all_visualizations(self, frames, results, motion_initial, motion_final=None, has_final=True):
        """Create all plots and videos."""
        print("\n=== Creating Visualizations ===")

        initial_dir = self.results_dir / 'initial'
        initial_dir.mkdir(parents=True, exist_ok=True)

        final_dir = self.results_dir / 'final'
        if has_final:
            final_dir.mkdir(parents=True, exist_ok=True)

        # Initial outputs
        self._save_pose_video(results.get('frames_initial', []), initial_dir / 'head_pose_estimation.mp4')
        self._plot_keypoint_trajectories_single(results.get('poses_initial', []), 'Initial', initial_dir)
        self._plot_scores_single(results.get('scores_initial', []), 'Key Point Scores (Initial)', initial_dir / 'scores.png')
        if motion_initial is not None:
            self._plot_motion(motion_initial, initial_dir)

        # Final outputs (if applicable)
        if has_final:
            self._save_pose_video(results.get('frames_final', []), final_dir / 'head_pose_estimation.mp4')
            self._plot_keypoint_trajectories_single(results.get('poses_final', []), 'Final', final_dir)
            self._plot_scores_single(results.get('scores_final', []), 'Key Point Scores (Final)', final_dir / 'scores.png')
            if motion_final is not None:
                self._plot_motion(motion_final, final_dir)

        print("Visualizations complete!")
    
    # ==================== Videos ====================
    
    def _save_pose_video(self, frames, save_path):
        """Save pose estimation video."""
        if frames:
            write_video_from_frames(
                frames,
                save_path,
                self.fps
            )
    
    def save_detection_video(self, frames, raw_boxes, filtered_boxes):
        """Save object detection video."""
        video_frames = []
        
        for idx, frame in enumerate(frames):
            frame_vis = frame.copy()
            
            # Draw raw boxes (pink)
            if idx < len(raw_boxes) and raw_boxes[idx]:
                frame_vis = DrawBoxes(
                    frame_vis, raw_boxes[idx], idx,
                    color=(255, 44, 44), thickness=6
                )
            
            # Draw filtered boxes (orange)
            if idx < len(filtered_boxes) and filtered_boxes[idx]:
                frame_vis = DrawBoxes(
                    frame_vis, filtered_boxes[idx], idx,
                    color=(60, 185, 255), thickness=11
                )
            
            video_frames.append(frame_vis)
        
        write_video_from_frames(
            video_frames,
            self.results_dir / 'bird_object_detection.mp4',
            self.fps
        )
    
    # ==================== Detection Plots ====================
    
    def save_detection_plots(self, raw_boxes, filtered_boxes):
        """Plot detection trajectories."""
        centers_raw = calculate_centers(raw_boxes)
        centers_filtered = calculate_centers(filtered_boxes)
        
        frames = range(1, len(filtered_boxes) + 1)
        
        # X coordinates
        x_raw = [c[0] if c else None for c in centers_raw]
        x_filt = [c[0] if c else None for c in centers_filtered]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, x_raw, 'o-', color='blue', label='Raw detections',
                linewidth=2, markersize=8)
        plt.plot(frames, x_filt, '.-', color='orange', label='Filtered detections',
                linewidth=2, markersize=8)
        
        plt.xlabel('Frame', fontsize=21)
        plt.ylabel('X Coordinate', fontsize=21)
        plt.title('YOLO Detection Filtering: X-Trajectory', fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        
        save_path = self.results_dir / 'Detections.png'
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    
    # ==================== Pose Trajectory Plots ====================
    
    def _plot_keypoint_trajectories_single(self, poses, label, out_dir):
        """Plot keypoint trajectories for a single pose sequence."""
        kpt_names = ['Beak', 'Leye', 'Reye', 'Head']
        kpt_indices = [3, 1, 2, 0]

        for name, idx in zip(kpt_names, kpt_indices):
            self._plot_single_trajectory(
                poses,
                idx, name, label,
                out_dir / f'Pose{name}Coordinates.png'
            )
    
    def _plot_single_trajectory(self, poses, kpt_idx, kpt_name, label, save_path):
        """Plot a single trajectory."""
        x_vals = []
        for p in poses:
            try:
                if p is not None and len(p) > kpt_idx and len(p[kpt_idx]) > 0:
                    val = p[kpt_idx][0]
                    if np.isscalar(val):
                        if not np.isnan(val):
                            x_vals.append(val)
                    else:
                        val = np.asarray(val).flatten()[0]
                        if not np.isnan(val):
                            x_vals.append(val)
            except (IndexError, TypeError, ValueError):
                continue

        if not x_vals:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, 'o-', color='green',
                label=f'{label} {kpt_name} trajectory', linewidth=2, markersize=8)

        plt.xlabel('Frame', fontsize=21)
        plt.ylabel('X Coordinate', fontsize=21)
        plt.title(f'{label}: {kpt_name} X-Trajectory', fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    
    # ==================== Score Plots ====================
    
    def _plot_scores_single(self, scores_list, title, save_path):
        """Plot confidence scores over time."""
        # Convert to plottable format
        data = []
        for scores in scores_list:
            if scores is not None:
                frame_scores = []
                for i in range(4):
                    try:
                        frame_scores.append([scores[i][0]])
                    except:
                        frame_scores.append([0.0])
                data.append(frame_scores)
            else:
                data.append([[0.0]] * 4)
        
        if not data:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot each keypoint
        for i in range(4):
            y_values = [d[i][0] for d in data]
            mean_val = np.mean([v for v in y_values if v > 0])
            label = f'{self.KEYPOINT_NAMES[i]} (Avg: {mean_val:.2f})'
            plt.plot(y_values, label=label, color=self.COLORS_MPL[i])
        
        plt.xlabel('Frame', fontsize=21)
        plt.ylabel('Confidence', fontsize=21)
        plt.title(title, fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    
    # ==================== Motion Plots ====================
    
    def _plot_motion(self, motion_data, out_dir):
        """Plot motion analysis results."""
        # Angles
        self._plot_angles(
            motion_data['angles'],
            motion_data.get('angles_unsmoothed', []),
            out_dir / 'Angles.png'
        )

        # Velocities
        self._plot_velocities(
            motion_data['velocities'],
            out_dir / 'Velocities.png'
        )

        # Frequencies
        from scipy.fft import fft, fftfreq

        cross_products = np.array(motion_data['cross_products'])
        n = len(cross_products)

        if n > 0:
            freqs = fftfreq(n, 1/self.fps)
            fft_vals = fft(cross_products)

            # Positive frequencies only
            pos_freqs = freqs[:n//2]
            psd = (np.abs(fft_vals[:n//2]) ** 2) / n

            self._plot_frequencies(
                pos_freqs, psd,
                out_dir / 'FrequenciesTotal.png'
            )
    
    def _plot_angles(self, angles_smooth, angles_unsmooth, save_path):
        """Plot angular displacement."""
        plt.figure(figsize=(10, 6))
        
        if angles_unsmooth:
            plt.plot(angles_unsmooth, '-', color='rebeccapurple',
                    label='Unsmoothed angles', linewidth=2)
        
        plt.plot(angles_smooth, '-', color='orange',
                label='Smoothed angles', linewidth=2)
        
        plt.xlabel('Frame', fontsize=21)
        plt.ylabel('Angle (degrees)', fontsize=21)
        plt.title('Head Rotation Angle Displacement', fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    
    def _plot_velocities(self, velocities, save_path):
        """Plot angular velocities."""
        plt.figure(figsize=(10, 6))
        
        frames = range(1, len(velocities) + 1)
        plt.plot(frames, velocities, '-', color='rebeccapurple', linewidth=2)
        
        # Annotate max velocity
        if velocities:
            max_idx = np.argmax(velocities)
            max_val = velocities[max_idx]
            plt.text(max_idx + 1, max_val, f'{max_val:.2f} rad/s',
                    fontsize=16, verticalalignment='bottom')
        
        plt.xlabel('Frame', fontsize=21)
        plt.ylabel('Angular Velocity (rad/s)', fontsize=21)
        plt.title('Angular Velocity of Head Rotations', fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')
    
    def _plot_frequencies(self, freqs, psd, save_path):
        """Plot frequency spectrum."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(freqs, psd, '-', color='rebeccapurple', linewidth=2)
        
        # Annotate dominant frequency
        if len(psd) > 0:
            max_idx = np.argmax(psd)
            max_freq = freqs[max_idx]
            max_psd = psd[max_idx]
            plt.text(max_freq, max_psd, f'{max_freq:.2f} Hz',
                    fontsize=16, verticalalignment='bottom')
        
        plt.xlabel('Frequency (Hz)', fontsize=21)
        plt.ylabel('Power Spectral Density', fontsize=21)
        plt.title('Frequency Spectrum of Head Rotations', fontsize=23)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')