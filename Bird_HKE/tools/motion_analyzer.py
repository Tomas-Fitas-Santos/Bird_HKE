"""
Motion Analysis Module
Analyzes head motion patterns including angular velocity and frequency.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from .utils import calculate_angular_displacement


class MotionAnalyzer:
    """Analyzes motion patterns from pose sequences."""
    
    def __init__(self, fps, total_time, results_dir):
        self.fps = fps
        self.total_time = total_time
        self.results_dir = results_dir
    
    def analyze(self, poses):
        """
        Analyze motion from pose sequence.
        
        Args:
            poses: List of pose arrays
        
        Returns:
            Dictionary containing motion metrics
        """
        print("\n=== Motion Analysis ===")
        
        motion_data = {
            'angles': [],
            'angles_unsmoothed': [],
            'velocities': [],
            'cross_products': [],
            'frequencies': None,
            'psd': None
        }
        
        prev_pose = None
        
        for frame_idx, pose in enumerate(poses):
            # Calculate angular metrics
            theta_rad, theta_deg, ang_vel, cross_prod = self._calculate_frame_motion(
                pose, prev_pose, frame_idx
            )
            
            motion_data['angles'].append(theta_deg)
            motion_data['velocities'].append(ang_vel)
            motion_data['cross_products'].append(cross_prod)
            
            prev_pose = pose
        
        # Frequency analysis
        if motion_data['cross_products']:
            freqs, psd = self._analyze_frequencies(motion_data['cross_products'])
            motion_data['frequencies'] = freqs
            motion_data['psd'] = psd
        
        # Print summary
        self._print_summary(motion_data)
        
        return motion_data
    
    def _calculate_frame_motion(self, pose, prev_pose, frame_idx):
        """Calculate motion metrics for a single frame."""
        if pose is None or prev_pose is None:
            return 0.0, 0.0, 0.0, 0.0
        
        pose_arr = np.asarray(pose)
        prev_arr = np.asarray(prev_pose)

        if pose_arr.ndim != 2 or prev_arr.ndim != 2:
            return 0.0, 0.0, 0.0, 0.0
        if pose_arr.shape[0] < 4 or prev_arr.shape[0] < 4:
            return 0.0, 0.0, 0.0, 0.0
        if pose_arr.shape[1] < 2 or prev_arr.shape[1] < 2:
            return 0.0, 0.0, 0.0, 0.0

        # Check for NaN
        if np.any(np.isnan(pose_arr)) or np.any(np.isnan(prev_arr)):
            return 0.0, 0.0, 0.0, 0.0
        
        # Time delta
        curr_time = frame_idx * self.total_time / len([pose])
        prev_time = (frame_idx - 1) * self.total_time / len([pose])
        dt = curr_time - prev_time if curr_time > prev_time else 1.0 / self.fps
        
        # Extract head and beak points
        try:
            curr_head = pose_arr[0]
            curr_beak = pose_arr[3]
            prev_head = prev_arr[0]
            prev_beak = prev_arr[3]
            
            # Validate points
            if not self._is_valid_point(curr_head) or not self._is_valid_point(curr_beak) or \
               not self._is_valid_point(prev_head) or not self._is_valid_point(prev_beak):
                return 0.0, 0.0, 0.0, 0.0
            
            # Calculate angular displacement
            theta_rad, theta_deg, cross_prod = calculate_angular_displacement(
                prev_head[0], prev_head[1], prev_beak[0], prev_beak[1],
                curr_head[0], curr_head[1], curr_beak[0], curr_beak[1]
            )
            
            # Angular velocity
            ang_vel = theta_rad / dt if dt > 0 else 0.0
            
            return theta_rad, theta_deg, ang_vel, cross_prod
            
        except Exception as e:
            print(f"Error calculating motion for frame {frame_idx}: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _analyze_frequencies(self, cross_products):
        """Perform frequency analysis using FFT."""
        cross_products = np.array(cross_products)
        n = len(cross_products)
        
        if n == 0:
            return np.array([]), np.array([])
        
        # Compute FFT
        freqs = fftfreq(n, 1.0 / self.fps)
        fft_vals = fft(cross_products)
        
        # Keep only positive frequencies
        pos_freqs = freqs[:n//2]
        psd = (np.abs(fft_vals[:n//2]) ** 2) / n
        
        return pos_freqs, psd
    
    def _print_summary(self, motion_data):
        """Print motion analysis summary."""
        angles = [a for a in motion_data['angles'] if a != 0]
        velocities = [v for v in motion_data['velocities'] if v != 0]
        
        if angles:
            print(f"Angular displacement - Mean: {np.mean(angles):.2f}°, "
                  f"Max: {np.max(angles):.2f}°, Min: {np.min(angles):.2f}°")
        
        if velocities:
            print(f"Angular velocity - Mean: {np.mean(velocities):.4f} rad/s, "
                  f"Max: {np.max(velocities):.4f} rad/s")
        
        if motion_data['psd'] is not None and len(motion_data['psd']) > 0:
            max_idx = np.argmax(motion_data['psd'])
            dominant_freq = motion_data['frequencies'][max_idx]
            print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    
    @staticmethod
    def _is_valid_point(point):
        """Check if point is valid (not zero or NaN)."""
        if point is None:
            return False
        try:
            return not (np.isnan(point[0]) or np.isnan(point[1]) or 
                       point[0] == 0 and point[1] == 0)
        except:
            return False