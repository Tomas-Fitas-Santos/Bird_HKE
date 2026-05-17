"""
Pose Filtering Module
Implements validation smoothing and One-Euro-based custom filtering.
"""

import numpy as np
import copy
import math
from .utils import calculate_distance, calculate_angular_displacement

class Validation_Smoothing_Filter:
    """Filters pose estimations to remove drift and errors."""
    
    def __init__(self):
        self.drift_count = 0
        self.flicker_count = 0
        
        # Adaptive thresholds
        self.accum_dist_threshold = 0.0
        self.accum_score_threshold = 0.0
        
        # Flag for tracking drift state
        self.was_removed = False
    
    def filter_pose(self, pose, prev_pose, score, frame_idx):
        """
        Filter pose to remove drift errors.
        
        Returns:
            Filtered pose and score (or None if pose should be removed)
        """
        if pose is None:
            return None, score
        
        # First frame - no filtering
        if prev_pose is None:
            return pose, score
        
        pose_copy = copy.deepcopy(pose)
        
        # Calculate metrics
        metrics = self._calculate_metrics(pose_copy, prev_pose, score, frame_idx)
        
        # Update adaptive thresholds
        self._update_thresholds(metrics, frame_idx)
        
        # Check for drift errors
        if self._should_remove_pose(metrics):
            self.drift_count += 1
            return None, score
        
        return pose_copy, score
    
    def smooth_pose(self, pose, prev_pose, score):
        """Apply smoothing to reduce flicker."""
        if pose is None:
            return None, score
        
        # Smoothing parameters
        angle_threshold = 7  # degrees
        k = 0.5
        alpha = 0.15
        
        smoothed_pose = copy.deepcopy(pose)
        
        if prev_pose is not None and pose is not None:
            for kpts, prev_kpts in zip(smoothed_pose, prev_pose):
                # Defensive: ensure at least 4 keypoints in both
                if len(kpts) < 4 or len(prev_kpts) < 4:
                    continue
                curr_beak = kpts[3]
                prev_beak = prev_kpts[3]
                curr_head = kpts[0]
                prev_head = prev_kpts[0]

                # Check if points are valid
                if self._is_valid_point(curr_beak) and self._is_valid_point(prev_beak) and \
                   self._is_valid_point(curr_head) and self._is_valid_point(prev_head):

                    # Calculate angular displacement
                    _, theta_deg, _ = calculate_angular_displacement(
                        prev_head[0], prev_head[1], prev_beak[0], prev_beak[1],
                        curr_head[0], curr_head[1], curr_beak[0], curr_beak[1]
                    )

                    # Calculate distances
                    head_dist = calculate_distance(
                        prev_head[0], prev_head[1], curr_head[0], curr_head[1]
                    )
                    beak_dist = calculate_distance(
                        prev_beak[0], prev_beak[1], curr_beak[0], curr_beak[1]
                    )
                    head_beak_dist = calculate_distance(
                        curr_head[0], curr_head[1], curr_beak[0], curr_beak[1]
                    )

                    dist_threshold = k * np.sqrt(head_beak_dist)

                    # Apply smoothing if small angle and movement
                    if theta_deg < angle_threshold and \
                       (head_dist < dist_threshold or beak_dist < dist_threshold):
                        self.flicker_count += 1
                        for i in range(4):
                            kpts[i] = alpha * kpts[i] + (1 - alpha) * prev_kpts[i]
                        break

        return smoothed_pose, score
    
    def _calculate_metrics(self, pose, prev_pose, score, frame_idx):
        """Calculate all metrics needed for filtering. Expects pose and prev_pose as (N,2) arrays/lists."""
        metrics = {
            'distances': [],
            'scores': [],
            'score_avg': 0.0,
            'dist_avg': 0.0,
            'head_beak_dist': 0.0,
            'head_leye_dist': 0.0,
            'head_reye_dist': 0.0
        }

        # Defensive: skip if pose or prev_pose is not a list/array of keypoints
        if not (isinstance(pose, (list, np.ndarray)) and isinstance(prev_pose, (list, np.ndarray))):
            return metrics
        if len(pose) < 4 or len(prev_pose) < 4:
            return metrics

        # Keypoint distances
        for i in range(4):
            try:
                dist = calculate_distance(
                    pose[i][0], pose[i][1],
                    prev_pose[i][0], prev_pose[i][1]
                )
            except Exception:
                dist = 0.0
            metrics['distances'].append(dist)

        # Scores
        if score is not None:
            for i in range(4):
                try:
                    metrics['scores'].append(score[i][0])
                except Exception:
                    try:
                        metrics['scores'].append(float(score[i]))
                    except Exception:
                        metrics['scores'].append(0.0)

        # Inter-keypoint distances
        try:
            metrics['head_beak_dist'] = calculate_distance(
                pose[0][0], pose[0][1], pose[3][0], pose[3][1]
            )
            metrics['head_leye_dist'] = calculate_distance(
                pose[0][0], pose[0][1], pose[1][0], pose[1][1]
            )
            metrics['head_reye_dist'] = calculate_distance(
                pose[0][0], pose[0][1], pose[2][0], pose[2][1]
            )
        except Exception:
            pass

        # Averages
        if metrics['scores']:
            metrics['score_avg'] = np.mean(metrics['scores'])
        if metrics['distances']:
            metrics['dist_avg'] = np.mean(metrics['distances'])
        return metrics
    
    def _update_thresholds(self, metrics, frame_idx):
        """Update adaptive thresholds based on running averages."""
        # Running average for scores
        self.accum_score_threshold += (
            metrics['score_avg'] - self.accum_score_threshold
        ) / (frame_idx + 1)
        
        # Running average for distances
        self.accum_dist_threshold += (
            metrics['dist_avg'] - self.accum_dist_threshold
        ) / (frame_idx + 1)
    
    def _should_remove_pose(self, metrics):
        """Determine if pose should be removed based on metrics."""
        # Minimum thresholds
        min_score = 0.45  # Slightly higher to be more strict
        
        # Adaptive thresholds - more sensitive to drift
        min_drift_dist = self._exponential_threshold(
            self.accum_dist_threshold, alpha=0.003, max_val=300  # Lower alpha and max_val
        )
        min_drift_score = self._exponential_threshold(
            self.accum_score_threshold, alpha=1.2, max_val=1.0  # Lower alpha
        )
        min_tp_score = self._exponential_threshold(
            self.accum_score_threshold, alpha=2.0, max_val=1.0  # Lower alpha
        )
        
        # Check for drift
        if any(d > min_drift_dist for d in metrics['distances']):
            if any(s < min_drift_score for s in metrics['scores']):
                self.was_removed = True
            else:
                self.was_removed = False
        
        # Check for true positives (override removal)
        if metrics['score_avg'] > min_tp_score:
            self.was_removed = False
        
        if self.was_removed:
            return True
        
        # Check eye distance symmetry
        dist_diff = abs(metrics['head_leye_dist'] - metrics['head_reye_dist'])
        smallest_dist = min(metrics['head_leye_dist'], metrics['head_reye_dist'])
        if dist_diff > smallest_dist * 2:
            return True
        
        # Check anatomical constraints (eyes shouldn't be farther than beak)
        if (metrics['head_leye_dist'] > metrics['head_beak_dist'] or
            metrics['head_reye_dist'] > metrics['head_beak_dist']):
            return True
        
        # Check minimum scores
        if any(s < min_score for s in metrics['scores']):
            return True
        
        return False
    
    @staticmethod
    def _exponential_threshold(value, alpha, max_val):
        """Calculate exponential threshold."""
        return max_val * (1 - np.exp(-alpha * value))
    
    @staticmethod
    def _is_valid_point(point):
        """Check if point has valid coordinates."""
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0
    

# --------------------------------------------------------------------------------------


class GeometricSkeletonCorrector:
    """
    Geometry-aware correction of 4-keypoint bird head poses after One-Euro smoothing.

    Keypoint indices (must match training convention):
        0 — crown  (top of skull)
        1 — left eye
        2 — right eye
        3 — beak tip

    ── Geometric model ──────────────────────────────────────────────────────────────
    Crown and beak define the *head axis* v_axis = beak - crown.
    The eye midpoint lies on that axis at fractional position t_eye (learned).
    Each eye is displaced perpendicularly from that midpoint by ±d_perp (learned):

        mid_eye   = crown   +  t_eye  * v_axis
        left_eye  = mid_eye +  d_pr   * v_perp    (v_perp ⊥ v_axis, CCW 90°)
        right_eye = mid_eye -  d_pr   * v_perp

    where d_pr = d_perp_ratio * ‖v_axis‖.  Both t_eye and d_perp_ratio are
    estimated automatically from the most reliable frames.

    ── Reliability ──────────────────────────────────────────────────────────────────
    Each joint gets a per-frame reliability score ∈ [0,1] from:
        • detection confidence (primary signal)
        • temporal velocity outlier: a joint displaced much more than the median
          head motion across the frame is penalised exponentially

    ── Correction strategies ────────────────────────────────────────────────────────
    n_reliable   strategy
    ──────────   ────────────────────────────────────────────────────────────────────
    4            passthrough – all joints reliable, no change
    3            weighted Procrustes → reconstruct the one bad joint
    2 (eyes)     head axis from temporal EMA; reconstruct crown & beak
    2 (cr+bk)    reconstruct eyes from axis + learned template ratios
    2 (mixed)    symmetry + axis alignment + temporal scale/angle
    1            temporal pose extrapolation (EMA of head center/scale/angle)
    0            hold last valid corrected pose
    """

    KP_CROWN = 0
    KP_LEYE  = 1
    KP_REYE  = 2
    KP_BEAK  = 3
    N        = 4

    def __init__(
        self,
        conf_thresh    = 0.60,   # min reliability score to treat a joint as valid
        vel_thresh     = 25.0,   # px/frame – excess displacement starts penalty here
        template_conf  = 0.60,   # min per-joint reliability to include frame in template
        corr_blend     = 0.75,   # weight toward geometric reconstruction (vs raw filtered)
        temporal_decay = 0.85,   # EMA decay for temporal head-pose state
    ):
        self.conf_thresh    = float(conf_thresh)
        self.vel_thresh     = float(vel_thresh)
        self.template_conf  = float(template_conf)
        self.corr_blend     = float(corr_blend)
        self.temporal_decay = float(temporal_decay)

        # Learned canonical shape: crown→(0,0), beak→(0,1), scale=crown-beak dist
        self._canonical    = None   # (N, 2)
        self._t_eye        = None   # float: fractional position of eye midpoint along axis
        self._d_perp_ratio = None   # float: lateral eye offset / crown-beak distance

        # Temporal EMA of head pose state
        self._t_center  = None   # (2,) – head centroid
        self._t_angle   = None   # float – crown→beak direction (atan2(vx,vy))
        self._t_scale   = None   # float – crown-beak distance
        self._last_good = None   # (N,2) – last fully corrected pose

    # ── Public ─────────────────────────────────────────────────────────────────────

    def correct_sequence(self, poses, scores=None):
        """
        Offline geometry-aware correction pass over a complete sequence.

        Args:
            poses  : list[T] of ndarray (N,2) – already One-Euro filtered
            scores : list[T] of per-joint confidences (optional)

        Returns:
            corrected : list[T] of ndarray (N,2)
        """
        if not poses:
            return []

        T = len(poses)
        if scores is None:
            scores = [None] * T

        poses_a  = [np.asarray(p, dtype=np.float32) if p is not None else None
                    for p in poses]
        scores_a = [self._parse_scores(s) for s in scores]

        rel = self._estimate_reliability(poses_a, scores_a)   # (T, N) ∈ [0,1]
        self._build_canonical(poses_a, rel)

        self._reset_temporal()
        corrected = []
        for t in range(T):
            p = poses_a[t]
            if p is None:
                corrected.append(None)
                continue
            corrected.append(self._correct_frame(p.copy(), rel[t]))

        return corrected

    # ── Reliability ────────────────────────────────────────────────────────────────

    def _parse_scores(self, score):
        """Normalise any score structure to a (N,) float32 array."""
        if score is None:
            return None
        out = np.ones(self.N, dtype=np.float32)
        for j in range(self.N):
            try:
                v = score[j]
                out[j] = float(v[0] if hasattr(v, '__len__') else v)
            except Exception:
                pass
        return out

    def _estimate_reliability(self, poses, scores):
        """
        Returns reliability[T, N] ∈ [0,1] combining:
            • per-joint detection confidence
            • velocity-outlier penalty (joint moved much more than the head)
        """
        T   = len(poses)
        rel = np.ones((T, self.N), dtype=np.float32)

        for t in range(T):
            if poses[t] is None:
                rel[t] = 0.0
            elif scores[t] is not None:
                rel[t] *= np.clip(scores[t], 0.0, 1.0)

        for t in range(1, T):
            if poses[t] is None or poses[t - 1] is None:
                continue
            disp        = np.linalg.norm(poses[t] - poses[t - 1], axis=-1)  # (N,)
            head_motion = np.median(disp) + 1e-6
            for j in range(self.N):
                ref = max(self.vel_thresh, 3.0 * head_motion)
                if disp[j] > ref:
                    excess     = disp[j] / ref
                    rel[t, j] *= float(np.exp(-0.5 * (excess - 1.0)))

        return np.clip(rel, 0.0, 1.0)

    # ── Canonical template ─────────────────────────────────────────────────────────

    def _build_canonical(self, poses, rel):
        """
        Estimate the normalised canonical head shape from high-reliability frames.

        Canonical frame definition:
            crown → (0, 0),   beak → (0, 1),   scale = crown-beak distance.

        The rotation R = [[vy, -vx], [vx, vy]] maps v_hat=(beak-crown)/scale → (0,1).
        """
        shapes = []
        for t, p in enumerate(poses):
            if p is None:
                continue
            if np.all(rel[t] >= self.template_conf):
                s = self._normalize_pose(p)
                if s is not None:
                    shapes.append(s)

        # Fallback: top 20% of frames ranked by mean reliability
        if not shapes:
            order = sorted(
                (t for t, p in enumerate(poses) if p is not None),
                key=lambda i: float(np.mean(rel[i])),
                reverse=True,
            )
            for t in order[:max(1, len(order) // 5)]:
                s = self._normalize_pose(poses[t])
                if s is not None:
                    shapes.append(s)

        if not shapes:
            return

        self._canonical = np.mean(shapes, axis=0).astype(np.float32)
        c    = self._canonical
        v_ax = c[self.KP_BEAK] - c[self.KP_CROWN]   # ≈ (0,1) in canonical frame
        ax_len = float(np.linalg.norm(v_ax))
        if ax_len < 1e-6:
            return

        mid_eye = 0.5 * (c[self.KP_LEYE] + c[self.KP_REYE])
        self._t_eye = float(
            np.dot(mid_eye - c[self.KP_CROWN], v_ax) / (ax_len * ax_len)
        )
        v_perp_unit        = np.array([-v_ax[1], v_ax[0]]) / ax_len
        self._d_perp_ratio = float(
            np.dot(c[self.KP_LEYE] - mid_eye, v_perp_unit) / ax_len
        )

    def _normalize_pose(self, p):
        """
        Return pose in canonical frame (crown→origin, beak→(0,1)).
        R = [[vy,-vx],[vx,vy]] rotates v_hat=(vx,vy) onto the y-axis (0,1).
        Returns None if degenerate (crown-beak too small).
        """
        crown = p[self.KP_CROWN].astype(np.float32)
        beak  = p[self.KP_BEAK ].astype(np.float32)
        v     = beak - crown
        scale = float(np.linalg.norm(v))
        if scale < 2.0:
            return None
        vx, vy = v / scale
        R = np.array([[vy, -vx], [vx, vy]], dtype=np.float32)
        return ((R @ (p - crown).T) / scale).T   # (N, 2)

    def _denormalize(self, norm, crown_pos, v_axis):
        """
        Inverse of _normalize_pose.
            norm      : (N,2) canonical coordinates
            crown_pos : (2,)  image position of crown
            v_axis    : (2,)  crown→beak vector in image space (encodes scale + angle)
        """
        scale = float(np.linalg.norm(v_axis))
        if scale < 1e-6:
            return None
        vx, vy = v_axis / scale
        R     = np.array([[vy, -vx], [vx, vy]], dtype=np.float32)
        R_inv = R.T
        return ((R_inv @ (norm * scale).T).T + crown_pos).astype(np.float32)

    # ── Temporal EMA state ─────────────────────────────────────────────────────────

    def _reset_temporal(self):
        self._t_center  = None
        self._t_angle   = None
        self._t_scale   = None
        self._last_good = None

    def _update_temporal(self, pose, mask):
        """EMA update of head centroid, crown-beak scale, and axis angle."""
        alpha = 1.0 - self.temporal_decay
        pts   = pose[mask]
        if len(pts) == 0:
            return
        center = pts.mean(axis=0)

        scale = None
        angle = None

        if mask[self.KP_CROWN] and mask[self.KP_BEAK]:
            v = pose[self.KP_BEAK] - pose[self.KP_CROWN]
            s = float(np.linalg.norm(v))
            if s > 1e-6:
                scale = s
                angle = float(np.arctan2(v[0], v[1]))

        elif mask[self.KP_LEYE] and mask[self.KP_REYE]:
            v_eye = pose[self.KP_REYE] - pose[self.KP_LEYE]
            eye_d = float(np.linalg.norm(v_eye))
            if eye_d > 1e-6 and self._d_perp_ratio and abs(self._d_perp_ratio) > 0.01:
                scale = eye_d / (2.0 * abs(self._d_perp_ratio))
                # Two perpendicular candidates; pick the one matching temporal angle
                v_eu   = v_eye / eye_d
                cand1  = float(np.arctan2( v_eu[1], -v_eu[0]))
                cand2  = float(np.arctan2(-v_eu[1],  v_eu[0]))
                if self._t_angle is not None:
                    d1 = abs(((cand1 - self._t_angle) + np.pi) % (2 * np.pi) - np.pi)
                    d2 = abs(((cand2 - self._t_angle) + np.pi) % (2 * np.pi) - np.pi)
                    angle = cand1 if d1 <= d2 else cand2
                else:
                    angle = cand1

        if self._t_center is None:
            self._t_center = center.copy()
            self._t_scale  = scale if scale is not None else 50.0
            self._t_angle  = angle if angle is not None else 0.0
        else:
            self._t_center = self.temporal_decay * self._t_center + alpha * center
            if scale is not None:
                self._t_scale = self.temporal_decay * self._t_scale + alpha * scale
            if angle is not None:
                da = ((angle - self._t_angle) + np.pi) % (2 * np.pi) - np.pi
                self._t_angle += alpha * da

    def _get_temporal_pose(self):
        """
        Reconstruct full skeleton from current temporal state.
        The reconstructed centroid matches _t_center exactly.
        """
        if self._canonical is None or self._t_scale is None:
            return self._last_good.copy() if self._last_good is not None else None

        scale  = self._t_scale
        angle  = self._t_angle if self._t_angle is not None else 0.0
        center = self._t_center

        v_axis = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * scale
        vx, vy = v_axis / scale
        R      = np.array([[vy, -vx], [vx, vy]], dtype=np.float32)
        R_inv  = R.T
        can_c  = (self._canonical - self._canonical.mean(axis=0)).astype(np.float32)
        return ((R_inv @ (can_c * scale).T).T + center).astype(np.float32)

    # ── Per-frame correction ───────────────────────────────────────────────────────

    def _correct_frame(self, p, r):
        """Correct a single frame: p=(N,2) pose, r=(N,) reliability scores."""
        mask  = r >= self.conf_thresh   # (N,) bool
        n_rel = int(mask.sum())

        if n_rel > 0:
            self._update_temporal(p, mask)

        if n_rel == self.N:
            # All reliable – passthrough
            self._last_good = p.copy()
            return p

        bad = ~mask
        out = p.copy()

        # ── 3 / 4 reliable: weighted Procrustes ────────────────────────────────
        if n_rel >= 3:
            recon = self._procrustes_reconstruct(p, mask)
            if recon is not None:
                out[bad] = recon[bad]
            self._last_good = out.copy()
            return out

        # ── 2 reliable: geometry + temporal blend ──────────────────────────────
        if n_rel == 2:
            rel_idx  = list(np.where(mask)[0])
            recon    = self._two_joint_reconstruct(p, rel_idx)
            temporal = self._get_temporal_pose()
            if recon is not None and temporal is not None:
                blend  = self.corr_blend * recon + (1.0 - self.corr_blend) * temporal
                out[bad] = blend[bad]
            elif recon is not None:
                out[bad] = recon[bad]
            elif temporal is not None:
                out[bad] = temporal[bad]
            self._last_good = out.copy()
            return out

        # ── ≤ 1 reliable: temporal extrapolation ───────────────────────────────
        temporal = self._get_temporal_pose()
        if temporal is not None:
            out[bad] = temporal[bad]
        elif self._last_good is not None:
            out[bad] = self._last_good[bad]
        return out

    # ── Reconstruction helpers ─────────────────────────────────────────────────────

    def _procrustes_reconstruct(self, p, mask):
        """
        Fit the canonical template to the reliable joints of p via Procrustes.
        Returns full (N,2) reconstruction or None.
        """
        if self._canonical is None:
            return None
        src = self._canonical[mask]
        dst = p[mask]
        if len(src) < 2:
            return None

        c_src = src.mean(axis=0)
        c_dst = dst.mean(axis=0)
        A = src - c_src
        B = dst - c_dst
        s_src = float(np.sqrt((A ** 2).sum()))
        s_dst = float(np.sqrt((B ** 2).sum()))
        if s_src < 1e-6:
            return None
        s = s_dst / s_src

        U, _, Vt = np.linalg.svd(A.T @ B)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        T     = c_dst - s * (R @ c_src)
        recon = (s * (R @ self._canonical.T).T + T).astype(np.float32)
        return recon

    def _two_joint_reconstruct(self, p, rel_idx):
        """Dispatch to the appropriate 2-joint reconstruction strategy."""
        if self._canonical is None or self._t_scale is None:
            return None

        pair = frozenset(rel_idx)
        out  = p.copy()

        if pair == frozenset([self.KP_LEYE, self.KP_REYE]):
            return self._from_eyes(p, out)

        if pair == frozenset([self.KP_CROWN, self.KP_BEAK]):
            return self._from_crown_beak(p, out)

        if self.KP_CROWN in pair:
            eye = rel_idx[0] if rel_idx[1] == self.KP_CROWN else rel_idx[1]
            return self._from_crown_and_eye(p, eye, out)

        if self.KP_BEAK in pair:
            eye = rel_idx[0] if rel_idx[1] == self.KP_BEAK else rel_idx[1]
            return self._from_beak_and_eye(p, eye, out)

        return None

    # ── Specific 2-joint strategies ────────────────────────────────────────────────

    def _from_crown_beak(self, p, out):
        """
        Known: crown + beak.
        Reconstruct eyes using template ratios (t_eye, d_perp_ratio).
        """
        crown  = p[self.KP_CROWN].astype(np.float32)
        beak   = p[self.KP_BEAK ].astype(np.float32)
        v_axis = beak - crown
        scale  = float(np.linalg.norm(v_axis))
        if scale < 1e-6:
            return None

        t_eye  = self._t_eye        if self._t_eye        is not None else 0.45
        d_rat  = self._d_perp_ratio if self._d_perp_ratio is not None else 0.20

        mid_eye = crown + t_eye * v_axis
        v_perp  = np.array([-v_axis[1], v_axis[0]], dtype=np.float32) / scale
        out[self.KP_LEYE] = (mid_eye + d_rat * scale * v_perp).astype(np.float32)
        out[self.KP_REYE] = (mid_eye - d_rat * scale * v_perp).astype(np.float32)
        return out

    def _from_eyes(self, p, out):
        """
        Known: both eyes.
        Reconstruct crown & beak using temporal axis direction + canonical scale ratio.
        Temporal angle disambiguates the two possible perpendicular directions.
        """
        leye    = p[self.KP_LEYE].astype(np.float32)
        reye    = p[self.KP_REYE].astype(np.float32)
        mid_eye = 0.5 * (leye + reye)
        eye_vec = reye - leye
        eye_d   = float(np.linalg.norm(eye_vec))
        if eye_d < 1e-6:
            return None

        d_rat = self._d_perp_ratio if self._d_perp_ratio is not None else 0.20
        if abs(d_rat) < 0.01:
            return None   # degenerate view: eyes are on the head-axis, can't solve

        # Crown-beak scale from eye distance and canonical lateral ratio
        cb_scale = eye_d / (2.0 * abs(d_rat))
        t_eye    = self._t_eye if self._t_eye is not None else 0.45

        # Head-axis unit vector: perpendicular to eye vector.
        # Two candidates; temporal angle picks the correct one.
        v_eu   = eye_vec / eye_d
        cand1  = np.array([ v_eu[1], -v_eu[0]], dtype=np.float32)   # 90° CW
        cand2  = np.array([-v_eu[1],  v_eu[0]], dtype=np.float32)   # 90° CCW
        if self._t_angle is not None:
            t_unit = np.array([np.sin(self._t_angle), np.cos(self._t_angle)],
                              dtype=np.float32)
            v_axis_unit = cand1 if float(np.dot(cand1, t_unit)) >= float(np.dot(cand2, t_unit)) else cand2
        else:
            v_axis_unit = cand1

        v_axis             = v_axis_unit * cb_scale
        crown              = mid_eye - t_eye * v_axis
        out[self.KP_CROWN] = crown.astype(np.float32)
        out[self.KP_BEAK ] = (crown + v_axis).astype(np.float32)
        return out

    def _from_crown_and_eye(self, p, eye_idx, out):
        """
        Known: crown + one eye.
        Strategy: estimate mid_eye from both (axis projection) and (observed eye),
        blend them, then place the other eye by symmetry and beak along axis.
        """
        other = self.KP_REYE if eye_idx == self.KP_LEYE else self.KP_LEYE
        sign  = 1.0 if eye_idx == self.KP_LEYE else -1.0

        crown    = p[self.KP_CROWN].astype(np.float32)
        this_eye = p[eye_idx      ].astype(np.float32)
        scale    = self._t_scale        if self._t_scale        is not None else 50.0
        angle    = self._t_angle        if self._t_angle        is not None else 0.0
        t_eye    = self._t_eye          if self._t_eye          is not None else 0.45
        d_rat    = self._d_perp_ratio   if self._d_perp_ratio   is not None else 0.20
        if abs(d_rat) < 0.01:
            d_rat = 0.20

        v_ax   = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * scale
        v_perp = np.array([-v_ax[1], v_ax[0]], dtype=np.float32) / scale

        mid_from_axis = crown + t_eye * v_ax
        mid_from_eye  = this_eye - sign * d_rat * scale * v_perp
        mid_eye       = 0.5 * mid_from_axis + 0.5 * mid_from_eye

        out[other       ] = (mid_eye - sign * d_rat * scale * v_perp).astype(np.float32)
        out[self.KP_BEAK] = (crown + v_ax).astype(np.float32)
        return out

    def _from_beak_and_eye(self, p, eye_idx, out):
        """
        Known: beak + one eye.
        Strategy: infer crown from beak along temporal axis, then reconstruct
        eye midpoint and other eye by symmetry.
        """
        other = self.KP_REYE if eye_idx == self.KP_LEYE else self.KP_LEYE
        sign  = 1.0 if eye_idx == self.KP_LEYE else -1.0

        beak     = p[self.KP_BEAK].astype(np.float32)
        this_eye = p[eye_idx     ].astype(np.float32)
        scale    = self._t_scale        if self._t_scale        is not None else 50.0
        angle    = self._t_angle        if self._t_angle        is not None else 0.0
        t_eye    = self._t_eye          if self._t_eye          is not None else 0.45
        d_rat    = self._d_perp_ratio   if self._d_perp_ratio   is not None else 0.20
        if abs(d_rat) < 0.01:
            d_rat = 0.20

        v_ax   = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * scale
        v_perp = np.array([-v_ax[1], v_ax[0]], dtype=np.float32) / scale

        crown              = (beak - v_ax).astype(np.float32)
        out[self.KP_CROWN] = crown

        mid_from_axis = crown + t_eye * v_ax
        mid_from_eye  = this_eye - sign * d_rat * scale * v_perp
        mid_eye       = 0.5 * mid_from_axis + 0.5 * mid_from_eye

        out[other] = (mid_eye - sign * d_rat * scale * v_perp).astype(np.float32)
        return out


# --------------------------------------------------------------------------------------


class One_Euro_Custom_Filter:
    """
    Stage-1 smoothing using a One-Euro-style filter
    adapted for confidence-weighted pose trajectories.

    This stage removes high-frequency jitter from
    high-confidence detections without hallucinating motion.
    """

    def __init__(
        self,
        fps=30.0,
        min_cutoff=0.5,
        beta=0.5,
        d_cutoff=1.2,
        min_confidence=0.0,
        max_confidence=1.0,
        use_velocity_prediction=False,
        conf_threshold=0.6,
        # Stage-2 geometric corrector parameters
        geom_conf_thresh    = 0.45,
        geom_vel_thresh     = 25.0,
        geom_template_conf  = 0.60,
        geom_corr_blend     = 0.75,
        geom_temporal_decay = 0.85,
    ):
        """
        Args:
            fps: Video frame rate.
            min_cutoff: Base cutoff frequency (Hz). Research uses 0.15-1.0 for poses.
            beta: Speed coefficient. Research uses 0.0-0.01 for poses.
            d_cutoff: Cutoff frequency for velocity filtering.
            min_confidence: Lower bound for confidence normalization.
            max_confidence: Upper bound for confidence normalization.
            use_velocity_prediction: Reduce lag via velocity extrapolation.
            conf_threshold: Minimum confidence to apply One-Euro smoothing.
            geom_conf_thresh: Min reliability for a joint to be considered valid in Stage 2.
            geom_vel_thresh: px/frame – excess displacement penalty threshold in Stage 2.
            geom_template_conf: Min per-joint reliability to include a frame in template.
            geom_corr_blend: Weight toward geometric reconstruction vs raw filtered (Stage 2).
            geom_temporal_decay: EMA decay for temporal head-pose state in Stage 2.
        """
        self.fps = float(fps)
        self.dt = 1.0 / self.fps

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.min_conf = float(min_confidence)
        self.max_conf = float(max_confidence)
        self.conf_threshold = float(conf_threshold)

        self.use_velocity_prediction = bool(use_velocity_prediction)

        # Stage-2 parameters stored for corrector construction
        self.geom_conf_thresh    = float(geom_conf_thresh)
        self.geom_vel_thresh     = float(geom_vel_thresh)
        self.geom_template_conf  = float(geom_template_conf)
        self.geom_corr_blend     = float(geom_corr_blend)
        self.geom_temporal_decay = float(geom_temporal_decay)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / self.dt)

    def _lowpass(self, x, x_prev, alpha):
        return alpha * x + (1.0 - alpha) * x_prev

    def _normalize_confidence(self, c):
        c = float(c)
        c = np.clip(c, self.min_conf, self.max_conf)
        if self.max_conf > self.min_conf:
            c = (c - self.min_conf) / (self.max_conf - self.min_conf)
        return np.clip(c, 0.0, 1.0)

    # -----------------------------
    # Stage-1 smoothing
    # -----------------------------
    def smooth_high_confidence(self, poses, scores=None):
        """
        Apply confidence-aware One-Euro filtering.

        Args:
            poses: list of (num_joints, 2)
            scores: optional list of per-joint confidence values

        Returns:
            smoothed_poses: list of (num_joints, 2)
        """
        if poses is None or len(poses) == 0:
            return []

        T = len(poses)
        if scores is None:
            scores = [None] * T

        smoothed = []
        prev_pos = None
        prev_vel = None

        for t in range(T):
            pose_t = poses[t]
            if pose_t is None:
                smoothed.append(None)
                prev_pos = None
                prev_vel = None
                continue

            pose_t = np.asarray(pose_t, dtype=np.float32)
            num_joints = pose_t.shape[0]
            out = pose_t.copy()

            if prev_pos is None:
                smoothed.append(out)
                prev_pos = out
                prev_vel = np.zeros_like(out)
                continue

            for j in range(num_joints):
                # --- confidence handling ---
                conf = 1.0
                if scores[t] is not None:
                    try:
                        conf = scores[t][j][0]
                    except Exception:
                        conf = scores[t][j] if np.isscalar(scores[t][j]) else 1.0

                conf = self._normalize_confidence(conf)

                # FIXED: Skip low-confidence joints entirely (leave for Stage 2)
                if conf < self.conf_threshold:
                    out[j] = pose_t[j]  # No smoothing, pass to Stage 2
                    continue

                # --- velocity estimate ---
                vel = (pose_t[j] - prev_pos[j]) / self.dt
                alpha_d = self._alpha(self.d_cutoff)
                vel_hat = self._lowpass(vel, prev_vel[j], alpha_d)

                # FIXED: Standard One Euro formula (NO confidence multiplication!)
                # This is what the research uses
                speed = np.linalg.norm(vel_hat)
                cutoff = self.min_cutoff + self.beta * speed
                alpha = self._alpha(cutoff)

                # --- prediction to reduce lag ---
                if self.use_velocity_prediction:
                    pred = prev_pos[j] + vel_hat * self.dt
                else:
                    pred = prev_pos[j]

                # --- final smoothing ---
                out[j] = self._lowpass(pose_t[j], pred, alpha)
                prev_vel[j] = vel_hat

            smoothed.append(out)
            prev_pos = out

        return smoothed


    def correct_low_confidence(self, poses, scores=None):
        """
        Stage-2: geometry-aware skeleton correction via GeometricSkeletonCorrector.

        For each frame, joints are assessed for reliability using confidence scores
        and temporal velocity consistency.  Unreliable joints are repaired by:
            • weighted Procrustes fit to a learned canonical head shape (3-4 reliable)
            • geometric reconstruction from eye symmetry / head-axis constraints (2)
            • temporal EMA extrapolation                                          (≤1)
        """
        corrector = GeometricSkeletonCorrector(
            conf_thresh    = self.geom_conf_thresh,
            vel_thresh     = self.geom_vel_thresh,
            template_conf  = self.geom_template_conf,
            corr_blend     = self.geom_corr_blend,
            temporal_decay = self.geom_temporal_decay,
        )
        return corrector.correct_sequence(poses, scores)

    def final_smooth(self, poses, scores=None):
        """
        Stage-3 placeholder for final weak global smoothing.
        """
        return poses





# --------------------------------------------------