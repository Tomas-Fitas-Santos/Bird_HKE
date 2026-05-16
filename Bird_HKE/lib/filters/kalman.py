import numpy as np

class KalmanFilter2D:
    """
    Confidence-aware, motion-adaptive Kalman filter for 2D points.

    State:      [x, y, vx, vy]
    Measurement [x, y]

    Designed for offline pose smoothing with fast & unpredictable motion.
    """

    def __init__(
        self,
        dt=1.0,
        var_process_pos=15.0,
        var_process_vel=50.0,
        var_meas=10.0,
        q_adapt_alpha=0.5,
        min_confidence=0.20,
        skip_confidence=0.35,
        outlier_gate=9.0,
        anti_drift_frames=15,
    ):
        self.dt = float(dt)

        # State transition
        self.F = np.array(
            [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32
        )

        # Measurement model
        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32
        )

        # Base process noise (will be adapted)
        self.var_process_pos_base = float(var_process_pos)
        self.var_process_vel_base = float(var_process_vel)
        
        self.Q_base = np.diag([
            var_process_pos,
            var_process_pos,
            var_process_vel,
            var_process_vel
        ]).astype(np.float32)
        
        self.Q = self.Q_base.copy()

        # Base measurement noise (scaled by confidence)
        self.R_base = np.diag([var_meas, var_meas]).astype(np.float32)

        # Adaptation parameters
        self.q_adapt_alpha = float(q_adapt_alpha)
        self.min_confidence = float(min_confidence)
        self.skip_confidence = float(skip_confidence)
        self.outlier_gate = float(outlier_gate)
        self.anti_drift_frames = int(anti_drift_frames)

        # State & covariance
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e3
        self.initialized = False
        self.update_count = 0

        # History (for RTS smoothing)
        self.x_pred_hist = []
        self.P_pred_hist = []
        self.x_filt_hist = []
        self.P_filt_hist = []
        self.measurement_flags = []  # NEW: Track which frames had measurements
        
        # For motion adaptation
        self.velocity_history = []
        self.max_vel_history = 20
        
        # Statistics
        self.outlier_count = 0
        self.total_updates = 0

    def initialize(self, meas, meas_conf=1.0):
        """Initialize filter with first measurement"""
        meas = np.asarray(meas, dtype=np.float32).reshape(2)

        self.x[:] = [meas[0], meas[1], 0.0, 0.0]
        
        # NEW: Scale initial uncertainty by confidence
        pos_var = 25.0 / max(meas_conf, 0.3)
        vel_var = 100.0
        
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float32)
        self.initialized = True
        self.update_count = 0

    def predict(self):
        """Predict next state"""
        if not self.initialized:
            return self.x.copy()

        # Adapt Q based on velocity
        self._adapt_Q()

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save predicted state (for RTS)
        self.x_pred_hist.append(self.x.copy())
        self.P_pred_hist.append(self.P.copy())

        return self.x.copy()

    def update(self, meas, meas_conf=1.0):
        """Update with measurement"""
        self.total_updates += 1
        
        if meas is None:
            self._store_filtered(has_measurement=False)
            return self.x.copy()

        meas = np.asarray(meas, dtype=np.float32).reshape(-1)
        if meas.shape[0] != 2:
            raise ValueError("Measurement must be shape (2,)")

        if not self.initialized:
            self.initialize(meas, meas_conf)
            self._store_filtered(has_measurement=True)
            return self.x.copy()

        # Clamp confidence
        conf = float(np.clip(meas_conf, self.min_confidence, 1.0))

        # Anti-drift - boost confidence for initial frames
        if self.update_count < self.anti_drift_frames:
            boost = 1.5 - (self.update_count / self.anti_drift_frames) * 0.5
            conf = min(1.0, conf * boost)

        # Skip very low-confidence updates
        if conf < self.skip_confidence:
            self._store_filtered(has_measurement=False)
            return self.x.copy()

        # Outlier gating
        md = self.mahalanobis_distance(meas, meas_conf=conf)
        adaptive_gate = self.outlier_gate / max(conf, 0.2)
        
        if md > adaptive_gate:
            # Outlier - skip update
            self.outlier_count += 1
            self._store_filtered(has_measurement=False)
            return self.x.copy()

        # Measurement noise ∝ 1 / confidence²
        R = self.R_base / (conf ** 2)

        z = meas.reshape(2, 1)
        x_pred = self.x.reshape(4, 1)

        innovation = z - self.H @ x_pred
        S = self.H @ self.P @ self.H.T + R
        
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Update state
        self.x = (x_pred + K @ innovation).flatten()
        I = np.eye(4, dtype=np.float32)
        
        # Joseph form for numerical stability
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        # Track velocity for adaptation
        vel = np.sqrt(self.x[2]**2 + self.x[3]**2)
        self.velocity_history.append(vel)
        if len(self.velocity_history) > self.max_vel_history:
            self.velocity_history.pop(0)
        
        self.update_count += 1

        self._store_filtered(has_measurement=True)
        return self.x.copy()
    
    def _adapt_Q(self):
        """Adapt process noise Q based on recent velocity"""
        if len(self.velocity_history) < 3:
            return
        
        recent_vel = np.mean(self.velocity_history[-10:])
        vel_variance = np.var(self.velocity_history[-10:]) if len(self.velocity_history) >= 10 else 0
        
        # Scale based on velocity and variance
        vel_scale = 1.0 + np.clip(recent_vel / 50.0, 0, 3.0)
        var_scale = 1.0 + np.clip(vel_variance / 100.0, 0, 2.0)
        total_scale = vel_scale * var_scale
        
        # New Q
        q_pos = self.var_process_pos_base * total_scale
        q_vel = self.var_process_vel_base * total_scale
        Q_new = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        
        # Smooth with exponential filter
        self.Q = self.q_adapt_alpha * Q_new + (1 - self.q_adapt_alpha) * self.Q
    
    def smooth(self):
        """RTS backward smoothing"""
        T = len(self.x_filt_hist)
        if T == 0:
            return None, None

        x_smooth = [None] * T
        P_smooth = [None] * T

        # Initialize with last filtered state
        x_smooth[-1] = self.x_filt_hist[-1].copy()
        P_smooth[-1] = self.P_filt_hist[-1].copy()

        for t in reversed(range(T - 1)):
            P_filt = self.P_filt_hist[t]
            P_pred_next = self.P_pred_hist[t + 1]

            # Smoothing gain
            G = P_filt @ self.F.T @ np.linalg.pinv(P_pred_next)

            x_smooth[t] = (
                self.x_filt_hist[t]
                + G @ (x_smooth[t + 1] - self.x_pred_hist[t + 1])
            )

            P_smooth[t] = (
                P_filt
                + G @ (P_smooth[t + 1] - P_pred_next) @ G.T
            )
            
            # Ensure symmetry for numerical stability
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        return np.stack(x_smooth), np.stack(P_smooth)

    def _store_filtered(self, has_measurement=True):
        """Store filtered state for RTS"""
        self.x_filt_hist.append(self.x.copy())
        self.P_filt_hist.append(self.P.copy())
        self.measurement_flags.append(has_measurement)

    def get_state(self):
        return self.x.copy()
    
    def get_stats(self):
        """Get filter statistics"""
        return {
            'total_updates': self.total_updates,
            'outlier_count': self.outlier_count,
            'outlier_rate': self.outlier_count / max(1, self.total_updates),
            'avg_velocity': np.mean(self.velocity_history) if self.velocity_history else 0.0,
            'frames_processed': len(self.x_filt_hist),
            'frames_with_measurements': sum(self.measurement_flags),
        }

    def mahalanobis_distance(self, meas, meas_conf=1.0):
        if not self.initialized:
            return 0.0

        meas = np.asarray(meas, dtype=np.float32).reshape(2)
        conf = float(np.clip(meas_conf, self.min_confidence, 1.0))
        R = self.R_base / (conf ** 2)

        z = meas.reshape(2, 1)
        x = self.x.reshape(4, 1)
        y = z - self.H @ x
        S = self.H @ self.P @ self.H.T + R

        invS = np.linalg.pinv(S)
        d2 = float(y.T @ invS @ y)
        return float(np.sqrt(max(0, d2)))
    
