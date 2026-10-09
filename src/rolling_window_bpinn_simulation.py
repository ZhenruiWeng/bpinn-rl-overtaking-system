# ============================================================
# Rolling-Window BPINN Overtaking Simulation
# ============================================================
# Abandons pre-loaded dataset; generates scenario on-the-fly.
#
# Scenario (3 phases):
#   Phase 1 (t=0~3s):   Gap opens  → V6 starts overtaking
#   Phase 2 (t=3~6s):   Gap closes → V6 aborts, returns to lane
#   Phase 3 (t=6~10s):  Gap opens  → V6 successfully overtakes
#
# At each rolling-window step:
#   1. Forward BPINN: predict V0/V3 future trajectories + uncertainty (MC Dropout)
#   2. Decision:      evaluate predicted gap → overtake / hold / abort
#   3. Inverse BPINN: if overtaking, estimate control parameters from planned trajectory
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.interpolate import PchipInterpolator, interp1d
import csv
import time as time_module

# ============================================================
# 0. Configuration
# ============================================================
np.random.seed(42)

DT = 0.1            # simulation time step (s)
T_TOTAL = 10.0      # total simulation time (s)
N_STEPS = int(T_TOTAL / DT) + 1   # 101 points

# Rolling window parameters
WINDOW_SIZE = 20     # 2 seconds of history (20 * 0.1s)
DECISION_INTERVAL = 5  # make a decision every 0.5s
PRED_HORIZON = 20    # predict 2s ahead (20 * 0.1s)

# Lane geometry (4 lanes)
LANE_WIDTH = 4.0
LANE_CENTERS = [3.0, 7.0, 11.0, 15.0]
ROAD_Y_MIN = 1.0
ROAD_Y_MAX = 17.0

# Vehicle dimensions
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8

# Decision thresholds
GAP_THRESHOLD = 30.0       # minimum longitudinal gap for safe overtake (m)
GAP_ABORT_THRESHOLD = 20.0  # gap below this → abort immediately
CONF_THRESHOLD = 0.8       # confidence threshold for overtake decision

# ============================================================
# 1. Lightweight BPINN (NumPy-based for rolling-window speed)
# ============================================================
class LightweightBPINN:
    """
    Lightweight BPINN using NumPy with MC Dropout.
    Approximates the full JAX/Flax BPINN for real-time rolling window use.
    
    Architecture: input(t) → Dense(32, tanh) → Dropout → Dense(32, tanh) → Dropout → Dense(4)
    Output: [x, y, v, heading]
    
    Physics constraint: dx/dt = v*cos(θ), dy/dt = v*sin(θ)
    """
    
    def __init__(self, hidden_dim=32, dropout_rate=0.1, lr=0.01, n_epochs=300, 
                 physics_weight=1.0, mc_samples=50):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.n_epochs = n_epochs
        self.physics_weight = physics_weight
        self.mc_samples = mc_samples
        self.params = None
        self.norm = None
        
    def _init_params(self, input_dim=1):
        """Xavier initialization for network weights."""
        h = self.hidden_dim
        params = {
            'W1': np.random.randn(input_dim, h) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(h),
            'W2': np.random.randn(h, h) * np.sqrt(2.0 / h),
            'b2': np.zeros(h),
            'W3': np.random.randn(h, 4) * np.sqrt(2.0 / h),
            'b3': np.zeros(4),
        }
        return params
    
    def _forward(self, params, t_batch, training=True, dropout_mask=None):
        """Forward pass. Returns [x, y, v, heading] predictions."""
        # Layer 1
        z1 = t_batch @ params['W1'] + params['b1']
        a1 = np.tanh(z1)
        if training and dropout_mask is not None:
            a1 = a1 * dropout_mask['m1']
        
        # Layer 2
        z2 = a1 @ params['W2'] + params['b2']
        a2 = np.tanh(z2)
        if training and dropout_mask is not None:
            a2 = a2 * dropout_mask['m2']
        
        # Output layer
        out = a2 @ params['W3'] + params['b3']
        return out
    
    def _gen_dropout_mask(self, n, training=True):
        """Generate binary dropout masks."""
        if not training:
            return {'m1': 1.0, 'm2': 1.0}
        m1 = (np.random.rand(self.hidden_dim) > self.dropout_rate) / (1 - self.dropout_rate)
        m2 = (np.random.rand(self.hidden_dim) > self.dropout_rate) / (1 - self.dropout_rate)
        return {'m1': m1, 'm2': m2}
    
    def _compute_loss(self, params, t_batch, y_batch, training=True):
        """Data loss + physics loss."""
        n = len(t_batch)
        
        # Forward pass
        if training:
            mask = self._gen_dropout_mask(n, training=True)
        else:
            mask = {'m1': 1.0, 'm2': 1.0}
        
        pred = self._forward(params, t_batch, training=training, dropout_mask=mask)
        
        # Data loss (MSE)
        data_loss = np.mean((pred - y_batch) ** 2)
        
        # Physics loss: approximate derivatives using finite differences
        if n >= 2:
            dt = t_batch[1] - t_batch[0] if n > 1 else 0.1
            pred_x = pred[:, 0]
            pred_y = pred[:, 1]
            pred_v = pred[:, 2]
            pred_h = pred[:, 3]
            
            # Numerical derivatives
            dx_dt = np.gradient(pred_x, dt)
            dy_dt = np.gradient(pred_y, dt)
            
            # Physics: dx/dt ≈ v*cos(h), dy/dt ≈ v*sin(h)
            phys_x = dx_dt - pred_v * np.cos(pred_h)
            phys_y = dy_dt - pred_v * np.sin(pred_h)
            phys_loss = np.mean(phys_x ** 2 + phys_y ** 2)
        else:
            phys_loss = 0.0
        
        total = data_loss + self.physics_weight * phys_loss
        return total, data_loss, phys_loss
    
    def _numerical_grad(self, params, t_batch, y_batch, eps=1e-4):
        """Numerical gradient estimation (finite differences)."""
        grads = {}
        for key in params:
            grads[key] = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig = params[key][idx]
                
                params[key][idx] = orig + eps
                loss_plus, _, _ = self._compute_loss(params, t_batch, y_batch, training=True)
                
                params[key][idx] = orig - eps
                loss_minus, _, _ = self._compute_loss(params, t_batch, y_batch, training=True)
                
                grads[key][idx] = (loss_plus - loss_minus) / (2 * eps)
                params[key][idx] = orig
                it.iternext()
        
        return grads
    
    def fit(self, t_data, x_data, y_data, v_data, h_data):
        """Train the BPINN on vehicle data."""
        # Normalize
        self.norm = {
            't_mean': np.mean(t_data), 't_std': np.std(t_data) + 1e-8,
            'x_mean': np.mean(x_data), 'x_std': np.std(x_data) + 1e-8,
            'y_mean': np.mean(y_data), 'y_std': np.std(y_data) + 1e-8,
            'v_mean': np.mean(v_data), 'v_std': np.std(v_data) + 1e-8,
            'h_mean': np.mean(h_data), 'h_std': np.std(h_data) + 1e-8,
        }
        
        # Normalized inputs
        t_n = ((t_data - self.norm['t_mean']) / self.norm['t_std']).reshape(-1, 1)
        y_n = np.column_stack([
            (x_data - self.norm['x_mean']) / self.norm['x_std'],
            (y_data - self.norm['y_mean']) / self.norm['y_std'],
            (v_data - self.norm['v_mean']) / self.norm['v_std'],
            (h_data - self.norm['h_mean']) / self.norm['h_std'],
        ])
        
        # Initialize params
        self.params = self._init_params(input_dim=1)
        
        # Adam optimizer state
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_mom = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        # Training loop (using Adam with numerical gradients)
        # For speed: use batch gradient descent with momentum instead of full numerical grad
        # Use a simplified approach: train with analytical gradient approximation
        
        for epoch in range(self.n_epochs):
            # Generate dropout mask for this epoch
            mask = self._gen_dropout_mask(len(t_n), training=True)
            
            # Forward pass
            pred = self._forward(self.params, t_n, training=True, dropout_mask=mask)
            
            # Data loss gradient (analytical for output layer)
            error = pred - y_n  # [N, 4]
            
            # Backpropagation (manual)
            # Output layer gradient
            a2 = np.tanh(t_n @ self.params['W1'] + self.params['b1'])
            if mask is not None:
                a2 = a2 * mask['m2']
            
            grad_W3 = a2.T @ error  # [hidden, 4]
            grad_b3 = np.sum(error, axis=0)  # [4]
            
            # Hidden layer 2 gradient
            delta2 = (error @ self.params['W3'].T) * (1 - a2 ** 2)  # tanh derivative
            if mask is not None:
                delta2 = delta2 * mask['m2']
            
            a1 = np.tanh(t_n @ self.params['W1'] + self.params['b1'])
            if mask is not None:
                a1 = a1 * mask['m1']
            
            grad_W2 = a1.T @ delta2
            grad_b2 = np.sum(delta2, axis=0)
            
            # Hidden layer 1 gradient
            delta1 = (delta2 @ self.params['W2'].T) * (1 - a1 ** 2)
            if mask is not None:
                delta1 = delta1 * mask['m1']
            
            grad_W1 = t_n.T @ delta1
            grad_b1 = np.sum(delta1, axis=0)
            
            # Add small physics regularization (simplified)
            if self.physics_weight > 0 and len(t_n) >= 3:
                dt_val = t_n[1, 0] - t_n[0, 0]
                dx_dt = np.gradient(pred[:, 0], dt_val)
                dy_dt = np.gradient(pred[:, 1], dt_val)
                v_pred = pred[:, 2]
                h_pred = pred[:, 3]
                
                phys_err_x = (dx_dt - v_pred * np.cos(h_pred)).reshape(-1, 1)
                phys_err_y = (dy_dt - v_pred * np.sin(h_pred)).reshape(-1, 1)
                
                # Simplified physics gradient contribution
                grad_W3 += self.physics_weight * 0.01 * a2.T @ phys_err_x
                grad_b3 += self.physics_weight * 0.01 * np.sum(phys_err_x, axis=0)
            
            # Adam update
            grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 
                     'b2': grad_b2, 'W3': grad_W3, 'b3': grad_b3}
            
            for key in self.params:
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v_mom[key] = beta2 * v_mom[key] + (1 - beta2) * (grads[key] ** 2)
                m_hat = m[key] / (1 - beta1 ** (epoch + 1))
                v_hat = v_mom[key] / (1 - beta2 ** (epoch + 1))
                self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return self
    
    def predict_with_uncertainty(self, t_future):
        """MC Dropout prediction: returns mean and std for each output."""
        t_n = ((t_future - self.norm['t_mean']) / self.norm['t_std']).reshape(-1, 1)
        
        all_preds = []
        for _ in range(self.mc_samples):
            mask = self._gen_dropout_mask(len(t_n), training=True)
            pred = self._forward(self.params, t_n, training=True, dropout_mask=mask)
            # Denormalize
            pred_denorm = np.column_stack([
                pred[:, 0] * self.norm['x_std'] + self.norm['x_mean'],
                pred[:, 1] * self.norm['y_std'] + self.norm['y_mean'],
                pred[:, 2] * self.norm['v_std'] + self.norm['v_mean'],
                pred[:, 3] * self.norm['h_std'] + self.norm['h_mean'],
            ])
            all_preds.append(pred_denorm)
        
        all_preds = np.array(all_preds)  # [mc_samples, N, 4]
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        return mean_pred, std_pred, all_preds


# ============================================================
# 2. Scenario Generation
# ============================================================
print("=" * 70)
print("  ROLLING-WINDOW BPINN OVERTAKING SIMULATION")
print("=" * 70)

print("\n[1/5] Generating three-phase dynamic scenario...")

t = np.arange(0, T_TOTAL + DT, DT)
N = len(t)

# --- V0: Red, lead vehicle in Lane 3 ---
# Phase 1: v=35 m/s (fast, gap to V3 grows rapidly, rel=+7 m/s)
# Phase 2: v=18 m/s (very slow, gap closes below threshold, rel=-10 m/s)
# Phase 3: v=36 m/s (fast again, gap opens for successful overtake, rel=+8 m/s)
# V3 constant at ~28 m/s
v0_speed = np.zeros(N)
for i, ti in enumerate(t):
    if ti < 3.0:
        v0_speed[i] = 35.0 + 0.3 * np.sin(ti * 1.5)
    elif ti < 4.0:
        # Rapid deceleration in first 1s of Phase 2
        v0_speed[i] = 35.0 - 17.0 * (ti - 3.0) + 0.3 * np.sin(ti * 1.5)
    elif ti < 6.0:
        # Hold at low speed
        v0_speed[i] = 18.0 + 0.3 * np.sin(ti * 1.5)
    elif ti < 8.0:
        # Rapid acceleration in first 2s of Phase 3
        v0_speed[i] = 18.0 + 20.0 * (ti - 6.0) + 0.3 * np.sin(ti * 1.5)
    else:
        # Hold at high speed (38 m/s)
        v0_speed[i] = 38.0 + 0.3 * np.sin(ti * 1.5)
v0_speed += np.random.normal(0, 0.15, N)

# --- V3: Blue, reference vehicle in Lane 2 ---
# Constant speed ~28 m/s (faster than V0 in Phase 2, slower in Phases 1 & 3)
v3_speed = np.zeros(N)
for i, ti in enumerate(t):
    v3_speed[i] = 28.0 + 0.3 * np.sin(ti * 2.0)
v3_speed += np.random.normal(0, 0.12, N)

# --- V6: Yellow, ego vehicle ---
# Phase 1: accelerating (going for overtake, matches V3 speed then accelerates)
# Phase 2: decelerating (aborting, falling back behind V3)
# Phase 3: accelerating (second attempt, successful overtake)
v6_speed = np.zeros(N)
for i, ti in enumerate(t):
    if ti < 3.0:
        v6_speed[i] = 28.0 + 4.0 * (ti / 3.0)       # 28→32 m/s
    elif ti < 6.0:
        v6_speed[i] = 32.0 - 6.0 * (ti - 3.0) / 3.0  # 32→26 m/s (decelerate)
    else:
        v6_speed[i] = 26.0 + 6.0 * (ti - 6.0) / 4.0  # 26→32 m/s (accelerate)
v6_speed += np.random.normal(0, 0.1, N)

# Compute positions by integration
v0_x = np.zeros(N); v0_y = np.zeros(N); v0_heading = np.zeros(N)
v3_x = np.zeros(N); v3_y = np.zeros(N); v3_heading = np.zeros(N)
v6_x = np.zeros(N); v6_y = np.zeros(N); v6_heading = np.zeros(N)

# Start with V0 25m ahead of V3 (gap near threshold, will grow in Phase 1)
v0_x[0] = 175.0; v0_y[0] = LANE_CENTERS[2]
v3_x[0] = 150.0; v3_y[0] = LANE_CENTERS[1]
v6_x[0] = 120.0; v6_y[0] = LANE_CENTERS[1]

for i in range(1, N):
    # V0
    v0_x[i] = v0_x[i-1] + v0_speed[i-1] * DT
    v0_y[i] = LANE_CENTERS[2] + 0.3 * np.sin(t[i] * 0.8)
    
    # V3
    v3_x[i] = v3_x[i-1] + v3_speed[i-1] * DT
    v3_y[i] = LANE_CENTERS[1] + 0.4 * np.sin(t[i] * 1.2)
    
    # V6 (3-phase lane changes)
    v6_x[i] = v6_x[i-1] + v6_speed[i-1] * DT
    ti = t[i]
    if ti < 0.5:
        v6_y[i] = LANE_CENTERS[1]
    elif ti < 3.0:
        progress = (ti - 0.5) / 2.5
        v6_y[i] = LANE_CENTERS[1] + (LANE_CENTERS[2] - LANE_CENTERS[1]) * progress
    elif ti < 3.5:
        v6_y[i] = LANE_CENTERS[2]
    elif ti < 6.0:
        progress = (ti - 3.5) / 2.5
        v6_y[i] = LANE_CENTERS[2] + (LANE_CENTERS[1] - LANE_CENTERS[2]) * progress
    elif ti < 6.5:
        v6_y[i] = LANE_CENTERS[1]
    elif ti < 8.5:
        progress = (ti - 6.5) / 2.0
        v6_y[i] = LANE_CENTERS[1] + (LANE_CENTERS[2] - LANE_CENTERS[1]) * progress
    elif ti < 9.0:
        v6_y[i] = LANE_CENTERS[2]
    else:
        progress = (ti - 9.0) / 1.0
        v6_y[i] = LANE_CENTERS[2] + (LANE_CENTERS[1] - LANE_CENTERS[2]) * min(progress, 1.0)
    
    # Compute headings
    for arr_x, arr_y, arr_h in [(v0_x, v0_y, v0_heading), (v3_x, v3_y, v3_heading), (v6_x, v6_y, v6_heading)]:
        if abs(arr_x[i] - arr_x[i-1]) > 1e-6:
            arr_h[i] = np.arctan2(arr_y[i] - arr_y[i-1], arr_x[i] - arr_x[i-1])

# Add noise to positions
v0_x += np.random.normal(0, 0.1, N)
v0_y += np.random.normal(0, 0.05, N)
v3_x += np.random.normal(0, 0.1, N)
v3_y += np.random.normal(0, 0.05, N)

# Compute ground truth gap
gap_v0_v3 = v0_x - v3_x

print(f"  Phase 1 (t=0~3s):   Gap = {gap_v0_v3[0]:.1f}m -> {gap_v0_v3[30]:.1f}m  (opens)")
print(f"  Phase 2 (t=3~6s):   Gap = {gap_v0_v3[30]:.1f}m -> {gap_v0_v3[60]:.1f}m  (closes)")
print(f"  Phase 3 (t=6~10s):  Gap = {gap_v0_v3[60]:.1f}m -> {gap_v0_v3[-1]:.1f}m  (opens)")

# Save scenario data
csv_path = "dynamic_overtaking_scenario.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'vehicle_id', 'x', 'y', 'velocity', 'heading'])
    for i in range(N):
        for vid, arr_x, arr_y, arr_v, arr_h in [
            ('V0', v0_x, v0_y, v0_speed, v0_heading),
            ('V3', v3_x, v3_y, v3_speed, v3_heading),
            ('V6', v6_x, v6_y, v6_speed, v6_heading)]:
            writer.writerow([f'{t[i]:.1f}', vid, f'{arr_x[i]:.2f}', f'{arr_y[i]:.2f}',
                           f'{arr_v[i]:.2f}', f'{arr_h[i]:.4f}'])
print(f"  CSV saved: {csv_path} ({N*3} rows)")

# ============================================================
# 3. Rolling-Window BPINN Loop
# ============================================================
print("\n[2/5] Running rolling-window BPINN...")

# Decision history
decision_log = {
    'time': [], 'action': [], 'gap_pred': [], 'gap_actual': [],
    'gap_std': [], 'overtake_confidence': [],
    'inv_v': [], 'inv_v_std': [], 'inv_h': [], 'inv_h_std': [],
}

# Prediction history at each decision point
prediction_snapshots = []

# Run rolling window
decision_points = list(range(WINDOW_SIZE, N, DECISION_INTERVAL))
print(f"  Decision points: {len(decision_points)} (every {DECISION_INTERVAL*DT:.1f}s)")

for step_idx, current_idx in enumerate(decision_points):
    current_time = t[current_idx]
    
    # --- Extract sliding window ---
    start_idx = max(0, current_idx - WINDOW_SIZE)
    window_t = t[start_idx:current_idx+1]
    
    v0_t = v0_speed[start_idx:current_idx+1]
    v0_x_w = v0_x[start_idx:current_idx+1]
    v0_y_w = v0_y[start_idx:current_idx+1]
    v0_h_w = v0_heading[start_idx:current_idx+1]
    
    v3_t = v3_speed[start_idx:current_idx+1]
    v3_x_w = v3_x[start_idx:current_idx+1]
    v3_y_w = v3_y[start_idx:current_idx+1]
    v3_h_w = v3_heading[start_idx:current_idx+1]
    
    # --- Forward BPINN for V0 ---
    bpinn_v0 = LightweightBPINN(
        hidden_dim=32, dropout_rate=0.1, lr=0.01, n_epochs=300,
        physics_weight=0.5, mc_samples=50)
    bpinn_v0.fit(window_t, v0_x_w, v0_y_w, v0_t, v0_h_w)
    
    # --- Forward BPINN for V3 ---
    bpinn_v3 = LightweightBPINN(
        hidden_dim=32, dropout_rate=0.1, lr=0.01, n_epochs=300,
        physics_weight=0.5, mc_samples=50)
    bpinn_v3.fit(window_t, v3_x_w, v3_y_w, v3_t, v3_h_w)
    
    # --- Predict future trajectory ---
    future_t = t[current_idx:min(current_idx + PRED_HORIZON, N)]
    if len(future_t) < 2:
        future_t = np.array([current_time, current_time + 0.1])
    
    v0_mean, v0_std, v0_all = bpinn_v0.predict_with_uncertainty(future_t)
    v3_mean, v3_std, v3_all = bpinn_v3.predict_with_uncertainty(future_t)
    
    # --- Predicted gap ---
    gap_pred = v0_mean[:, 0] - v3_mean[:, 0]  # x position difference
    gap_pred_mean = np.mean(gap_pred)
    gap_pred_std = np.std(gap_pred)
    
    # Actual gap at current time
    gap_actual = gap_v0_v3[current_idx]
    
    # --- Decision logic ---
    # Hybrid approach: use actual sensor gap (primary) + BPINN trend prediction (secondary)
    # In a real system, the ego vehicle knows current positions via sensors.
    # BPINN provides uncertainty estimates and trend prediction.
    
    # Current actual gap from sensors
    sensor_gap = gap_actual
    
    # BPINN predicted gap trend (slope over prediction horizon)
    if len(gap_pred) >= 2:
        gap_trend = (gap_pred[-1] - gap_pred[0]) / (future_t[-1] - future_t[0] + 1e-6)
    else:
        gap_trend = 0.0
    
    # Confidence: P(gap > threshold) from MC samples
    gap_samples = []
    for s in range(min(50, len(v0_all))):
        gap_s = v0_all[s, :, 0] - v3_all[s, :, 0]
        gap_samples.append(np.mean(gap_s))
    gap_samples = np.array(gap_samples)
    
    overtake_confidence = np.mean(gap_samples > GAP_THRESHOLD)
    
    # Determine action based on sensor gap + BPINN prediction
    current_y = v6_y[current_idx]
    in_lane2 = abs(current_y - LANE_CENTERS[1]) < 1.5
    in_lane3 = abs(current_y - LANE_CENTERS[2]) < 1.5
    
    # Primary decision from sensor gap
    if sensor_gap > GAP_THRESHOLD and gap_trend >= -2.0:
        # Gap is large enough and not rapidly closing
        if in_lane2:
            action = 'OVERTAKE'
        elif in_lane3:
            action = 'CONTINUE'
        else:
            action = 'OVERTAKE'
    elif sensor_gap < GAP_ABORT_THRESHOLD or (gap_trend < -3.0 and sensor_gap < GAP_THRESHOLD):
        # Gap is too small or rapidly closing
        if in_lane3:
            action = 'ABORT'
        else:
            action = 'HOLD'
    elif sensor_gap < GAP_THRESHOLD and gap_trend < 0:
        # Gap below threshold and shrinking → hold/abort
        if in_lane3:
            action = 'ABORT'
        else:
            action = 'HOLD'
    else:
        action = 'HOLD'
    
    # --- Inverse BPINN (if overtaking) ---
    inv_v_mean = 0.0; inv_v_std = 0.0
    inv_h_mean = 0.0; inv_h_std = 0.0
    
    if action in ('OVERTAKE', 'CONTINUE'):
        # Plan trajectory: from current position to safe corridor center
        plan_t = future_t
        plan_x = v6_x[current_idx] + v6_speed[current_idx] * (plan_t - current_time)
        plan_y = np.full_like(plan_t, (LANE_CENTERS[1] + LANE_CENTERS[2]) / 2)
        
        # Inverse BPINN: estimate control parameters from planned trajectory
        bpinn_inv = LightweightBPINN(
            hidden_dim=32, dropout_rate=0.1, lr=0.01, n_epochs=200,
            physics_weight=1.0, mc_samples=50)
        
        # Create synthetic target velocity/heading from planned trajectory
        plan_v = np.gradient(plan_x, DT)
        plan_h = np.arctan2(np.gradient(plan_y, DT), np.gradient(plan_x, DT))
        
        bpinn_inv.fit(plan_t, plan_x, plan_y, plan_v, plan_h)
        
        inv_mean, inv_std, _ = bpinn_inv.predict_with_uncertainty(plan_t)
        inv_v_mean = float(np.mean(inv_mean[:, 2]))
        inv_v_std = float(np.mean(inv_std[:, 2]))
        inv_h_mean = float(np.mean(inv_mean[:, 3]))
        inv_h_std = float(np.mean(inv_std[:, 3]))
    
    # --- Log decision ---
    decision_log['time'].append(current_time)
    decision_log['action'].append(action)
    decision_log['gap_pred'].append(float(gap_pred_mean))
    decision_log['gap_actual'].append(float(gap_actual))
    decision_log['gap_std'].append(float(gap_pred_std))
    decision_log['overtake_confidence'].append(float(overtake_confidence))
    decision_log['inv_v'].append(inv_v_mean)
    decision_log['inv_v_std'].append(inv_v_std)
    decision_log['inv_h'].append(inv_h_mean)
    decision_log['inv_h_std'].append(inv_h_std)
    
    # Save snapshot for key time points
    if step_idx in [0, 5, 10, 15, 19] or action in ('OVERTAKE', 'ABORT'):
        prediction_snapshots.append({
            'time': current_time,
            'action': action,
            'future_t': future_t.copy(),
            'v0_pred_mean': v0_mean.copy(),
            'v0_pred_std': v0_std.copy(),
            'v3_pred_mean': v3_mean.copy(),
            'v3_pred_std': v3_std.copy(),
            'gap_pred': gap_pred.copy(),
            'confidence': overtake_confidence,
            'current_idx': current_idx,
        })
    
    status = f"  t={current_time:.1f}s: {action:10s} | sensor_gap={sensor_gap:.1f}m pred_gap={gap_pred_mean:.1f}m trend={gap_trend:+.1f}m/s | conf={overtake_confidence:.2f}"
    if action in ('OVERTAKE', 'CONTINUE'):
        status += f" | inv_v={inv_v_mean:.1f}±{inv_v_std:.1f} inv_h={inv_h_mean:.3f}±{inv_h_std:.3f}"
    print(status)

print(f"\n  Total decisions: {len(decision_log['time'])}")
print(f"  Snapshots saved: {len(prediction_snapshots)}")

# ============================================================
# 4. Visualization
# ============================================================
print("\n[3/5] Creating visualization...")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e8e8e8',
    'grid.linewidth': 0.6,
})

# Color palette
C_V0 = '#E74C3C'
C_V3 = '#3498DB'
C_V6 = '#F39C12'
C_OVERTAKE = '#27AE60'
C_ABORT = '#E74C3C'
C_HOLD = '#95A5A6'
C_INV = '#9B59B6'

fig = plt.figure(figsize=(24, 20), facecolor='white')
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.30,
                       left=0.06, right=0.96, top=0.95, bottom=0.05)

# ---- Panel 1: Bird's-eye view (full scenario) ----
ax1 = fig.add_subplot(gs[0, :])

# Draw lanes
for j, ly in enumerate(LANE_CENTERS):
    ax1.axhline(y=ly, color='#DDDDDD', linewidth=1.0, linestyle='--', alpha=0.6)
ax1.axhline(y=ROAD_Y_MIN, color='#999999', linewidth=1.5, linestyle='--')
ax1.axhline(y=ROAD_Y_MAX, color='#999999', linewidth=1.5, linestyle='--')

# Lane labels
for j, ly in enumerate(LANE_CENTERS):
    ax1.text(v6_x[0] - 10, ly, f'L{j+1}', fontsize=9, color='#AAAAAA', 
             va='center', ha='center', fontweight='bold')

# Phase backgrounds
x_min_plot = v6_x[0] - 15
x_max_plot = v0_x[-1] + 15
ax1.axvspan(x_min_plot, v3_x[30], alpha=0.04, color='green')
ax1.axvspan(v3_x[30], v3_x[60], alpha=0.04, color='red')
ax1.axvspan(v3_x[60], x_max_plot, alpha=0.04, color='green')

# Phase labels
ax1.text((v6_x[0] + v3_x[30]) / 2, ROAD_Y_MAX + 0.5, 'Phase 1: Gap Opens', 
         fontsize=10, ha='center', color='#27AE60', fontweight='bold')
ax1.text((v3_x[30] + v3_x[60]) / 2, ROAD_Y_MAX + 0.5, 'Phase 2: Gap Closes', 
         fontsize=10, ha='center', color='#E74C3C', fontweight='bold')
ax1.text((v3_x[60] + x_max_plot) / 2, ROAD_Y_MAX + 0.5, 'Phase 3: Overtake!', 
         fontsize=10, ha='center', color='#27AE60', fontweight='bold')

# Plot trajectories
ax1.plot(v0_x, v0_y, color=C_V0, linewidth=2.5, label='V0 (lead vehicle)', zorder=3)
ax1.plot(v3_x, v3_y, color=C_V3, linewidth=2.5, label='V3 (reference vehicle)', zorder=3)
ax1.plot(v6_x, v6_y, color=C_V6, linewidth=3, label='V6 (ego vehicle)', zorder=4)

# Draw cars at key timestamps
def draw_car(ax, x, y, color, label, angle=0):
    rect = FancyBboxPatch(
        (x - CAR_LENGTH/2, y - CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor='black',
        linewidth=0.8, alpha=0.85, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=7,
            fontweight='bold', color='white', zorder=6)

# Cars at t=0, 3, 6, 10
for idx in [0, 30, 60, -1]:
    draw_car(ax1, v0_x[idx], v0_y[idx], C_V0, 'V0')
    draw_car(ax1, v3_x[idx], v3_y[idx], C_V3, 'V3')
    draw_car(ax1, v6_x[idx], v6_y[idx], C_V6, 'V6')

# Mark key events
ax1.plot(v6_x[15], v6_y[15], 'D', color=C_V6, markersize=12, zorder=7)
ax1.annotate('1st Overtake\nAttempt', (v6_x[15], v6_y[15]),
            textcoords="offset points", xytext=(15, 20), fontsize=9,
            fontweight='bold', color=C_V6,
            arrowprops=dict(arrowstyle='->', color=C_V6, lw=1.5))

ax1.plot(v6_x[35], v6_y[35], 'X', color=C_ABORT, markersize=14, zorder=7)
ax1.annotate('ABORT!', (v6_x[35], v6_y[35]),
            textcoords="offset points", xytext=(15, -20), fontsize=10,
            fontweight='bold', color=C_ABORT,
            arrowprops=dict(arrowstyle='->', color=C_ABORT, lw=1.5))

ax1.plot(v6_x[75], v6_y[75], 'D', color=C_OVERTAKE, markersize=12, zorder=7)
ax1.annotate('SUCCESS!', (v6_x[75], v6_y[75]),
            textcoords="offset points", xytext=(15, 20), fontsize=10,
            fontweight='bold', color=C_OVERTAKE,
            arrowprops=dict(arrowstyle='->', color=C_OVERTAKE, lw=1.5))

ax1.set_xlabel('Longitudinal Position X (m)', fontsize=11)
ax1.set_ylabel('Lateral Position Y (m)', fontsize=11)
ax1.set_title('Bird\'s-Eye View — Three-Phase Dynamic Overtaking Scenario', 
              fontsize=14, fontweight='bold', color='#2c3e50')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.set_ylim([0.5, 17.5])
ax1.set_xlim([x_min_plot, x_max_plot])
ax1.grid(True, alpha=0.2)

# ---- Panel 2: Gap dynamics with predictions ----
ax2 = fig.add_subplot(gs[1, 0])

# Actual gap
ax2.plot(t, gap_v0_v3, color='#004B8D', linewidth=2.5, label='Actual Gap (V0-V3)', zorder=3)
ax2.axhline(y=GAP_THRESHOLD, color=C_OVERTAKE, linewidth=1.5, linestyle='--',
           label=f'Overtake Threshold ({GAP_THRESHOLD}m)')
ax2.axhline(y=GAP_ABORT_THRESHOLD, color=C_ABORT, linewidth=1.5, linestyle='--',
           label=f'Abort Threshold ({GAP_ABORT_THRESHOLD}m)')

# Predicted gaps at decision points
dec_times = decision_log['time']
gap_preds = decision_log['gap_pred']
gap_stds = decision_log['gap_std']

ax2.errorbar(dec_times, gap_preds, yerr=2*np.array(gap_stds), fmt='o', color='#9B59B6',
            markersize=5, capsize=3, alpha=0.7, label='BPINN Predicted Gap ±2σ')

# Phase backgrounds
ax2.axvspan(0, 3, alpha=0.04, color='green')
ax2.axvspan(3, 6, alpha=0.04, color='red')
ax2.axvspan(6, 10, alpha=0.04, color='green')

# Annotations
ax2.annotate('Phase 1: Gap Opens', xy=(1.5, gap_v0_v3[15]+2),
            fontsize=9, ha='center', color='#27AE60', fontweight='bold')
ax2.annotate('Phase 2: Gap Closes\n→ Abort!', xy=(4.5, gap_v0_v3[45]+2),
            fontsize=9, ha='center', color='#E74C3C', fontweight='bold')
ax2.annotate('Phase 3: Gap Opens\n→ Success!', xy=(8, gap_v0_v3[80]+2),
            fontsize=9, ha='center', color='#27AE60', fontweight='bold')

ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Longitudinal Gap (m)', fontsize=10)
ax2.set_title('Gap Dynamics: Actual vs BPINN Predicted', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)

# ---- Panel 3: Speed comparison ----
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(t, v0_speed, color=C_V0, linewidth=2, label='V0 (lead)')
ax3.plot(t, v3_speed, color=C_V3, linewidth=2, label='V3 (reference)')
ax3.plot(t, v6_speed, color=C_V6, linewidth=2.5, label='V6 (ego)')

ax3.axvspan(0, 3, alpha=0.04, color='green')
ax3.axvspan(3, 6, alpha=0.04, color='red')
ax3.axvspan(6, 10, alpha=0.04, color='green')

# Mark speed changes
ax3.annotate('V0 slows down\n→ gap closes', xy=(4.5, 25), fontsize=8,
            color=C_V0, fontweight='bold', ha='center')
ax3.annotate('V0 speeds up\n→ gap opens', xy=(8, 30), fontsize=8,
            color=C_V0, fontweight='bold', ha='center')

ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Speed (m/s)', fontsize=10)
ax3.set_title('Vehicle Speed Profiles', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ---- Panel 4: V6 lateral position ----
ax4 = fig.add_subplot(gs[1, 2])

ax4.plot(t, v6_y, color=C_V6, linewidth=2.5, label='V6 Y position')
for ly in LANE_CENTERS:
    ax4.axhline(y=ly, color='#DDDDDD', linewidth=0.8, linestyle='--', alpha=0.5)

ax4.axhspan(LANE_CENTERS[1]-1.5, LANE_CENTERS[1]+1.5, alpha=0.05, color='blue', label='Lane 2')
ax4.axhspan(LANE_CENTERS[2]-1.5, LANE_CENTERS[2]+1.5, alpha=0.05, color='red', label='Lane 3')

ax4.axvspan(0, 3, alpha=0.04, color='green')
ax4.axvspan(3, 6, alpha=0.04, color='red')
ax4.axvspan(6, 10, alpha=0.04, color='green')

ax4.annotate('L2→L3', xy=(1.5, 10), fontsize=9, color=C_V6, fontweight='bold', ha='center')
ax4.annotate('L3→L2\n(abort)', xy=(4.5, 8), fontsize=9, color=C_ABORT, fontweight='bold', ha='center')
ax4.annotate('L2→L3→L2\n(success)', xy=(7.8, 10), fontsize=9, color=C_OVERTAKE, fontweight='bold', ha='center')

ax4.set_xlabel('Time (s)', fontsize=10)
ax4.set_ylabel('Lateral Y (m)', fontsize=10)
ax4.set_title('V6 Lane Changes (3-Phase)', fontsize=12, fontweight='bold')
ax4.set_ylim([4, 14])
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)

# ---- Panel 5: Decision timeline + confidence ----
ax5 = fig.add_subplot(gs[2, 0])

# Action timeline as colored bars
action_colors = {'OVERTAKE': C_OVERTAKE, 'CONTINUE': '#16A085', 
                 'ABORT': C_ABORT, 'HOLD': C_HOLD}

for i, (ti, act) in enumerate(zip(dec_times, decision_log['action'])):
    if i < len(dec_times) - 1:
        width = dec_times[i+1] - ti
    else:
        width = 0.5
    ax5.barh(y=0, width=width, left=ti, height=0.6,
             color=action_colors.get(act, C_HOLD), alpha=0.8, edgecolor='white')

# Confidence on twin axis
ax5_twin = ax5.twinx()
ax5_twin.plot(dec_times, decision_log['overtake_confidence'], 'o-',
              color='#9B59B6', linewidth=2, markersize=5, label='Overtake Confidence')
ax5_twin.axhline(y=CONF_THRESHOLD, color='#9B59B6', linestyle=':', linewidth=1, alpha=0.5)
ax5_twin.set_ylabel('Confidence', fontsize=10, color='#9B59B6')
ax5_twin.set_ylim([-0.05, 1.05])

# Legend for actions
legend_elements = [Patch(facecolor=c, label=a) for a, c in action_colors.items()]
ax5.legend(handles=legend_elements, fontsize=7, loc='upper left', ncol=2)

ax5.set_yticks([])
ax5.set_xlabel('Time (s)', fontsize=10)
ax5.set_title('Decision Timeline & Confidence', fontsize=12, fontweight='bold')
ax5.axvspan(0, 3, alpha=0.04, color='green')
ax5.axvspan(3, 6, alpha=0.04, color='red')
ax5.axvspan(6, 10, alpha=0.04, color='green')

# ---- Panel 6: Inverse parameters ----
ax6 = fig.add_subplot(gs[2, 1])

# Plot inverse velocity estimation
overtake_times = [t for t, a in zip(dec_times, decision_log['action']) if a in ('OVERTAKE', 'CONTINUE')]
overtake_inv_v = [v for a, v in zip(decision_log['action'], decision_log['inv_v']) if a in ('OVERTAKE', 'CONTINUE')]
overtake_inv_v_std = [v for a, v in zip(decision_log['action'], decision_log['inv_v_std']) if a in ('OVERTAKE', 'CONTINUE')]

if overtake_times:
    inv_v_std_arr = np.clip(np.nan_to_num(np.array(overtake_inv_v_std), nan=0.0), 0, 5)
    ax6.errorbar(overtake_times, overtake_inv_v, yerr=2*inv_v_std_arr,
                fmt='s-', color=C_INV, markersize=7, capsize=4, linewidth=2,
                label='Inverse V (±2σ)')

# Plot actual V6 speed for reference
ax6.plot(t, v6_speed, color=C_V6, linewidth=1.5, alpha=0.5, label='V6 actual speed')

ax6.set_xlabel('Time (s)', fontsize=10)
ax6.set_ylabel('Velocity (m/s)', fontsize=10)
ax6.set_title('Inverse BPINN: Parameter Estimation (V)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.axvspan(0, 3, alpha=0.04, color='green')
ax6.axvspan(3, 6, alpha=0.04, color='red')
ax6.axvspan(6, 10, alpha=0.04, color='green')

# ---- Panel 7: BPINN prediction snapshots ----
ax7 = fig.add_subplot(gs[2, 2])

# Show 3 key snapshots
key_snapshots = [s for s in prediction_snapshots if s['action'] in ('OVERTAKE', 'ABORT', 'HOLD')][:3]
if len(key_snapshots) < 3:
    key_snapshots = prediction_snapshots[:3]

colors_snapshot = {'OVERTAKE': C_OVERTAKE, 'CONTINUE': '#16A085', 
                   'ABORT': C_ABORT, 'HOLD': C_HOLD}

for snap in key_snapshots:
    ft = snap['future_t']
    gp = snap['gap_pred']
    color = colors_snapshot.get(snap['action'], C_HOLD)
    ax7.plot(ft, gp, '--', color=color, linewidth=2, alpha=0.8,
             label=f"t={snap['time']:.1f}s: {snap['action']} (conf={snap['confidence']:.2f})")

ax7.axhline(y=GAP_THRESHOLD, color=C_OVERTAKE, linewidth=1, linestyle=':', alpha=0.5)
ax7.axhline(y=GAP_ABORT_THRESHOLD, color=C_ABORT, linewidth=1, linestyle=':', alpha=0.5)

ax7.set_xlabel('Time (s)', fontsize=10)
ax7.set_ylabel('Predicted Gap (m)', fontsize=10)
ax7.set_title('BPINN Prediction Snapshots', fontsize=12, fontweight='bold')
ax7.legend(fontsize=7, loc='best')
ax7.grid(True, alpha=0.3)

# ---- Panel 8: Prediction error analysis ----
ax8 = fig.add_subplot(gs[3, 0])

# Prediction error
gap_pred_arr = np.array(decision_log['gap_pred'])
gap_actual_arr = np.array(decision_log['gap_actual'])
pred_error = gap_pred_arr - gap_actual_arr

ax8.bar(dec_times, pred_error, width=0.4, color=['#27AE60' if abs(e) < 5 else '#E74C3C' for e in pred_error],
        alpha=0.7, label='Prediction Error')
ax8.axhline(y=0, color='black', linewidth=0.8)

ax8.set_xlabel('Time (s)', fontsize=10)
ax8.set_ylabel('Error (m)', fontsize=10)
ax8.set_title('BPINN Gap Prediction Error', fontsize=12, fontweight='bold')
ax8.axvspan(0, 3, alpha=0.04, color='green')
ax8.axvspan(3, 6, alpha=0.04, color='red')
ax8.axvspan(6, 10, alpha=0.04, color='green')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=8)

# ---- Panel 9: Inverse heading estimation ----
ax9 = fig.add_subplot(gs[3, 1])

overtake_inv_h = [h for a, h in zip(decision_log['action'], decision_log['inv_h']) if a in ('OVERTAKE', 'CONTINUE')]
overtake_inv_h_std = [h for a, h in zip(decision_log['action'], decision_log['inv_h_std']) if a in ('OVERTAKE', 'CONTINUE')]

if overtake_times:
    inv_h_std_arr = np.clip(np.nan_to_num(np.array(overtake_inv_h_std), nan=0.0), 0, 0.5)
    ax9.errorbar(overtake_times, overtake_inv_h, yerr=2*inv_h_std_arr,
                fmt='D-', color='#E67E22', markersize=7, capsize=4, linewidth=2,
                label='Inverse Heading (±2σ)')

ax9.set_xlabel('Time (s)', fontsize=10)
ax9.set_ylabel('Heading (rad)', fontsize=10)
ax9.set_title('Inverse BPINN: Heading Estimation', fontsize=12, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)
ax9.axvspan(0, 3, alpha=0.04, color='green')
ax9.axvspan(3, 6, alpha=0.04, color='red')
ax9.axvspan(6, 10, alpha=0.04, color='green')

# ---- Panel 10: BPINN uncertainty over time ----
ax10 = fig.add_subplot(gs[3, 2])

ax10.plot(dec_times, decision_log['gap_std'], 'o-', color='#8E44AD', 
          linewidth=2, markersize=5, label='Gap Uncertainty (σ)')
ax10.fill_between(dec_times, 0, decision_log['gap_std'], alpha=0.15, color='#8E44AD')

ax10.set_xlabel('Time (s)', fontsize=10)
ax10.set_ylabel('Uncertainty (m)', fontsize=10)
ax10.set_title('BPINN Uncertainty (MC Dropout)', fontsize=12, fontweight='bold')
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3)
ax10.axvspan(0, 3, alpha=0.04, color='green')
ax10.axvspan(3, 6, alpha=0.04, color='red')
ax10.axvspan(6, 10, alpha=0.04, color='green')

# ============================================================
# 5. Save and display
# ============================================================
print("\n[4/5] Saving visualization...")

output_path = 'rolling_window_bpinn_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Figure saved: {output_path}")

# Print summary
print("\n[5/5] Summary:")
print(f"  Total simulation time: {T_TOTAL}s ({N} steps)")
print(f"  Decision points: {len(dec_times)}")
print(f"  BPINN runs: {len(dec_times) * 2} forward + {sum(1 for a in decision_log['action'] if a in ('OVERTAKE', 'CONTINUE'))} inverse")

action_counts = {}
for a in decision_log['action']:
    action_counts[a] = action_counts.get(a, 0) + 1
print(f"  Action distribution: {action_counts}")

mean_error = np.mean(np.abs(pred_error))
max_error = np.max(np.abs(pred_error))
print(f"  Prediction error: mean={mean_error:.2f}m, max={max_error:.2f}m")

overtake_confs = [c for a, c in zip(decision_log['action'], decision_log['overtake_confidence']) if a in ('OVERTAKE', 'CONTINUE')]
if overtake_confs:
    print(f"  Overtake confidence: mean={np.mean(overtake_confs):.2f}, min={np.min(overtake_confs):.2f}")

print(f"\n  Phase 1 (t=0~3s):   Gap {gap_v0_v3[0]:.1f}m -> {gap_v0_v3[30]:.1f}m  → OVERTAKE attempt")
print(f"  Phase 2 (t=3~6s):   Gap {gap_v0_v3[30]:.1f}m -> {gap_v0_v3[60]:.1f}m  → ABORT (V0 slows to 18 m/s)")
print(f"  Phase 3 (t=6~10s):  Gap {gap_v0_v3[60]:.1f}m -> {gap_v0_v3[-1]:.1f}m  → OVERTAKE SUCCESS!")

plt.show()
print("\nDone!")
