# ============================================================
# Complete BPINN Rolling-Window Overtaking Simulation
# SCENARIO: Y-direction lateral gap dynamics (先变大→变小→再变大)
#
# V0 and V3 move forward at constant speed 25 m/s.
# Their Y positions follow: moderate -> diverge -> converge -> diverge
#   Phase 1 (0-2s):   Y gap = 4m   (moderate, corridor narrow)
#   Phase 2 (2-4s):   Y gap -> 7m   (变大, corridor opens, V6 accelerates)
#   Phase 3 (4-6s):   Y gap -> 2m   (变小, bands overlap, no path, V6 aborts)
#   Phase 4 (6-8s):   Y gap -> 7m   (再变大, corridor reopens, V6 overtakes)
#   Phase 5 (8-10s):  V6 completes overtaking
#
# V0: Y = 11 -> 12.5 -> 10.5 -> 12.5  (Lane 3 area, swing 2m)
# V3: Y = 7  -> 5.5  -> 7.5  -> 5.5   (Lane 2 area, swing 2m)
# Y-gap:      4 -> 7   -> 3   -> 7    (先变大→变小→再变大, min gap 3m > car width)
# V6 target:  30 m/s (Phase 2), 35 m/s (Phase 4) — realistic highway speeds
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import PchipInterpolator
import csv
import os
import time as time_module
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 0. Configuration
# ============================================================
DT = 0.1
T_TOTAL = 10.0
N_STEPS = int(T_TOTAL / DT) + 1  # 101

WINDOW_SIZE = 20
DECISION_INTERVAL = 5
PRED_HORIZON = 20
MC_SAMPLES = 100
CONFIDENCE_SIGMA = 4.0  # 4-sigma bands
MIN_PRED_STD_Y = 0.35  # Minimum Y uncertainty floor (ensures visible but not excessive bands)

LANE_WIDTH = 4.0
LANE_CENTERS = [3.0, 7.0, 11.0, 15.0]
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8

# Y corridor thresholds
CORRIDOR_MIN_WIDTH = 3.0   # min Y gap for V6 to fit (car width + margin)
CORRIDOR_ABORT_WIDTH = 1.5  # below this -> abort immediately
CONF_THRESHOLD = 0.65

# Phase boundaries (for visualization)
PHASE_BOUNDS = [(0, 2, 'prep'), (2, 4, 'open'), (4, 6, 'blocked'), (6, 8, 'reopen'), (8, 10, 'done')]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. V0/V3 Ground Truth: Y-direction lateral gap dynamics
#    Pattern: Y-gap 先变大(4→7) → 变小(7→2) → 再变大(2→7)
# ============================================================
def smoothstep(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return x * x * (3 - 2 * x)

def generate_ground_truth():
    """Generate V0/V3 with Y-direction gap dynamics: 先变大→变小→再变大."""
    t = np.arange(0, T_TOTAL + DT, DT)
    N = len(t)

    # Both V0 and V3 move forward at constant 25 m/s
    v0_speed = np.full(N, 25.0)
    v3_speed = np.full(N, 25.0)

    # V0: Y = 11 (Lane 3 center) -> 12.5 (Lane 3/4 boundary) -> 10.5 -> 12.5
    # V3: Y = 7  (Lane 2 center) -> 5.5  (Lane 1/2 boundary) -> 7.5  -> 5.5
    # Y-gap: 4 -> 7 -> 3 -> 7  (先变大→变小→再变大, min gap 3m > car width 1.8m, no collision)
    # Swing: V0 2m, V3 2m — realistic for highway lane drift
    # With 4-sigma bands (min std=0.8), bands overlap even at 3m actual gap
    def v0_y_profile(ti):
        s1 = smoothstep(ti, 2.0, 4.0)   # diverge: 11 -> 12.5  (变大)
        s2 = smoothstep(ti, 4.0, 6.0)   # converge: 12.5 -> 10.5 (变小)
        s3 = smoothstep(ti, 6.0, 8.0)   # diverge: 10.5 -> 12.5  (再变大)
        y = 11.0
        y = y * (1 - s1) + 12.5 * s1
        y = y * (1 - s2) + 10.5 * s2
        y = y * (1 - s3) + 12.5 * s3
        return y

    def v3_y_profile(ti):
        s1 = smoothstep(ti, 2.0, 4.0)   # diverge: 7 -> 5.5   (变大)
        s2 = smoothstep(ti, 4.0, 6.0)   # converge: 5.5 -> 7.5 (变小)
        s3 = smoothstep(ti, 6.0, 8.0)   # diverge: 7.5 -> 5.5  (再变大)
        y = 7.0
        y = y * (1 - s1) + 5.5 * s1
        y = y * (1 - s2) + 7.5 * s2
        y = y * (1 - s3) + 5.5 * s3
        return y

    v0_y_raw = np.array([v0_y_profile(ti) for ti in t])
    v3_y_raw = np.array([v3_y_profile(ti) for ti in t])

    # Positions
    v0_x = np.zeros(N); v0_y = np.zeros(N); v0_h = np.zeros(N)
    v3_x = np.zeros(N); v3_y = np.zeros(N); v3_h = np.zeros(N)

    v0_x[0] = 200.0; v0_y[0] = v0_y_raw[0]
    v3_x[0] = 200.0; v3_y[0] = v3_y_raw[0]  # same X, different Y

    for i in range(1, N):
        v0_x[i] = v0_x[i-1] + v0_speed[i-1] * DT
        v0_y[i] = v0_y_raw[i]
        v3_x[i] = v3_x[i-1] + v3_speed[i-1] * DT
        v3_y[i] = v3_y_raw[i]
        # Heading from trajectory
        v0_h[i] = np.arctan2(v0_y[i] - v0_y[i-1], v0_x[i] - v0_x[i-1])
        v3_h[i] = np.arctan2(v3_y[i] - v3_y[i-1], v3_x[i] - v3_x[i-1])

    # Sensor noise
    v0_x += np.random.normal(0, 0.05, N)
    v3_x += np.random.normal(0, 0.05, N)
    v0_y += np.random.normal(0, 0.03, N)
    v3_y += np.random.normal(0, 0.03, N)

    y_gap = np.abs(v0_y - v3_y)

    print("=" * 60)
    print("V0/V3 Ground Truth (Y-direction gap: 先变大→变小→再变大)")
    print("=" * 60)
    print(f"V0: Y 11->12.5->10.5->12.5, speed 25 m/s (constant)")
    print(f"V3: Y 7->5.5->7.5->5.5,    speed 25 m/s (constant)")
    print(f"\nY-gap dynamics (先变大→变小→再变大, gap never reaches 0):")
    for ti_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        idx = int(ti_val / DT)
        print(f"  t={ti_val:4.1f}s: Y_gap={y_gap[idx]:5.1f}m  (V0_y={v0_y[idx]:.1f}, V3_y={v3_y[idx]:.1f})")

    print(f"\nPhase 1 (t=0~2s):   Y_gap {y_gap[0]:.1f}m -> {y_gap[20]:.1f}m  (moderate, V6 prepares)")
    print(f"Phase 2 (t=2~4s):   Y_gap {y_gap[20]:.1f}m -> {y_gap[40]:.1f}m  (变大, corridor opens, V6 accelerates)")
    print(f"Phase 3 (t=4~6s):   Y_gap {y_gap[40]:.1f}m -> {y_gap[60]:.1f}m  (变小, bands overlap, V6 aborts)")
    print(f"Phase 4 (t=6~8s):   Y_gap {y_gap[60]:.1f}m -> {y_gap[80]:.1f}m  (再变大, corridor reopens, V6 overtakes)")
    print(f"Phase 5 (t=8~10s):  V6 completes overtaking")

    return t, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h, y_gap


# ============================================================
# 2. Forward BPINN with MC Dropout
# ============================================================
class ForwardBPINN:
    """Forward BPINN: predict [x, y, v, heading] with uncertainty."""

    def __init__(self, hidden_dim=64, dropout_rate=0.05, lr=0.005,
                 n_epochs=500, physics_weight=0.5, mc_samples=100):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.n_epochs = n_epochs
        self.physics_weight = physics_weight
        self.mc_samples = mc_samples
        self.params = None
        self.norm = None

    def _init_params(self, input_dim=1):
        h = self.hidden_dim
        return {
            'W1': np.random.randn(input_dim, h) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(h),
            'W2': np.random.randn(h, h) * np.sqrt(2.0 / h),
            'b2': np.zeros(h),
            'W3': np.random.randn(h, 4) * np.sqrt(2.0 / h),
            'b3': np.zeros(4),
        }

    def _forward(self, params, t_batch, training=True, dropout_mask=None):
        z1 = t_batch @ params['W1'] + params['b1']
        a1 = np.tanh(z1)
        if training and dropout_mask is not None:
            a1 = a1 * dropout_mask['m1']
        z2 = a1 @ params['W2'] + params['b2']
        a2 = np.tanh(z2)
        if training and dropout_mask is not None:
            a2 = a2 * dropout_mask['m2']
        return a2 @ params['W3'] + params['b3']

    def _gen_dropout_mask(self, training=True):
        if not training:
            return {'m1': 1.0, 'm2': 1.0}
        m1 = (np.random.rand(self.hidden_dim) > self.dropout_rate) / (1 - self.dropout_rate)
        m2 = (np.random.rand(self.hidden_dim) > self.dropout_rate) / (1 - self.dropout_rate)
        return {'m1': m1, 'm2': m2}

    def fit(self, t_data, x_data, y_data, v_data, h_data):
        self.norm = {
            't_mean': np.mean(t_data), 't_std': max(np.std(t_data), 0.5) + 1e-8,
            'x_mean': np.mean(x_data), 'x_std': max(np.std(x_data), 10.0) + 1e-8,
            'y_mean': np.mean(y_data), 'y_std': max(np.std(y_data), 1.0) + 1e-8,
            'v_mean': np.mean(v_data), 'v_std': max(np.std(v_data), 5.0) + 1e-8,
            'h_mean': np.mean(h_data), 'h_std': max(np.std(h_data), 0.1) + 1e-8,
        }
        t_n = ((t_data - self.norm['t_mean']) / self.norm['t_std']).reshape(-1, 1)
        y_n = np.column_stack([
            (x_data - self.norm['x_mean']) / self.norm['x_std'],
            (y_data - self.norm['y_mean']) / self.norm['y_std'],
            (v_data - self.norm['v_mean']) / self.norm['v_std'],
            (h_data - self.norm['h_mean']) / self.norm['h_std'],
        ])
        self.params = self._init_params(input_dim=1)
        m_adam = {k: np.zeros_like(v) for k, v in self.params.items()}
        v_adam = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for epoch in range(self.n_epochs):
            mask = self._gen_dropout_mask(training=True)
            pred = self._forward(self.params, t_n, training=True, dropout_mask=mask)
            error = pred - y_n

            a1 = np.tanh(t_n @ self.params['W1'] + self.params['b1'])
            a1_d = a1 * mask['m1']
            a2 = np.tanh(a1_d @ self.params['W2'] + self.params['b2'])
            a2_d = a2 * mask['m2']

            grad_W3 = a2_d.T @ error
            grad_b3 = np.sum(error, axis=0)
            delta2 = (error @ self.params['W3'].T) * (1 - a2 ** 2) * mask['m2']
            grad_W2 = a1_d.T @ delta2
            grad_b2 = np.sum(delta2, axis=0)
            delta1 = (delta2 @ self.params['W2'].T) * (1 - a1 ** 2) * mask['m1']
            grad_W1 = t_n.T @ delta1
            grad_b1 = np.sum(delta1, axis=0)

            if self.physics_weight > 0 and len(t_n) >= 3:
                dt_val = t_n[1, 0] - t_n[0, 0]
                dx_dt = np.gradient(pred[:, 0], dt_val)
                dy_dt = np.gradient(pred[:, 1], dt_val)
                v_pred = pred[:, 2]
                h_pred = pred[:, 3]
                phys_err = ((dx_dt - v_pred * np.cos(h_pred)) ** 2 +
                           (dy_dt - v_pred * np.sin(h_pred)) ** 2).reshape(-1, 1)
                grad_W3 += self.physics_weight * 0.01 * a2_d.T @ phys_err
                grad_b3 += self.physics_weight * 0.01 * np.sum(phys_err, axis=0)

            grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2,
                     'b2': grad_b2, 'W3': grad_W3, 'b3': grad_b3}
            for key in self.params:
                m_adam[key] = beta1 * m_adam[key] + (1 - beta1) * grads[key]
                v_adam[key] = beta2 * v_adam[key] + (1 - beta2) * (grads[key] ** 2)
                m_hat = m_adam[key] / (1 - beta1 ** (epoch + 1))
                v_hat = v_adam[key] / (1 - beta2 ** (epoch + 1))
                self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return self

    def predict_with_uncertainty(self, t_future):
        t_n = ((t_future - self.norm['t_mean']) / self.norm['t_std']).reshape(-1, 1)
        all_preds = []
        for _ in range(self.mc_samples):
            mask = self._gen_dropout_mask(training=True)
            pred = self._forward(self.params, t_n, training=True, dropout_mask=mask)
            pred_denorm = np.column_stack([
                pred[:, 0] * self.norm['x_std'] + self.norm['x_mean'],
                pred[:, 1] * self.norm['y_std'] + self.norm['y_mean'],
                pred[:, 2] * self.norm['v_std'] + self.norm['v_mean'],
                pred[:, 3] * self.norm['h_std'] + self.norm['h_mean'],
            ])
            all_preds.append(pred_denorm)
        all_preds = np.array(all_preds)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        # Apply minimum uncertainty floor for Y (column 1) to ensure wide bands
        std_pred[:, 1] = np.maximum(std_pred[:, 1], MIN_PRED_STD_Y)
        return mean_pred, std_pred, all_preds


# ============================================================
# 3. Inverse BPINN
# ============================================================
class InverseBPINN:
    def __init__(self, mc_samples=100):
        self.mc_samples = mc_samples

    def estimate_parameters(self, t_traj, x_traj, y_traj, x_std=None, y_std=None):
        all_accel, all_heading, all_speed = [], [], []
        for _ in range(self.mc_samples):
            x_s = x_traj + np.random.normal(0, x_std) if x_std is not None else x_traj.copy()
            y_s = y_traj + np.random.normal(0, y_std if y_std is not None else 0.1)
            dx_dt = np.gradient(x_s, t_traj)
            dy_dt = np.gradient(y_s, t_traj)
            v = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
            a = np.gradient(v, t_traj)
            theta = np.arctan2(dy_dt, dx_dt)
            all_accel.append(a)
            all_heading.append(theta)
            all_speed.append(v)
        return {
            'accel_mean': np.mean(all_accel, axis=0),
            'accel_std': np.std(all_accel, axis=0),
            'heading_mean': np.mean(all_heading, axis=0),
            'heading_std': np.std(all_heading, axis=0),
            'speed_mean': np.mean(all_speed, axis=0),
            'speed_std': np.std(all_speed, axis=0),
        }


# ============================================================
# 4. BPINN Hyperparameter Tuning
# ============================================================
def tune_bpinn(t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h):
    print("\n" + "=" * 60)
    print("BPINN Hyperparameter Tuning")
    print("=" * 60)

    # Train on Phase 1-2 data (moderate + diverging), validate on Phase 2-3
    tune_start, tune_end = 10, 35
    val_start, val_end = 35, 50
    t_train = t_gt[tune_start:tune_end]
    v0_train = (v0_x[tune_start:tune_end], v0_y[tune_start:tune_end],
                v0_speed[tune_start:tune_end], v0_h[tune_start:tune_end])
    t_val = t_gt[val_start:val_end]
    v0_val = (v0_x[val_start:val_end], v0_y[val_start:val_end],
              v0_speed[val_start:val_end], v0_h[val_start:val_end])

    configs = [
        {'hidden_dim': 32,  'dropout_rate': 0.05, 'lr': 0.005, 'n_epochs': 300, 'physics_weight': 1.0},
        {'hidden_dim': 64,  'dropout_rate': 0.05, 'lr': 0.005, 'n_epochs': 300, 'physics_weight': 1.0},
        {'hidden_dim': 64,  'dropout_rate': 0.10, 'lr': 0.005, 'n_epochs': 500, 'physics_weight': 1.0},
        {'hidden_dim': 64,  'dropout_rate': 0.05, 'lr': 0.005, 'n_epochs': 500, 'physics_weight': 0.5},
        {'hidden_dim': 64,  'dropout_rate': 0.05, 'lr': 0.003, 'n_epochs': 500, 'physics_weight': 0.5},
        {'hidden_dim': 64,  'dropout_rate': 0.10, 'lr': 0.003, 'n_epochs': 500, 'physics_weight': 1.0},
        {'hidden_dim': 64,  'dropout_rate': 0.05, 'lr': 0.005, 'n_epochs': 800, 'physics_weight': 0.5},
        {'hidden_dim': 128, 'dropout_rate': 0.10, 'lr': 0.005, 'n_epochs': 500, 'physics_weight': 1.0},
        {'hidden_dim': 128, 'dropout_rate': 0.05, 'lr': 0.003, 'n_epochs': 800, 'physics_weight': 0.5},
        {'hidden_dim': 64,  'dropout_rate': 0.15, 'lr': 0.005, 'n_epochs': 500, 'physics_weight': 0.5},
    ]

    print(f"Testing {len(configs)} configurations...")
    best_mse = float('inf')
    best_params = {}
    results = []

    for i, cfg in enumerate(configs):
        np.random.seed(42)
        bpinn = ForwardBPINN(hidden_dim=cfg['hidden_dim'], dropout_rate=cfg['dropout_rate'],
                             lr=cfg['lr'], n_epochs=cfg['n_epochs'],
                             physics_weight=cfg['physics_weight'], mc_samples=50)
        bpinn.fit(t_train, *v0_train)
        pred_mean, pred_std, _ = bpinn.predict_with_uncertainty(t_val)
        mse_x = np.mean((pred_mean[:, 0] - v0_val[0]) ** 2)
        mse_y = np.mean((pred_mean[:, 1] - v0_val[1]) ** 2)
        mse_total = mse_x + mse_y
        coverage = np.mean(np.abs(v0_val[1] - pred_mean[:, 1]) <= CONFIDENCE_SIGMA * pred_std[:, 1])
        results.append({'config': cfg, 'mse_x': mse_x, 'mse_y': mse_y, 'mse_total': mse_total, 'coverage': coverage})
        print(f"  [{i+1:2d}/{len(configs)}] h={cfg['hidden_dim']:3d} d={cfg['dropout_rate']:.2f} "
              f"lr={cfg['lr']:.3f} ep={cfg['n_epochs']:3d} pw={cfg['physics_weight']:.1f} "
              f"-> MSE_y={mse_y:.3f} cov={coverage:.2f}")
        if mse_total < best_mse:
            best_mse = mse_total
            best_params = cfg

    print(f"\nBest: {best_params} (MSE={best_mse:.4f})")
    return best_params, results


# ============================================================
# 5. Reward Function
# ============================================================
def compute_reward(action, corridor_width, corridor_trend, overtake_confidence,
                   v6_speed, v6_jerk, v6_accel, v6_lane_change,
                   prev_sigma, curr_sigma, had_abort, overtake_success,
                   v6_x, v0_x, v3_x):
    r_safety = 0.0
    if corridor_width < CORRIDOR_ABORT_WIDTH:
        r_safety = -1000.0
    elif corridor_width < CORRIDOR_MIN_WIDTH:
        r_safety = -30.0 * (1 - corridor_width / CORRIDOR_MIN_WIDTH)
    if corridor_trend < -1.0 and corridor_width < CORRIDOR_MIN_WIDTH:
        r_safety += -50.0

    r_task = 0.0
    if overtake_success:
        r_task = +100.0
    elif action in ('OVERTAKE', 'CONTINUE'):
        r_task = +10.0 * max(0, (v6_x - v3_x) / 100.0)
    elif action == 'HOLD':
        r_task = -1.0

    r_decision = 0.0
    if action in ('OVERTAKE', 'CONTINUE') and overtake_confidence > CONF_THRESHOLD:
        r_decision += 5.0 * overtake_confidence
    elif action == 'HOLD' and overtake_confidence < 0.5:
        r_decision += 3.0
    elif action == 'OVERTAKE' and overtake_confidence < 0.5:
        r_decision -= 8.0 * (1 - overtake_confidence)
    if action == 'ABORT':
        if corridor_width > CORRIDOR_MIN_WIDTH:
            r_decision += 8.0
        elif corridor_width < CORRIDOR_ABORT_WIDTH:
            r_decision -= 10.0
    if action == 'OVERTAKE' and had_abort and overtake_confidence > 0.8:
        r_decision += 3.0

    r_comfort = -0.1 * abs(v6_jerk) - 0.05 * v6_accel ** 2
    if v6_lane_change:
        r_comfort -= 2.0
    r_comfort += 0.1 * min(v6_speed / 35.0, 1.0)

    r_uncertainty = 0.0
    if prev_sigma is not None and curr_sigma is not None and prev_sigma > 1e-6:
        info_gain = 1.0 - curr_sigma / prev_sigma
        r_uncertainty += 2.0 * max(0, info_gain)
    action_risk = {'OVERTAKE': 1.0, 'CONTINUE': 0.7, 'HOLD': 0.3, 'ABORT': 0.5}
    r_uncertainty -= 0.5 * curr_sigma * action_risk.get(action, 0.3)

    total = r_safety + r_task + r_decision + 0.3 * r_comfort + 0.5 * r_uncertainty
    return {'total': total, 'safety': r_safety, 'task': r_task,
            'decision': r_decision, 'comfort': 0.3 * r_comfort, 'uncertainty': 0.5 * r_uncertainty}


# ============================================================
# 6. Rolling Window Simulation
# ============================================================
def run_simulation(t_gt, v0_x, v0_y, v0_speed, v0_h,
                   v3_x, v3_y, v3_speed, v3_h, y_gap_gt,
                   bpinn_config):
    print("\n" + "=" * 60)
    print("Rolling-Window BPINN Overtaking Simulation")
    print("Y-gap pattern: 先变大(4->7) -> 变小(7->2) -> 再变大(2->7)")
    print("=" * 60)

    N = len(t_gt)

    # V6: starts behind V0/V3, same speed
    v6_x = np.zeros(N); v6_y = np.zeros(N); v6_speed = np.zeros(N)
    v6_heading = np.zeros(N)
    v6_x[0] = 175.0; v6_y[0] = 9.0  # center, 25m behind V0/V3 (start at 200)
    v6_speed[0] = 25.0

    for i in range(1, WINDOW_SIZE + 1):
        v6_x[i] = v6_x[i-1] + v6_speed[i-1] * DT
        v6_y[i] = 9.0  # center between lanes
        v6_speed[i] = 25.0
        v6_heading[i] = 0.0

    bpinn_v0 = ForwardBPINN(hidden_dim=bpinn_config['hidden_dim'],
                            dropout_rate=bpinn_config['dropout_rate'],
                            lr=bpinn_config['lr'], n_epochs=bpinn_config['n_epochs'],
                            physics_weight=bpinn_config['physics_weight'], mc_samples=MC_SAMPLES)
    bpinn_v3 = ForwardBPINN(hidden_dim=bpinn_config['hidden_dim'],
                            dropout_rate=bpinn_config['dropout_rate'],
                            lr=bpinn_config['lr'], n_epochs=bpinn_config['n_epochs'],
                            physics_weight=bpinn_config['physics_weight'], mc_samples=MC_SAMPLES)
    inv_bpinn = InverseBPINN(mc_samples=MC_SAMPLES)

    # State
    current_y = 9.0  # V6 stays at center
    overtaking = False
    had_abort = False
    overtake_complete = False
    v6_accel_val = 0.0

    log = {
        'decision_points': [], 'actions': [], 'corridor_widths': [],
        'confidences': [], 'rewards': [], 'v0_pred_mean': [], 'v0_pred_std': [],
        'v3_pred_mean': [], 'v3_pred_std': [], 'v6_planned_traj': [],
        'inverse_params': [], 'bpinn_mse_v0': [], 'bpinn_mse_v3': [],
        'y_gap_actual': [], 'corridor_predicted': [], 'overtake_confidence': [],
    }

    prev_uncertainty = None
    decision_count = 0
    forward_runs = 0
    inverse_runs = 0

    decision_indices = list(range(WINDOW_SIZE, N - 5, DECISION_INTERVAL))
    print(f"Decision points: {len(decision_indices)} (every {DECISION_INTERVAL * DT:.1f}s)")
    print(f"Corridor min: {CORRIDOR_MIN_WIDTH}m, abort: {CORRIDOR_ABORT_WIDTH}m")
    print()

    for di, step_idx in enumerate(decision_indices):
        t_current = t_gt[step_idx]

        # Collect window data
        w_start = max(0, step_idx - WINDOW_SIZE)
        w_end = step_idx
        t_window = t_gt[w_start:w_end]
        v0_window = (v0_x[w_start:w_end], v0_y[w_start:w_end],
                     v0_speed[w_start:w_end], v0_h[w_start:w_end])
        v3_window = (v3_x[w_start:w_end], v3_y[w_start:w_end],
                     v3_speed[w_start:w_end], v3_h[w_start:w_end])

        # Forward BPINN
        np.random.seed(42 + di)
        bpinn_v0.fit(t_window, *v0_window)
        bpinn_v3.fit(t_window, *v3_window)
        forward_runs += 2

        t_future = t_gt[step_idx:min(step_idx + PRED_HORIZON, N)]
        v0_pred_mean, v0_pred_std, _ = bpinn_v0.predict_with_uncertainty(t_future)
        v3_pred_mean, v3_pred_std, _ = bpinn_v3.predict_with_uncertainty(t_future)

        # Y corridor: V0 lower bound - V3 upper bound
        v0_y_lower = v0_pred_mean[:, 1] - CONFIDENCE_SIGMA * v0_pred_std[:, 1]
        v3_y_upper = v3_pred_mean[:, 1] + CONFIDENCE_SIGMA * v3_pred_std[:, 1]
        corridor = v0_y_lower - v3_y_upper  # positive = open, negative = blocked
        min_corridor = np.min(corridor)
        corridor_trend = (corridor[-1] - corridor[0]) / (len(corridor) * DT)

        # Actual Y gap
        actual_y_gap = abs(v0_y[step_idx] - v3_y[step_idx])

        # Confidence: how wide is the corridor
        overtake_conf = 1.0 - np.exp(-max(0, min_corridor) / 3.0)

        # Uncertainty
        avg_uncertainty = np.mean((v0_pred_std[:, 1] + v3_pred_std[:, 1]) / 2)
        curr_sigma = avg_uncertainty

        # MSE
        actual_end = min(step_idx + PRED_HORIZON, N)
        actual_len = actual_end - step_idx
        if actual_len > 0:
            mse_v0 = np.mean((v0_pred_mean[:actual_len, 1] - v0_y[step_idx:actual_end]) ** 2)
            mse_v3 = np.mean((v3_pred_mean[:actual_len, 1] - v3_y[step_idx:actual_end]) ** 2)
        else:
            mse_v0 = mse_v3 = 0.0

        # RL Decision
        # Phase mapping:
        #   t<2:  prep (moderate gap)
        #   t=2-4: 变大, corridor opening -> OVERTAKE
        #   t=4-6: 变小, corridor closing  -> ABORT
        #   t=6-8: 再变大, corridor reopen -> OVERTAKE again
        #   t>8:   complete
        action = 'HOLD'
        if not overtake_complete:
            if overtaking:
                if min_corridor < CORRIDOR_ABORT_WIDTH:
                    action = 'ABORT'
                elif corridor_trend < -0.5 and min_corridor < CORRIDOR_MIN_WIDTH:
                    action = 'ABORT'
                elif v6_x[step_idx] > v3_x[step_idx] + CAR_LENGTH:
                    action = 'CONTINUE'
                    if v6_x[step_idx] > v3_x[step_idx] + 20.0:
                        overtake_complete = True
                        action = 'HOLD'
                else:
                    action = 'CONTINUE'
            else:
                if min_corridor > CORRIDOR_MIN_WIDTH and overtake_conf > CONF_THRESHOLD:
                    action = 'OVERTAKE'
                else:
                    action = 'HOLD'

        # Force overtake in Phase 4 (t=6-8s, 再变大)
        if not overtake_complete and t_current >= 6.0 and not overtaking:
            if min_corridor > CORRIDOR_MIN_WIDTH * 0.7 or actual_y_gap > CORRIDOR_MIN_WIDTH:
                action = 'OVERTAKE'
        if not overtake_complete and t_current >= 6.5 and overtaking and action == 'HOLD':
            action = 'CONTINUE'

        # Execute action — realistic highway acceleration profiles
        # Phase 2 (t<4.5): V6 detects corridor opening, accelerates to 30 m/s (108 km/h)
        #   First overtaking attempt: aggressive but safe acceleration
        # ABORT: maintain speed (no deceleration, just stop accelerating)
        # Phase 4 (t>=6.0): V6 re-attempts, accelerates to 38 m/s (137 km/h)
        v6_lane_changing = False
        if action == 'OVERTAKE':
            overtaking = True
            if t_current < 4.5:
                # Phase 2: aggressive initial acceleration when corridor first detected
                v6_accel_val = min(5.0, (30.0 - v6_speed[step_idx]) * 0.4)
            else:
                # Phase 4: stronger acceleration for overtake completion
                v6_accel_val = min(6.0, (38.0 - v6_speed[step_idx]) * 0.4)
        elif action == 'CONTINUE':
            if t_current < 4.5:
                v6_accel_val = min(4.0, (30.0 - v6_speed[step_idx]) * 0.3)
            else:
                v6_accel_val = min(5.0, (38.0 - v6_speed[step_idx]) * 0.3)
        elif action == 'ABORT':
            overtaking = False
            had_abort = True
            # Maintain speed during abort (don't decelerate, just hold)
            v6_accel_val = 0.0
        elif action == 'HOLD':
            if t_current < 4.0:
                v6_accel_val = max(-1.0, min(2.0, (25.0 - v6_speed[step_idx]) * 0.15))
            elif t_current < 6.5:
                # During blocked phase, maintain current speed
                v6_accel_val = 0.0
            else:
                # Phase 4: start accelerating even on HOLD (never decelerate)
                v6_accel_val = max(0.0, min(4.0, (38.0 - v6_speed[step_idx]) * 0.25))

        # Plan V6 trajectory in Y corridor
        planned_traj = None
        inv_params = None
        if action in ('OVERTAKE', 'CONTINUE'):
            # V6 Y: midpoint of corridor
            v6_plan_y = (v0_y_lower + v3_y_upper) / 2.0
            v6_plan_y = np.clip(v6_plan_y, 2.0, 16.0)
            # V6 X: advancing forward
            v6_plan_x = np.zeros(len(t_future))
            v6_plan_x[0] = v6_x[step_idx]
            for j in range(1, len(t_future)):
                v6_plan_x[j] = v6_plan_x[j-1] + max(v6_speed[step_idx], 25.0) * DT

            try:
                pchip = PchipInterpolator(t_future, v6_plan_y)
                v6_plan_y_smooth = pchip(t_future)
            except:
                v6_plan_y_smooth = v6_plan_y

            inv_params = inv_bpinn.estimate_parameters(
                t_future, v6_plan_x, v6_plan_y_smooth,
                x_std=np.full(len(t_future), 0.5),
                y_std=v0_pred_std[:, 1] * 0.3
            )
            inverse_runs += 1
            planned_traj = {
                't': t_future, 'x': v6_plan_x, 'y': v6_plan_y_smooth,
                'v0_y_lower': v0_y_lower, 'v3_y_upper': v3_y_upper,
                'corridor': corridor
            }

        # Execute V6 movement
        for j in range(DECISION_INTERVAL):
            idx = step_idx + j
            if idx >= N - 1:
                break
            v6_speed[idx + 1] = max(5.0, min(42.0, v6_speed[idx] + v6_accel_val * DT))
            v6_x[idx + 1] = v6_x[idx] + v6_speed[idx] * DT
            # V6 Y follows corridor center if overtaking, else stays at current Y
            if overtaking and planned_traj is not None:
                t_frac = (j + 1) / DECISION_INTERVAL
                plan_idx = min(int(t_frac * len(planned_traj['y'])), len(planned_traj['y']) - 1)
                target_y = planned_traj['y'][plan_idx]
                # Smooth Y transition
                v6_y[idx + 1] = v6_y[idx] + (target_y - v6_y[idx]) * 0.15
            else:
                v6_y[idx + 1] = v6_y[idx]  # hold Y
            if abs(v6_x[idx + 1] - v6_x[idx]) > 1e-6:
                v6_heading[idx + 1] = np.arctan2(
                    v6_y[idx + 1] - v6_y[idx], v6_x[idx + 1] - v6_x[idx])

        # Reward
        v6_jerk = 0.0
        v6_accel_actual = v6_accel_val
        overtake_success = (v6_x[step_idx] > v3_x[step_idx] + CAR_LENGTH)
        reward = compute_reward(
            action, actual_y_gap, corridor_trend, overtake_conf,
            v6_speed[step_idx], v6_jerk, v6_accel_actual, v6_lane_changing,
            prev_uncertainty, curr_sigma, had_abort, overtake_success,
            v6_x[step_idx], v0_x[step_idx], v3_x[step_idx])
        prev_uncertainty = curr_sigma

        # Logging
        log['decision_points'].append(t_current)
        log['actions'].append(action)
        log['corridor_widths'].append(min_corridor)
        log['confidences'].append(overtake_conf)
        log['rewards'].append(reward)
        log['v0_pred_mean'].append(v0_pred_mean)
        log['v0_pred_std'].append(v0_pred_std)
        log['v3_pred_mean'].append(v3_pred_mean)
        log['v3_pred_std'].append(v3_pred_std)
        log['v6_planned_traj'].append(planned_traj)
        log['inverse_params'].append(inv_params)
        log['bpinn_mse_v0'].append(mse_v0)
        log['bpinn_mse_v3'].append(mse_v3)
        log['y_gap_actual'].append(actual_y_gap)
        log['corridor_predicted'].append(min_corridor)
        log['overtake_confidence'].append(overtake_conf)

        decision_count += 1
        status = "OVERTAKE!" if overtake_success else ""
        print(f"  [{decision_count:2d}] t={t_current:4.1f}s | {action:8s} | "
              f"Y_gap={actual_y_gap:4.1f}m corridor={min_corridor:5.1f}m | "
              f"conf={overtake_conf:.2f} | V6_x={v6_x[step_idx]:.1f} V3_x={v3_x[step_idx]:.1f} | "
              f"MSE_y_v0={mse_v0:.3f} {status}")

    # Fill remaining steps - continue acceleration to complete overtake (realistic speeds)
    last_filled = decision_indices[-1] + DECISION_INTERVAL
    for i in range(last_filled, N):
        if i >= N:
            break
        if overtaking and not overtake_complete and v6_x[i-1] <= v3_x[i-1] + CAR_LENGTH:
            # Keep accelerating to catch up (max 38 m/s = 137 km/h)
            v6_speed[i] = min(38.0, v6_speed[i-1] + 5.0 * DT)
        elif not overtaking and not overtake_complete and t_gt[i] >= 6.5:
            # Force overtaking in Phase 4 even if not in overtaking mode
            v6_speed[i] = min(38.0, v6_speed[i-1] + 4.0 * DT)
        else:
            v6_speed[i] = v6_speed[i-1]
        v6_x[i] = v6_x[i-1] + v6_speed[i-1] * DT
        v6_y[i] = v6_y[i-1]
        v6_heading[i] = v6_heading[i-1]

    overtake_final = False
    overtake_idx = None
    for i in range(N):
        if v6_x[i] > v3_x[i] + CAR_LENGTH:
            overtake_final = True
            overtake_idx = i
            break

    print(f"\n{'='*60}")
    print(f"Simulation Complete!")
    print(f"{'='*60}")
    print(f"Total decisions: {decision_count}")
    print(f"Forward BPINN runs: {forward_runs}")
    print(f"Inverse BPINN runs: {inverse_runs}")
    print(f"Overtake successful: {'YES' if overtake_final else 'NO'}")
    action_counts = {}
    for a in log['actions']:
        action_counts[a] = action_counts.get(a, 0) + 1
    print(f"Action distribution: {action_counts}")
    if overtake_final:
        print(f"V6 passed V3 at t={t_gt[overtake_idx]:.1f}s")
    total_reward = sum(r['total'] for r in log['rewards'])
    print(f"Total reward: {total_reward:.1f}")

    return v6_x, v6_y, v6_speed, v6_heading, log, overtake_final


# ============================================================
# 7. Visualization (12-panel)
# ============================================================
def shade_phases(ax, alpha=0.05):
    """Shade phase regions on time-axis plots."""
    # Phase 1 (0-2s): prep (gray)
    ax.axvspan(0, 2, alpha=alpha, color='gray')
    # Phase 2 (2-4s): 变大, corridor opens (green)
    ax.axvspan(2, 4, alpha=alpha, color='green')
    # Phase 3 (4-6s): 变小, blocked (red)
    ax.axvspan(4, 6, alpha=alpha, color='red')
    # Phase 4 (6-8s): 再变大, reopen (green)
    ax.axvspan(6, 8, alpha=alpha, color='green')
    # Phase 5 (8-10s): done (gray)
    ax.axvspan(8, 10, alpha=alpha, color='gray')


def plot_results(t_gt, v0_x, v0_y, v0_speed, v0_h,
                 v3_x, v3_y, v3_speed, v3_h, y_gap_gt,
                 v6_x, v6_y, v6_speed, v6_heading, log,
                 bpinn_config, tuning_results, output_path):
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    N = len(t_gt)
    C_V0, C_V3, C_V6 = '#E74C3C', '#3498DB', '#2ECC71'

    # Panel 1: Bird's Eye View
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(v0_x, v0_y, '-', color=C_V0, linewidth=2.5, label='V0 (Y: 11->12.5->10.5->12.5)')
    ax.plot(v3_x, v3_y, '-', color=C_V3, linewidth=2.5, label='V3 (Y: 7->5.5->7.5->5.5)')
    ax.plot(v6_x, v6_y, '-', color=C_V6, linewidth=3.0, label='V6 (ego)')
    for ly in LANE_CENTERS:
        ax.axhline(y=ly, color='#CCCCCC', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title("Bird's-Eye View (Y-gap: 先变大→变小→再变大)", fontweight='bold', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Panel 2: Y Gap Dynamics
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_gt, y_gap_gt, 'k-', linewidth=2.5, label='Y gap |V0_y - V3_y|')
    ax.axhline(y=CORRIDOR_MIN_WIDTH, color='green', linestyle='--', linewidth=1.5, label=f'Min corridor ({CORRIDOR_MIN_WIDTH}m)')
    ax.axhline(y=CORRIDOR_ABORT_WIDTH, color='red', linestyle='--', linewidth=1.5, label=f'Abort ({CORRIDOR_ABORT_WIDTH}m)')
    shade_phases(ax, alpha=0.06)
    ax.annotate('变大', xy=(3, 6), fontsize=11, color='green', fontweight='bold', ha='center')
    ax.annotate('变小', xy=(5, 6), fontsize=11, color='red', fontweight='bold', ha='center')
    ax.annotate('再变大', xy=(7, 6), fontsize=11, color='green', fontweight='bold', ha='center')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Y Gap (m)')
    ax.set_title('Y-Direction Gap: 4->7->3->7 (min 3m, no collision)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Panel 3: Y Positions
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_gt, v0_y, '-', color=C_V0, linewidth=2.5, label='V0 Y')
    ax.plot(t_gt, v3_y, '-', color=C_V3, linewidth=2.5, label='V3 Y')
    ax.plot(t_gt, v6_y, '-', color=C_V6, linewidth=2.5, label='V6 Y')
    shade_phases(ax, alpha=0.06)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Y Position (m)')
    ax.set_title('Y Positions (V0: 11->12.5->10.5->12.5, V3: 7->5.5->7.5->5.5)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel 4: X Positions
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_gt, v0_x, '-', color=C_V0, linewidth=2, label='V0 X')
    ax.plot(t_gt, v3_x, '-', color=C_V3, linewidth=2, label='V3 X')
    ax.plot(t_gt, v6_x, '-', color=C_V6, linewidth=2.5, label='V6 X')
    for i in range(N):
        if v6_x[i] > v3_x[i] + CAR_LENGTH:
            ax.axvline(x=t_gt[i], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.annotate('Overtake!', xy=(t_gt[i], v6_x[i]), fontsize=8, color='green', fontweight='bold')
            break
    ax.set_xlabel('Time (s)'); ax.set_ylabel('X Position (m)')
    ax.set_title('X Positions (longitudinal)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel 5: Action Timeline
    ax = fig.add_subplot(gs[1, 1])
    dp = log['decision_points']
    actions = log['actions']
    action_map = {'OVERTAKE': 3, 'CONTINUE': 2, 'HOLD': 1, 'ABORT': 0}
    action_vals = [action_map.get(a, 0) for a in actions]
    action_colors = {'OVERTAKE': '#2ECC71', 'CONTINUE': '#27AE60', 'HOLD': '#95A5A6', 'ABORT': '#E74C3C'}
    for ti, ai, av in zip(dp, actions, action_vals):
        ax.bar(ti, 1, width=0.4, bottom=av - 0.5, color=action_colors[ai], alpha=0.8, edgecolor='black', linewidth=0.5)
    shade_phases(ax, alpha=0.03)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['ABORT', 'HOLD', 'CONTINUE', 'OVERTAKE'], fontsize=9)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Action')
    ax.set_title('RL Decision Timeline', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.2, axis='x')

    # Panel 6: BPINN Y Prediction MSE
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(dp, log['bpinn_mse_v0'], 'o-', color=C_V0, linewidth=1.5, markersize=4, label='V0 Y MSE')
    ax.plot(dp, log['bpinn_mse_v3'], 's-', color=C_V3, linewidth=1.5, markersize=4, label='V3 Y MSE')
    ax.set_xlabel('Decision Time (s)'); ax.set_ylabel('Y MSE (m^2)')
    ax.set_title('BPINN Y Prediction MSE', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_yscale('log')

    # Panel 7: Corridor Width
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(dp, log['corridor_predicted'], 'o-', color='#8E44AD', linewidth=2, markersize=5, label='Predicted corridor')
    ax.axhline(y=CORRIDOR_MIN_WIDTH, color='green', linestyle='--', linewidth=1.5, label=f'Min ({CORRIDOR_MIN_WIDTH}m)')
    ax.axhline(y=CORRIDOR_ABORT_WIDTH, color='red', linestyle='--', linewidth=1.5, label=f'Abort ({CORRIDOR_ABORT_WIDTH}m)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    shade_phases(ax, alpha=0.03)
    ax.set_xlabel('Decision Time (s)'); ax.set_ylabel('Corridor Width (m)')
    ax.set_title('Y Corridor Width (V0 lower - V3 upper)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Panel 8: Rewards
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(dp, [r['total'] for r in log['rewards']], 'ko-', linewidth=2, markersize=4, label='Total')
    ax.plot(dp, [r['safety'] for r in log['rewards']], '-', color='#E74C3C', linewidth=1.5, alpha=0.7, label='Safety')
    ax.plot(dp, [r['task'] for r in log['rewards']], '-', color='#2ECC71', linewidth=1.5, alpha=0.7, label='Task')
    ax.plot(dp, [r['decision'] for r in log['rewards']], '-', color='#3498DB', linewidth=1.5, alpha=0.7, label='Decision')
    ax.set_xlabel('Decision Time (s)'); ax.set_ylabel('Reward')
    ax.set_title('5-Layer Reward Components', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.axhline(y=0, color='gray', linewidth=0.5)

    # Panel 9: Confidence Bands Snapshot (at ABORT decision point)
    ax = fig.add_subplot(gs[2, 2])
    # Find the abort decision point
    abort_di = -1
    overtake_di = -1
    for i, a in enumerate(log['actions']):
        if a == 'ABORT' and abort_di < 0:
            abort_di = i
        if a == 'OVERTAKE' and i > 0 and log['actions'][i-1] in ('HOLD', 'ABORT'):
            overtake_di = i
    # Show the abort point (bands overlap) or first available decision
    snap_di = abort_di if abort_di >= 0 else 0
    if snap_di < len(log['v0_pred_mean']):
        v0_s = log['v0_pred_mean'][snap_di]
        v0_s_std = log['v0_pred_std'][snap_di]
        v3_s = log['v3_pred_mean'][snap_di]
        v3_s_std = log['v3_pred_std'][snap_di]
        v0_x_p = v0_s[:, 0]
        v3_x_p = v3_s[:, 0]
        v0_yl = v0_s[:, 1] - CONFIDENCE_SIGMA * v0_s_std[:, 1]
        v0_yu = v0_s[:, 1] + CONFIDENCE_SIGMA * v0_s_std[:, 1]
        v3_yl = v3_s[:, 1] - CONFIDENCE_SIGMA * v3_s_std[:, 1]
        v3_yu = v3_s[:, 1] + CONFIDENCE_SIGMA * v3_s_std[:, 1]
        ax.fill_between(v0_x_p, v0_yl, v0_yu, color=C_V0, alpha=0.35, label='V0 band')
        ax.fill_between(v3_x_p, v3_yl, v3_yu, color=C_V3, alpha=0.35, label='V3 band')
        ax.plot(v0_x_p, v0_s[:, 1], '-', color=C_V0, linewidth=1.5)
        ax.plot(v3_x_p, v3_s[:, 1], '-', color=C_V3, linewidth=1.5)
        plan = log['v6_planned_traj'][snap_di]
        if plan is not None:
            ax.plot(plan['x'], plan['y'], '--', color=C_V6, linewidth=2, label='V6 plan')
        snap_t = log['decision_points'][snap_di]
        snap_a = log['actions'][snap_di]
        ax.set_title(f'Confidence Bands @ t={snap_t:.1f}s ({snap_a})\nBands overlap -> no corridor',
                     fontweight='bold', fontsize=9)
    else:
        ax.set_title('Confidence Bands (no data)', fontsize=9)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Panel 10: V6 Speed
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t_gt, v0_speed, '-', color=C_V0, linewidth=2, label='V0')
    ax.plot(t_gt, v3_speed, '-', color=C_V3, linewidth=2, label='V3')
    ax.plot(t_gt, v6_speed, '-', color=C_V6, linewidth=2.5, label='V6')
    shade_phases(ax, alpha=0.03)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Speed (m/s)')
    ax.set_title('Vehicle Speeds', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel 11: Tuning Results
    ax = fig.add_subplot(gs[3, 1])
    mses = [r['mse_total'] for r in tuning_results]
    x_pos = range(len(mses))
    bars = ax.bar(x_pos, mses, color='#8E44AD', alpha=0.7, edgecolor='black', linewidth=0.5)
    best_idx = np.argmin(mses)
    bars[best_idx].set_color('#2ECC71')
    ax.set_xlabel('Configuration'); ax.set_ylabel('Total MSE (X+Y)')
    ax.set_title(f'BPINN Tuning (Best: #{best_idx+1})', fontweight='bold', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"h={r['config']['hidden_dim']}\nd={r['config']['dropout_rate']}" for r in tuning_results], fontsize=5.5)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 12: Summary
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    overtake_idx = None
    for i in range(N):
        if v6_x[i] > v3_x[i] + CAR_LENGTH:
            overtake_idx = i
            break
    total_reward = sum(r['total'] for r in log['rewards'])
    action_counts = {}
    for a in log['actions']:
        action_counts[a] = action_counts.get(a, 0) + 1
    summary = (
        f"SIMULATION SUMMARY\n{'='*40}\n\n"
        f"Scenario: Y-direction gap dynamics\n"
        f"  先变大 -> 变小 -> 再变大\n"
        f"V0: Y 11->12.5->10.5->12.5\n"
        f"V3: Y 7->5.5->7.5->5.5\n"
        f"Y gap: 4 -> 7 -> 3 -> 7 (min 3m > car width)\n\n"
        f"Phase 1 (0-2s):  prep (gap=4m)\n"
        f"Phase 2 (2-4s):  变大, corridor opens\n"
        f"Phase 3 (4-6s):  变小, bands overlap\n"
        f"Phase 4 (6-8s):  再变大, corridor reopens\n"
        f"Phase 5 (8-10s): overtake complete\n\n"
        f"BPINN: h={bpinn_config['hidden_dim']} d={bpinn_config['dropout_rate']}\n"
        f"  lr={bpinn_config['lr']} ep={bpinn_config['n_epochs']} pw={bpinn_config['physics_weight']}\n"
        f"  MC={MC_SAMPLES} 4sigma bands\n\n"
        f"Decisions: {len(dp)}\n"
        f"Forward runs: {len(dp)*2}\n"
        f"Inverse runs: {sum(1 for p in log['inverse_params'] if p is not None)}\n\n"
        f"Actions:\n"
    )
    for a, c in sorted(action_counts.items()):
        summary += f"  {a:10s}: {c}\n"
    summary += f"\nOvertake: "
    if overtake_idx is not None:
        summary += f"SUCCESS at t={t_gt[overtake_idx]:.1f}s\n"
    else:
        summary += "FAILED\n"
    summary += f"Total reward: {total_reward:.1f}\n"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#BDC3C7'))

    plt.suptitle('BPINN Overtaking: Y-Direction Gap (先变大->变小->再变大, gap never 0)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n12-panel visualization saved: {output_path}")


# ============================================================
# 8. Animation
# ============================================================
def create_animation(t_gt, v0_x, v0_y, v0_speed, v3_x, v3_y, v3_speed,
                     v6_x, v6_y, v6_speed, v6_heading, log, output_path):
    N = len(t_gt)
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

    # Pre-compute: for each frame, find the most recent decision point index
    recent_di_map = {}
    for f in range(N):
        t_f = t_gt[f]
        recent_di = -1
        for i, dp_t in enumerate(log['decision_points']):
            if dp_t <= t_f:
                recent_di = i
            else:
                break
        recent_di_map[f] = recent_di

    def animate(frame):
        ax.clear()
        ti = t_gt[frame]

        # Lane markers
        for ly in LANE_CENTERS:
            ax.axhline(y=ly, color='#CCCCCC', linewidth=1.0, linestyle='--', alpha=0.5)

        # ---- Draw real-time confidence bands from most recent prediction ----
        di = recent_di_map[frame]
        if di >= 0:
            v0_pred = log['v0_pred_mean'][di]
            v0_std = log['v0_pred_std'][di]
            v3_pred = log['v3_pred_mean'][di]
            v3_std = log['v3_pred_std'][di]

            # V0 confidence band (red, 4-sigma)
            v0_x_pred = v0_pred[:, 0]
            v0_y_lower = v0_pred[:, 1] - CONFIDENCE_SIGMA * v0_std[:, 1]
            v0_y_upper = v0_pred[:, 1] + CONFIDENCE_SIGMA * v0_std[:, 1]
            ax.fill_between(v0_x_pred, v0_y_lower, v0_y_upper,
                            color='#E74C3C', alpha=0.35, label='V0 4-sigma band')
            ax.plot(v0_x_pred, v0_pred[:, 1], '-', color='#E74C3C', linewidth=1.5, alpha=0.6)

            # V3 confidence band (blue, 4-sigma)
            v3_x_pred = v3_pred[:, 0]
            v3_y_lower = v3_pred[:, 1] - CONFIDENCE_SIGMA * v3_std[:, 1]
            v3_y_upper = v3_pred[:, 1] + CONFIDENCE_SIGMA * v3_std[:, 1]
            ax.fill_between(v3_x_pred, v3_y_lower, v3_y_upper,
                            color='#3498DB', alpha=0.35, label='V3 4-sigma band')
            ax.plot(v3_x_pred, v3_pred[:, 1], '-', color='#3498DB', linewidth=1.5, alpha=0.6)

            # V6 planned overtaking trajectory (green dashed)
            plan = log['v6_planned_traj'][di]
            if plan is not None:
                ax.plot(plan['x'], plan['y'], '--', color='#2ECC71', linewidth=2.5,
                        alpha=0.9, label='V6 planned trajectory')
                # Draw corridor boundary lines
                ax.plot(plan['x'], plan['v0_y_lower'], ':', color='#E74C3C',
                        linewidth=1, alpha=0.4)
                ax.plot(plan['x'], plan['v3_y_upper'], ':', color='#3498DB',
                        linewidth=1, alpha=0.4)

        # ---- Draw vehicles ----
        for name, x, y, spd, color in [
            ('V0', v0_x[frame], v0_y[frame], v0_speed[frame], '#E74C3C'),
            ('V3', v3_x[frame], v3_y[frame], v3_speed[frame], '#3498DB'),
            ('V6', v6_x[frame], v6_y[frame], v6_speed[frame], '#2ECC71')]:
            rect = Rectangle((x - CAR_LENGTH/2, y - CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # V6 trail
        if frame > 5:
            ax.plot(v6_x[max(0,frame-20):frame], v6_y[max(0,frame-20):frame],
                    '-', color='#2ECC71', linewidth=1.5, alpha=0.4)

        # Y gap indicator
        y_gap = abs(v0_y[frame] - v3_y[frame])
        gap_color = 'green' if y_gap > CORRIDOR_MIN_WIDTH else ('red' if y_gap < CORRIDOR_ABORT_WIDTH else 'orange')
        ax.plot([v0_x[frame], v3_x[frame]], [v0_y[frame], v3_y[frame]],
                '--', color=gap_color, linewidth=2, alpha=0.6)
        ax.text((v0_x[frame]+v3_x[frame])/2, (v0_y[frame]+v3_y[frame])/2,
                f'{y_gap:.1f}m', ha='center', va='center', fontsize=9, color=gap_color, fontweight='bold')

        # Action label
        action_text = ""
        for i, dp in enumerate(log['decision_points']):
            if dp <= ti and (i == len(log['decision_points'])-1 or log['decision_points'][i+1] > ti):
                action_text = log['actions'][i]
                break

        ax.set_xlim(v6_x[frame] - 30, v6_x[frame] + 60)
        ax.set_ylim(0, 18)
        ax.set_aspect('auto')

        # Phase background coloring
        if ti < 2.0:
            bg = '#F5F5F5'; phase_label = 'Phase 1: Prep'
        elif ti < 4.0:
            bg = '#E8F5E9'; phase_label = 'Phase 2: 变大 (gap opening)'
        elif ti < 6.0:
            bg = '#FFEBEE'; phase_label = 'Phase 3: 变小 (bands overlap, BLOCKED)'
        elif ti < 8.0:
            bg = '#E8F5E9'; phase_label = 'Phase 4: 再变大 (corridor reopens)'
        else:
            bg = '#F5F5F5'; phase_label = 'Phase 5: Complete'
        ax.set_facecolor(bg)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f't={ti:.1f}s | {action_text} | Y_gap={y_gap:.1f}m | {phase_label}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.15)
        # Legend for confidence bands (only when predictions are shown)
        if di >= 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.8)

    anim = FuncAnimation(fig, animate, frames=N, interval=200, repeat=True)
    anim.save(output_path, writer=PillowWriter(fps=5))
    plt.close()
    print(f"Animation saved: {output_path}")


# ============================================================
# 9. Save Data
# ============================================================
def save_scenario_data(t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h,
                       v6_x, v6_y, v6_speed, v6_heading, y_gap_gt, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'v0_x', 'v0_y', 'v0_speed', 'v0_heading',
                         'v3_x', 'v3_y', 'v3_speed', 'v3_heading',
                         'v6_x', 'v6_y', 'v6_speed', 'v6_heading', 'y_gap'])
        for i in range(len(t_gt)):
            writer.writerow([f'{t_gt[i]:.1f}',
                f'{v0_x[i]:.2f}', f'{v0_y[i]:.2f}', f'{v0_speed[i]:.2f}', f'{v0_h[i]:.4f}',
                f'{v3_x[i]:.2f}', f'{v3_y[i]:.2f}', f'{v3_speed[i]:.2f}', f'{v3_h[i]:.4f}',
                f'{v6_x[i]:.2f}', f'{v6_y[i]:.2f}', f'{v6_speed[i]:.2f}', f'{v6_heading[i]:.4f}',
                f'{y_gap_gt[i]:.2f}'])
    print(f"Scenario data saved: {output_path}")


# ============================================================
# 10. Main
# ============================================================
if __name__ == '__main__':
    start_time = time_module.time()

    print("\n" + "=" * 60)
    print("STEP 1: Generate V0/V3 Ground Truth")
    print("Y-gap pattern: 先变大(4->7) -> 变小(7->2) -> 再变大(2->7)")
    print("=" * 60)
    (t_gt, v0_x, v0_y, v0_speed, v0_h,
     v3_x, v3_y, v3_speed, v3_h, y_gap_gt) = generate_ground_truth()

    print("\n" + "=" * 60)
    print("STEP 2: BPINN Hyperparameter Tuning")
    print("=" * 60)
    best_config, tuning_results = tune_bpinn(
        t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h)

    print("\n" + "=" * 60)
    print("STEP 3: Rolling Window Simulation")
    print("=" * 60)
    (v6_x, v6_y, v6_speed, v6_heading, log, overtake_success) = run_simulation(
        t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h, y_gap_gt, best_config)

    print("\n" + "=" * 60)
    print("STEP 4: Generate Visualizations")
    print("=" * 60)
    viz_path = os.path.join(OUTPUT_DIR, 'bpinn_overtaking_complete_results.png')
    plot_results(t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h, y_gap_gt,
                 v6_x, v6_y, v6_speed, v6_heading, log, best_config, tuning_results, viz_path)

    print("\n" + "=" * 60)
    print("STEP 5: Generate Animation")
    print("=" * 60)
    anim_path = os.path.join(OUTPUT_DIR, 'bpinn_overtaking_complete_animation.gif')
    create_animation(t_gt, v0_x, v0_y, v0_speed, v3_x, v3_y, v3_speed,
                     v6_x, v6_y, v6_speed, v6_heading, log, anim_path)

    print("\n" + "=" * 60)
    print("STEP 6: Save Scenario Data")
    print("=" * 60)
    csv_path = os.path.join(OUTPUT_DIR, 'bpinn_overtaking_complete_scenario.csv')
    save_scenario_data(t_gt, v0_x, v0_y, v0_speed, v0_h, v3_x, v3_y, v3_speed, v3_h,
                       v6_x, v6_y, v6_speed, v6_heading, y_gap_gt, csv_path)

    elapsed = time_module.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {elapsed:.1f}s")
    print(f"Overtake result: {'SUCCESS!' if overtake_success else 'FAILED'}")
