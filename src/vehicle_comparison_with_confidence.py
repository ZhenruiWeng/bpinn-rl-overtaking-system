"""
Three-Vehicle Comparison with BPINN Confidence Bands
=====================================================
Trains a BPINN for each vehicle (V0, V3, V6) and plots:
  X, Y, Velocity, Heading with mean ± 2σ confidence bands
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  BPINN + Four-Panel Comparison with Confidence Bands")
print("=" * 70)

# ============================================================
# Config
# ============================================================
DATA_PATH = 'data/overtaking_scene_3vehicles.csv'
OUTPUT_DIR = 'results'

RNG_KEY = random.PRNGKey(42)
N_EPOCHS = 8000
N_MC_SAMPLES = 100
CONF_SIGMA = 4.0

# Fixed hyperparameters (reasonable defaults)
HIDDEN_DIMS = (128, 128, 128)
DROPOUT_RATE = 0.15
LR = 5e-4
PHYSICS_WEIGHT = 1.0

# ============================================================
# Data Loading
# ============================================================
df = pd.read_csv(DATA_PATH)
print("\n[1/4] Data loaded: %d rows, %d vehicles" % (len(df), df['vehicle_id'].nunique()))

NORM = {}
for vid in [0, 3, 6]:
    vdf = df[df['vehicle_id'] == vid]
    NORM[vid] = {
        't_mean': vdf['time'].mean(), 't_std': vdf['time'].std(),
        'x_mean': vdf['x'].mean(), 'x_std': vdf['x'].std(),
        'y_mean': vdf['y'].mean(), 'y_std': vdf['y'].std(),
        'v_mean': vdf['velocity'].mean(), 'v_std': vdf['velocity'].std(),
        'h_mean': vdf['heading'].mean(), 'h_std': vdf['heading'].std(),
    }

# ============================================================
# BPINN Model
# ============================================================
class BPINN(nn.Module):
    hidden_dims: tuple
    dropout_rate: float

    @nn.compact
    def __call__(self, t, training=True):
        x = t
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.tanh(x)
        x = nn.Dense(4)(x)
        return x

def make_apply_fn(model):
    def apply_fn(params_dict, t, **kwargs):
        return model.apply({'params': params_dict}, t, **kwargs)
    return apply_fn

# ============================================================
# Loss Functions
# ============================================================
def physics_loss_fn(params, model, t_batch, apply_fn):
    def loss_per_point(t_single):
        out = apply_fn(params, t_single, training=False)
        x_pred, y_pred, v_pred, h_pred = out[0], out[1], out[2], out[3]
        def state_fn(t):
            o = apply_fn(params, t, training=False)
            return o[0], o[1]
        dx_dt = grad(lambda t: state_fn(t)[0])(t_single)
        dy_dt = grad(lambda t: state_fn(t)[1])(t_single)
        phys_x = dx_dt - v_pred * jnp.cos(h_pred)
        phys_y = dy_dt - v_pred * jnp.sin(h_pred)
        return phys_x**2 + phys_y**2
    return jnp.mean(vmap(loss_per_point)(t_batch))

def data_loss_fn(params, model, t_batch, x_batch, y_batch, v_batch, h_batch, apply_fn, rng):
    def loss_per_point(t, x_true, y_true, v_true, h_true):
        out = apply_fn(params, t, training=True, rngs={'dropout': rng})
        x_pred, y_pred, v_pred, h_pred = out[0], out[1], out[2], out[3]
        return (x_pred - x_true)**2 + (y_pred - y_true)**2 + \
               (v_pred - v_true)**2 + (h_pred - h_true)**2
    return jnp.mean(vmap(loss_per_point)(t_batch, x_batch, y_batch, v_batch, h_batch))

def total_loss(params, model, t_b, x_b, y_b, v_b, h_b, apply_fn, rng, pw):
    d_loss = data_loss_fn(params, model, t_b, x_b, y_b, v_b, h_b, apply_fn, rng)
    p_loss = physics_loss_fn(params, model, t_b, apply_fn)
    return d_loss + pw * p_loss, {'data_loss': d_loss, 'physics_loss': p_loss}

# ============================================================
# Training Function
# ============================================================
def train_bpinn(vehicle_id, t_train, x_train, y_train, v_train, h_train, key):
    norm = NORM[vehicle_id]
    t_n = (t_train - norm['t_mean']) / norm['t_std']
    x_n = (x_train - norm['x_mean']) / norm['x_std']
    y_n = (y_train - norm['y_mean']) / norm['y_std']
    v_n = (v_train - norm['v_mean']) / norm['v_std']
    h_n = (h_train - norm['h_mean']) / norm['h_std']

    t_n = t_n[:, None]; x_n = x_n[:, None]; y_n = y_n[:, None]
    v_n = v_n[:, None]; h_n = h_n[:, None]

    model = BPINN(hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE)
    rng, init_rng = random.split(key)
    variables = model.init(init_rng, t_n[:1], training=True)
    params = variables['params']
    model_apply = make_apply_fn(model)

    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, t_b, x_b, y_b, v_b, h_b, rng):
        rng, step_rng = random.split(rng)
        loss_val, aux = total_loss(params, model, t_b, x_b, y_b, v_b, h_b,
                                    model_apply, step_rng, PHYSICS_WEIGHT)
        grads = jax.grad(lambda p: total_loss(p, model, t_b, x_b, y_b, v_b, h_b,
                                               model_apply, step_rng, PHYSICS_WEIGHT)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    for epoch in range(N_EPOCHS):
        rng, step_rng = random.split(rng)
        params, opt_state, loss_val, aux = train_step(
            params, opt_state, t_n, x_n, y_n, v_n, h_n, step_rng)

    return params, model, float(loss_val)

# ============================================================
# Train BPINNs for V0, V3, V6
# ============================================================
print("\n[2/4] Training BPINNs...")
t_all = np.arange(0.0, 10.01, 0.1)

bpinn_models = {}
bpinn_params = {}

for vid in [0, 3, 6]:
    vdf = df[df['vehicle_id'] == vid].sort_values('time').reset_index(drop=True)
    print("  Training V%d ..." % vid, end="")
    rng_key = random.fold_in(RNG_KEY, vid)
    params, model, _ = train_bpinn(
        vid, vdf['time'].values, vdf['x'].values, vdf['y'].values,
        vdf['velocity'].values, vdf['heading'].values, rng_key)
    bpinn_models[vid] = model
    bpinn_params[vid] = params
    print(" done")

# ============================================================
# MC Dropout Prediction
# ============================================================
print("\n[3/4] MC Dropout prediction (N=%d)..." % N_MC_SAMPLES)
predictions = {}

for vid in [0, 3, 6]:
    norm = NORM[vid]
    t_n = (t_all - norm['t_mean']) / norm['t_std']
    t_n = t_n[:, None]
    rng_key = random.fold_in(RNG_KEY, vid + 10)

    all_preds = []
    for i in range(N_MC_SAMPLES):
        rng = random.fold_in(rng_key, i)
        out = bpinn_models[vid].apply({'params': bpinn_params[vid]}, t_n, training=True, rngs={'dropout': rng})
        all_preds.append(out)

    all_preds = jnp.stack(all_preds)
    mean_pred = jnp.mean(all_preds, axis=0)
    std_pred = jnp.std(all_preds, axis=0)

    # Smooth std boundaries with Savitzky-Golay filter
    def smooth_std(s):
        s = np.array(s)
        if len(s) >= 7:
            return savgol_filter(s, window_length=7, polyorder=3)
        return s

    predictions[vid] = {
        'mean': (np.array(mean_pred[:, 0] * norm['x_std'] + norm['x_mean']),
                 np.array(mean_pred[:, 1] * norm['y_std'] + norm['y_mean']),
                 np.array(mean_pred[:, 2] * norm['v_std'] + norm['v_mean']),
                 np.array(mean_pred[:, 3] * norm['h_std'] + norm['h_mean'])),
        'std': (smooth_std(std_pred[:, 0] * norm['x_std']),
                smooth_std(std_pred[:, 1] * norm['y_std']),
                smooth_std(std_pred[:, 2] * norm['v_std']),
                smooth_std(std_pred[:, 3] * norm['h_std'])),
    }
    print("  V%d: y_std=%.3f~%.3f m" % (vid, predictions[vid]['std'][1].min(), predictions[vid]['std'][1].max()))

# ============================================================
# Visualization: 2x2 with confidence bands
# ============================================================
print("\n[4/4] Generating comparison plots with confidence bands...")

COLORS = {
    0: ('#FF4444', '#CC2222', 'V0 predicted'),
    3: ('#3399FF', '#2266CC', 'V3 predicted'),
    6: ('#F4C430', '#D4A017', 'V6'),
}

fig, axes = plt.subplots(2, 2, figsize=(18, 13), facecolor='white')
fig.suptitle('Three-Vehicle Trajectory Comparison with BPINN Confidence Bands (±4σ, 99.99%)',
             fontsize=18, fontweight='bold', color='#222222', y=0.98)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color='#555555')
    ax.set_ylabel(ylabel, fontsize=11, color='#555555')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

# Helper to get vehicle dataframe
vehicles = {}
for vid in [0, 3, 6]:
    vehicles[vid] = df[df['vehicle_id'] == vid].sort_values('time').reset_index(drop=True)

# --- Plot 1: X vs Time with confidence band ---
ax = axes[0, 0]
for vid in [0, 3, 6]:
    vdf = vehicles[vid]
    c, c_dark, label = COLORS[vid]
    # Confidence band
    x_mean = predictions[vid]['mean'][0]
    x_std = predictions[vid]['std'][0]
    ax.fill_between(t_all, x_mean - CONF_SIGMA * x_std, x_mean + CONF_SIGMA * x_std,
                    color=c, alpha=0.35, zorder=2)
    # Mean prediction
    ax.plot(t_all, x_mean, color=c, linewidth=2.2, label=label, zorder=3)
    # Actual data points
    ax.scatter(vdf['time'].iloc[0], vdf['x'].iloc[0], color=c, s=40, marker='o', zorder=4, edgecolors='white', linewidths=0.5)
    ax.scatter(vdf['time'].iloc[-1], vdf['x'].iloc[-1], color=c, s=40, marker='s', zorder=4, edgecolors='white', linewidths=0.5)
style_ax(ax, 'Longitudinal Position X', 'Time (s)', 'X (m)')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')

# --- Plot 2: Y vs Time with confidence band ---
ax = axes[0, 1]
for vid in [0, 3, 6]:
    vdf = vehicles[vid]
    c, c_dark, label = COLORS[vid]
    y_mean = predictions[vid]['mean'][1]
    y_std = predictions[vid]['std'][1]
    ax.fill_between(t_all, y_mean - CONF_SIGMA * y_std, y_mean + CONF_SIGMA * y_std,
                    color=c, alpha=0.35, zorder=2)
    ax.plot(t_all, y_mean, color=c, linewidth=2.2, label=label, zorder=3)
    ax.scatter(vdf['time'].iloc[0], vdf['y'].iloc[0], color=c, s=40, marker='o', zorder=4, edgecolors='white', linewidths=0.5)
    ax.scatter(vdf['time'].iloc[-1], vdf['y'].iloc[-1], color=c, s=40, marker='s', zorder=4, edgecolors='white', linewidths=0.5)
style_ax(ax, 'Lateral Position Y', 'Time (s)', 'Y (m)')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')

# --- Plot 3: Velocity vs Time with confidence band ---
ax = axes[1, 0]
for vid in [0, 3, 6]:
    vdf = vehicles[vid]
    c, c_dark, label = COLORS[vid]
    v_mean = predictions[vid]['mean'][2]
    v_std = predictions[vid]['std'][2]
    ax.fill_between(t_all, v_mean - CONF_SIGMA * v_std, v_mean + CONF_SIGMA * v_std,
                    color=c, alpha=0.35, zorder=2)
    ax.plot(t_all, v_mean, color=c, linewidth=2.2, label=label, zorder=3)
    ax.scatter(vdf['time'].iloc[0], vdf['velocity'].iloc[0], color=c, s=40, marker='o', zorder=4, edgecolors='white', linewidths=0.5)
    ax.scatter(vdf['time'].iloc[-1], vdf['velocity'].iloc[-1], color=c, s=40, marker='s', zorder=4, edgecolors='white', linewidths=0.5)
style_ax(ax, 'Velocity', 'Time (s)', 'Velocity (m/s)')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')

# --- Plot 4: Heading vs Time with confidence band ---
ax = axes[1, 1]
for vid in [0, 3, 6]:
    vdf = vehicles[vid]
    c, c_dark, label = COLORS[vid]
    h_mean = predictions[vid]['mean'][3]
    h_std = predictions[vid]['std'][3]
    ax.fill_between(t_all, h_mean - CONF_SIGMA * h_std, h_mean + CONF_SIGMA * h_std,
                    color=c, alpha=0.35, zorder=2)
    ax.plot(t_all, h_mean, color=c, linewidth=2.2, label=label, zorder=3)
    ax.scatter(vdf['time'].iloc[0], vdf['heading'].iloc[0], color=c, s=40, marker='o', zorder=4, edgecolors='white', linewidths=0.5)
    ax.scatter(vdf['time'].iloc[-1], vdf['heading'].iloc[-1], color=c, s=40, marker='s', zorder=4, edgecolors='white', linewidths=0.5)
style_ax(ax, 'Heading Angle', 'Time (s)', 'Heading (rad)')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/vehicle_comparison_with_confidence.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nSaved: vehicle_comparison_with_confidence.png')
plt.show()
