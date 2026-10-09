"""
BPINN Overtaking System v2 - Enhanced Implementation
======================================================
Improvements:
  1. Safe overtaking trajectory strictly between confidence bands
  2. Optuna hyperparameter tuning for all BPINNs
  3. Parameter inversion distribution visualization

Framework: JAX + Flax + Optax + Optuna
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator, UnivariateSpline
import optuna
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("  BPINN OVERTAKING SYSTEM v2")
print("=" * 70)

# ============================================================
# 0. Configuration
# ============================================================
DATA_PATH = 'data/overtaking_scene_3vehicles.csv'
OUTPUT_DIR = '.'

RNG_KEY = random.PRNGKey(42)
N_EPOCHS_TRAIN = 8000      # full training epochs
N_EPOCHS_TUNING = 2000     # tuning epochs (faster)
N_MC_SAMPLES = 100
CONF_SIGMA = 2.0
MARGIN = 0.5               # safety margin in meters

# Optuna tuning config
N_OPTUNA_TRIALS = 15

# ============================================================
# 1. Data Loading
# ============================================================
df = pd.read_csv(DATA_PATH)
print("\n[1/6] Data loaded: %d rows, %d vehicles" % (len(df), df['vehicle_id'].nunique()))

# Normalization parameters
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
# 2. BPINN Model Definition
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
# 3. Loss Functions
# ============================================================
def physics_loss_fn(params, model, t_batch, apply_fn):
    """Physics: dx/dt = v*cos(theta), dy/dt = v*sin(theta)"""
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

def data_loss_xy_fn(params, model, t_batch, x_batch, y_batch, apply_fn, rng):
    def loss_per_point(t, x_true, y_true):
        out = apply_fn(params, t, training=True, rngs={'dropout': rng})
        x_pred, y_pred = out[0], out[1]
        return (x_pred - x_true)**2 + (y_pred - y_true)**2
    return jnp.mean(vmap(loss_per_point)(t_batch, x_batch, y_batch))

def total_loss(params, model, t_b, x_b, y_b, v_b, h_b, apply_fn, rng, pw):
    d_loss = data_loss_fn(params, model, t_b, x_b, y_b, v_b, h_b, apply_fn, rng)
    p_loss = physics_loss_fn(params, model, t_b, apply_fn)
    return d_loss + pw * p_loss, {'data_loss': d_loss, 'physics_loss': p_loss}

def total_loss_xy(params, model, t_b, x_b, y_b, apply_fn, rng, pw):
    d_loss = data_loss_xy_fn(params, model, t_b, x_b, y_b, apply_fn, rng)
    p_loss = physics_loss_fn(params, model, t_b, apply_fn)
    return d_loss + pw * p_loss, {'data_loss': d_loss, 'physics_loss': p_loss}

# ============================================================
# 4. Training Function
# ============================================================
def train_bpinn(vehicle_id, t_train, x_train, y_train, v_train, h_train, key,
                hidden_dims, dropout_rate, lr, physics_weight, n_epochs):
    """Train a BPINN with given hyperparameters."""
    norm = NORM[vehicle_id]
    t_n = (t_train - norm['t_mean']) / norm['t_std']
    x_n = (x_train - norm['x_mean']) / norm['x_std']
    y_n = (y_train - norm['y_mean']) / norm['y_std']
    v_n = (v_train - norm['v_mean']) / norm['v_std']
    h_n = (h_train - norm['h_mean']) / norm['h_std']

    t_n = t_n[:, None]; x_n = x_n[:, None]; y_n = y_n[:, None]
    v_n = v_n[:, None]; h_n = h_n[:, None]

    model = BPINN(hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    rng, init_rng = random.split(key)
    variables = model.init(init_rng, t_n[:1], training=True)
    params = variables['params']
    model_apply = make_apply_fn(model)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, t_b, x_b, y_b, v_b, h_b, rng):
        rng, step_rng = random.split(rng)
        loss_val, aux = total_loss(params, model, t_b, x_b, y_b, v_b, h_b,
                                    model_apply, step_rng, physics_weight)
        grads = jax.grad(lambda p: total_loss(p, model, t_b, x_b, y_b, v_b, h_b,
                                               model_apply, step_rng, physics_weight)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    for epoch in range(n_epochs):
        rng, step_rng = random.split(rng)
        params, opt_state, loss_val, aux = train_step(
            params, opt_state, t_n, x_n, y_n, v_n, h_n, step_rng)

    return params, model, float(loss_val)

# ============================================================
# 5. Optuna Hyperparameter Tuning
# ============================================================
print("\n" + "=" * 70)
print("  STAGE 0: Optuna Hyperparameter Tuning")
print("=" * 70)

def create_optuna_objective(vehicle_id, t_data, x_data, y_data, v_data, h_data, key):
    """Create Optuna objective for a specific vehicle."""
    def objective(trial):
        hidden_dims_choice = trial.suggest_categorical('hidden_dims', [
            (32, 32, 32), (64, 64, 64), (128, 128, 128), (64, 128, 64)
        ])
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        physics_weight = trial.suggest_float('physics_weight', 0.1, 5.0, log=True)

        rng_key = random.fold_in(key, trial.number)
        _, _, final_loss = train_bpinn(
            vehicle_id, t_data, x_data, y_data, v_data, h_data, rng_key,
            hidden_dims_choice, dropout_rate, lr, physics_weight, N_EPOCHS_TUNING)
        return final_loss
    return objective

# Tuning for V0
v0_df = df[df['vehicle_id'] == 0].sort_values('time').reset_index(drop=True)
v3_df = df[df['vehicle_id'] == 3].sort_values('time').reset_index(drop=True)
v6_df = df[df['vehicle_id'] == 6].sort_values('time').reset_index(drop=True)

print("  Tuning BPINN for V0...")
study_v0 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_v0.optimize(
    create_optuna_objective(0, v0_df['time'].values, v0_df['x'].values, v0_df['y'].values,
                            v0_df['velocity'].values, v0_df['heading'].values, RNG_KEY),
    n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

best_v0 = study_v0.best_params
print("    Best V0: hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f, loss=%.4f" % (
    best_v0['hidden_dims'], best_v0['dropout_rate'], best_v0['lr'],
    best_v0['physics_weight'], study_v0.best_value))

print("  Tuning BPINN for V3...")
study_v3 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=43))
study_v3.optimize(
    create_optuna_objective(3, v3_df['time'].values, v3_df['x'].values, v3_df['y'].values,
                            v3_df['velocity'].values, v3_df['heading'].values, RNG_KEY),
    n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

best_v3 = study_v3.best_params
print("    Best V3: hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f, loss=%.4f" % (
    best_v3['hidden_dims'], best_v3['dropout_rate'], best_v3['lr'],
    best_v3['physics_weight'], study_v3.best_value))

print("  Tuning BPINN for inverse problem...")
study_inv = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=44))

# For inverse tuning, we use V6's actual x,y as target
def objective_inv(trial):
    hidden_dims_choice = trial.suggest_categorical('hidden_dims', [
        (32, 32, 32), (64, 64, 64), (128, 128, 128), (64, 128, 64)
    ])
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    physics_weight = trial.suggest_float('physics_weight', 0.1, 10.0, log=True)

    # Use V6 actual trajectory for tuning
    norm6 = NORM[6]
    t_n = (v6_df['time'].values - norm6['t_mean']) / norm6['t_std']
    x_n = (v6_df['x'].values - norm6['x_mean']) / norm6['x_std']
    y_n = (v6_df['y'].values - norm6['y_mean']) / norm6['y_std']
    t_n = t_n[:, None]; x_n = x_n[:, None]; y_n = y_n[:, None]

    model = BPINN(hidden_dims=hidden_dims_choice, dropout_rate=dropout_rate)
    rng_key = random.fold_in(RNG_KEY, trial.number + 100)
    rng_init, rng_key = random.split(rng_key)
    variables = model.init(rng_init, t_n[:1], training=True)
    params = variables['params']
    model_apply = make_apply_fn(model)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jit
    def train_step_inv_t(params, opt_state, t_b, x_b, y_b, rng):
        rng, step_rng = random.split(rng)
        loss_val, aux = total_loss_xy(params, model, t_b, x_b, y_b, model_apply, step_rng, physics_weight)
        grads = jax.grad(lambda p: total_loss_xy(p, model, t_b, x_b, y_b, model_apply, step_rng, physics_weight)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    for epoch in range(N_EPOCHS_TUNING):
        rng_key, step_rng = random.split(rng_key)
        params, opt_state, loss_val, aux = train_step_inv_t(
            params, opt_state, t_n, x_n, y_n, step_rng)

    return float(loss_val)

study_inv.optimize(objective_inv, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_inv = study_inv.best_params
print("    Best Inv: hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f, loss=%.4f" % (
    best_inv['hidden_dims'], best_inv['dropout_rate'], best_inv['lr'],
    best_inv['physics_weight'], study_inv.best_value))

# ============================================================
# 6. Stage 1: Train BPINN with Best Hyperparameters
# ============================================================
print("\n" + "=" * 70)
print("  STAGE 1: Forward BPINN Prediction (Tuned)")
print("=" * 70)

t_all = np.arange(0.0, 10.01, 0.1)

bpinn_models = {}
bpinn_params = {}

for vid, best_params, vdf in [(0, best_v0, v0_df), (3, best_v3, v3_df)]:
    print("\n  Training V%d with tuned hyperparameters..." % vid)
    rng_key = random.fold_in(RNG_KEY, vid)
    params, model, _ = train_bpinn(
        vid, vdf['time'].values, vdf['x'].values, vdf['y'].values,
        vdf['velocity'].values, vdf['heading'].values, rng_key,
        best_params['hidden_dims'], best_params['dropout_rate'],
        best_params['lr'], best_params['physics_weight'], N_EPOCHS_TRAIN)
    bpinn_models[vid] = model
    bpinn_params[vid] = params

# MC Dropout prediction
print("\n  Running MC Dropout (N=%d)..." % N_MC_SAMPLES)
predictions = {}
for vid in [0, 3]:
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

    predictions[vid] = {
        'mean': (np.array(mean_pred[:, 0] * norm['x_std'] + norm['x_mean']),
                 np.array(mean_pred[:, 1] * norm['y_std'] + norm['y_mean']),
                 np.array(mean_pred[:, 2] * norm['v_std'] + norm['v_mean']),
                 np.array(mean_pred[:, 3] * norm['h_std'] + norm['h_mean'])),
        'std': (np.array(std_pred[:, 0] * norm['x_std']),
                np.array(std_pred[:, 1] * norm['y_std']),
                np.array(std_pred[:, 2] * norm['v_std']),
                np.array(std_pred[:, 3] * norm['h_std'])),
        'all_preds': all_preds,  # keep raw predictions for distribution
    }

print("  V0: y_std=%.3f~%.3f m" % (predictions[0]['std'][1].min(), predictions[0]['std'][1].max()))
print("  V3: y_std=%.3f~%.3f m" % (predictions[3]['std'][1].min(), predictions[3]['std'][1].max()))

# ============================================================
# 7. Stage 2: Safe Overtaking Trajectory (Strictly Between Bands)
# ============================================================
print("\n" + "=" * 70)
print("  STAGE 2: Safe Overtaking Planning")
print("=" * 70)

# Compute safe bounds at each time
gap_analysis = []
for i, t in enumerate(t_all):
    if i >= len(predictions[0]['mean'][0]):
        break

    v0_y_mean = predictions[0]['mean'][1][i]
    v0_y_std = predictions[0]['std'][1][i]
    v0_lower = v0_y_mean - CONF_SIGMA * v0_y_std  # lower bound of V0 confidence band

    v3_y_mean = predictions[3]['mean'][1][i]
    v3_y_std = predictions[3]['std'][1][i]
    v3_upper = v3_y_mean + CONF_SIGMA * v3_y_std  # upper bound of V3 confidence band

    # Safe region: between V3 upper bound and V0 lower bound
    safe_lower = v3_upper + MARGIN
    safe_upper = v0_lower - MARGIN
    gap = safe_upper - safe_lower

    v6_idx = (v6_df['time'] - t).abs().argsort().iloc[0]
    v6_y = v6_df.iloc[v6_idx]['y']
    v6_x = v6_df.iloc[v6_idx]['x']

    gap_analysis.append({
        'time': t,
        'v0_lower': v0_lower, 'v0_upper': v0_y_mean + CONF_SIGMA * v0_y_std,
        'v3_lower': v3_y_mean - CONF_SIGMA * v3_y_std, 'v3_upper': v3_upper,
        'safe_lower': safe_lower, 'safe_upper': safe_upper,
        'gap': gap, 'safe_center': (safe_lower + safe_upper) / 2,
        'v6_x': v6_x, 'v6_y': v6_y,
    })

gap_df = pd.DataFrame(gap_analysis)

# Check where gap is positive
valid_gap = gap_df[gap_df['gap'] > 0]
if len(valid_gap) > 0:
    print("  Safe region: t=[%.1f, %.1f]s, gap=%.2f~%.2f m" % (
        valid_gap['time'].min(), valid_gap['time'].max(),
        valid_gap['gap'].min(), valid_gap['gap'].max()))
else:
    print("  WARNING: No safe region found! Reducing margin...")
    gap_df['safe_lower'] = gap_df['v3_upper']
    gap_df['safe_upper'] = gap_df['v0_lower']
    gap_df['gap'] = gap_df['safe_upper'] - gap_df['safe_lower']
    valid_gap = gap_df[gap_df['gap'] > 0]
    print("  Adjusted: t=[%.1f, %.1f]s" % (valid_gap['time'].min(), valid_gap['time'].max()))

# Design overtaking trajectory: waypoints constrained within safe region
# Pick key time points and compute safe y positions
t_waypoints = [0.0]
t_waypoints.append(valid_gap['time'].min())
t_waypoints.append(valid_gap['time'].mean())
t_waypoints.append(valid_gap['time'].max())
t_waypoints.append(10.0)

y_waypoints = []
for t_wp in t_waypoints:
    row = gap_df.iloc[(gap_df['time'] - t_wp).abs().argsort().iloc[0]]
    # Start and end: follow V6's actual lane
    if t_wp < 1.0:
        y_wp = v6_df.iloc[0]['y']
    elif t_wp > 9.0:
        y_wp = v6_df.iloc[-1]['y']
    else:
        # Through the gap: use safe center
        y_wp = row['safe_center']
        # Clamp to safe region
        y_wp = max(row['safe_lower'], min(row['safe_upper'], y_wp))
    y_waypoints.append(y_wp)

# Ensure monotonic time and remove duplicates
t_waypoints = np.array(t_waypoints)
y_waypoints = np.array(y_waypoints)
sort_idx = np.argsort(t_waypoints)
t_waypoints = t_waypoints[sort_idx]
y_waypoints = y_waypoints[sort_idx]

# Remove duplicate time points
t_unique, uniq_idx = np.unique(t_waypoints, return_index=True)
y_unique = y_waypoints[uniq_idx]
t_waypoints = t_unique
y_waypoints = y_unique

# Monotonic interpolation — no artificial wiggles
cs_y = PchipInterpolator(t_waypoints, y_waypoints)
t_overtaking = np.linspace(0, 10, 101)
y_overtaking = cs_y(t_overtaking)

# Clamp to safe region at all times
for i, t in enumerate(t_overtaking):
    row = gap_df.iloc[(gap_df['time'] - t).abs().argsort().iloc[0]]
    y_overtaking[i] = max(row['safe_lower'], min(row['safe_upper'], y_overtaking[i]))

# X position from V6's actual trajectory
v6_x_interp = interp1d(v6_df['time'], v6_df['x'], kind='linear',
                       bounds_error=False, fill_value='extrapolate')
x_overtaking = v6_x_interp(t_overtaking)

print("  Trajectory designed:")
print("    Waypoints: %d" % len(t_waypoints))
print("    Y range: %.2f ~ %.2f m" % (y_overtaking.min(), y_overtaking.max()))
print("    Safe region: %.2f ~ %.2f m" % (
    gap_df['safe_lower'].min(), gap_df['safe_upper'].max()))

# Verify trajectory is within safe region
violations = 0
for i, t in enumerate(t_overtaking):
    row = gap_df.iloc[(gap_df['time'] - t).abs().argsort().iloc[0]]
    if y_overtaking[i] < row['safe_lower'] - 0.01 or y_overtaking[i] > row['safe_upper'] + 0.01:
        violations += 1
print("  Safety check: %d violations out of %d points" % (violations, len(t_overtaking)))

# ============================================================
# 8. Stage 3: Inverse BPINN with Tuned Hyperparameters
# ============================================================
print("\n" + "=" * 70)
print("  STAGE 3: Inverse BPINN Parameter Estimation (Tuned)")
print("=" * 70)

# Normalize overtaking trajectory
t_inv = t_overtaking
x_inv = x_overtaking
y_inv = y_overtaking

norm_inv = {
    't_mean': t_inv.mean(), 't_std': t_inv.std(),
    'x_mean': x_inv.mean(), 'x_std': x_inv.std(),
    'y_mean': y_inv.mean(), 'y_std': y_inv.std(),
    'v_mean': v6_df['velocity'].mean(), 'v_std': v6_df['velocity'].std(),
    'h_mean': v6_df['heading'].mean(), 'h_std': v6_df['heading'].std(),
}
NORM['inverse'] = norm_inv

t_inv_n = (t_inv - norm_inv['t_mean']) / norm_inv['t_std']
x_inv_n = (x_inv - norm_inv['x_mean']) / norm_inv['x_std']
y_inv_n = (y_inv - norm_inv['y_mean']) / norm_inv['y_std']
t_inv_n = t_inv_n[:, None]; x_inv_n = x_inv_n[:, None]; y_inv_n = y_inv_n[:, None]

model_inv = BPINN(hidden_dims=best_inv['hidden_dims'], dropout_rate=best_inv['dropout_rate'])
rng_inv = random.fold_in(RNG_KEY, 42)
rng_init, rng_inv = random.split(rng_inv)
variables_inv = model_inv.init(rng_init, t_inv_n[:1], training=True)
params_inv = variables_inv['params']

model_apply_inv = make_apply_fn(model_inv)
optimizer_inv = optax.adam(learning_rate=best_inv['lr'])
opt_state_inv = optimizer_inv.init(params_inv)

@jit
def train_step_inv(params, opt_state, t_b, x_b, y_b, rng):
    rng, step_rng = random.split(rng)
    loss_val, aux = total_loss_xy(
        params, model_inv, t_b, x_b, y_b,
        model_apply_inv, step_rng, best_inv['physics_weight'])
    grads = jax.grad(lambda p: total_loss_xy(
        p, model_inv, t_b, x_b, y_b,
        model_apply_inv, step_rng, best_inv['physics_weight'])[0])(params)
    updates, opt_state = optimizer_inv.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, aux

for epoch in range(N_EPOCHS_TRAIN):
    rng_inv, step_rng = random.split(rng_inv)
    params_inv, opt_state_inv, loss_val, aux = train_step_inv(
        params_inv, opt_state_inv, t_inv_n, x_inv_n, y_inv_n, step_rng)

    if epoch % 2000 == 0:
        print("  Epoch %d: total=%.6f, data=%.6f, physics=%.6f" % (
            epoch, float(loss_val), float(aux['data_loss']), float(aux['physics_loss'])))

print("  Final: total=%.6f, data=%.6f, physics=%.6f" % (
    float(loss_val), float(aux['data_loss']), float(aux['physics_loss'])))

# MC Dropout for inverse model - collect all predictions for distribution
print("\n  Running MC Dropout for inverse (N=%d)..." % N_MC_SAMPLES)
rng_inv = random.fold_in(RNG_KEY, 99)
all_inv_preds = []

for i in range(N_MC_SAMPLES):
    rng = random.fold_in(rng_inv, i)
    out = model_inv.apply({'params': params_inv}, t_inv_n, training=True, rngs={'dropout': rng})
    all_inv_preds.append(out)

all_inv_preds = jnp.stack(all_inv_preds)  # [N_samples, N_time, 4]

# Denormalize all predictions for distribution analysis
inv_v_all = np.array(all_inv_preds[:, :, 2] * norm_inv['v_std'] + norm_inv['v_mean'])
inv_h_all = np.array(all_inv_preds[:, :, 3] * norm_inv['h_std'] + norm_inv['h_mean'])
inv_x_all = np.array(all_inv_preds[:, :, 0] * norm_inv['x_std'] + norm_inv['x_mean'])
inv_y_all = np.array(all_inv_preds[:, :, 1] * norm_inv['y_std'] + norm_inv['y_mean'])

# Compute mean and std
inv_mean = jnp.mean(all_inv_preds, axis=0)
inv_std = jnp.std(all_inv_preds, axis=0)

inv_v_mean = np.array(inv_mean[:, 2] * norm_inv['v_std'] + norm_inv['v_mean'])
inv_h_mean = np.array(inv_mean[:, 3] * norm_inv['h_std'] + norm_inv['h_mean'])
inv_v_std = np.array(inv_std[:, 2] * norm_inv['v_std'])
inv_h_std = np.array(inv_std[:, 3] * norm_inv['h_std'])

# Acceleration from velocity
inv_a = np.gradient(inv_v_mean, t_overtaking)
inv_a_std = np.gradient(inv_v_std, t_overtaking)

# Distribution statistics
print("\n  Inverse estimation (distribution across %d samples):" % N_MC_SAMPLES)
print("    Velocity: %.2f ± %.2f m/s (%.1f ± %.1f km/h)" % (
    inv_v_mean.mean(), inv_v_mean.std(),
    inv_v_mean.mean() * 3.6, inv_v_mean.std() * 3.6))
print("    Heading: %.4f ± %.4f rad (%.2f ± %.2f deg)" % (
    inv_h_mean.mean(), inv_h_mean.std(),
    np.degrees(inv_h_mean.mean()), np.degrees(inv_h_mean.std())))
print("    Acceleration: %.3f ± %.3f m/s²" % (abs(inv_a).mean(), np.std(inv_a)))

# ============================================================
# 9. Visualization (Enhanced)
# ============================================================
print("\n" + "=" * 70)
print("  STAGE 4: Visualization")
print("=" * 70)

# Modern style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e8e8e8',
    'grid.linewidth': 0.6,
    'axes.labelcolor': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
})

# Color palette
C_V0 = '#E74C3C'   # red
C_V3 = '#3498DB'   # blue
C_V6 = '#2ECC71'   # green
C_PLAN = '#F39C12' # orange
C_SAFE = '#27AE60' # green for safe region
C_INV_V = '#9B59B6' # purple
C_INV_H = '#E67E22'  # orange
C_INV_A = '#1ABC9C'  # teal

fig = plt.figure(figsize=(38, 16), facecolor='white')
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32,
                       left=0.05, right=0.97, top=0.95, bottom=0.05)

# Helper: style an axis
def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50', pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color='#555555')
    ax.set_ylabel(ylabel, fontsize=9, color='#555555')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)

# Compute Y-axis limits — squeeze to band edges with zero padding for minimal gap
v0_lower_band = predictions[0]['mean'][1] - CONF_SIGMA * predictions[0]['std'][1]
v3_upper_band = predictions[3]['mean'][1] + CONF_SIGMA * predictions[3]['std'][1]
y_min = v3_upper_band.min() - 0.02  # virtually zero padding
y_max = v0_lower_band.max() + 0.02

# Ensure V6 planned trajectory is visible
y_min = min(y_min, y_overtaking.min() - 0.05)
y_max = max(y_max, y_overtaking.max() + 0.05)
print("  Y-axis limits: %.2f ~ %.2f m (maximally compressed)" % (y_min, y_max))

# Lane boundary estimation from vehicle data
lane2_center = v3_df['y'].mean()   # V3 travels in lane 2
lane3_center = v0_df['y'].mean()   # V0 travels in lane 3
lane_boundary_y = (lane2_center + lane3_center) / 2

# ============================================================
# 3D Lane Surface Helper — draws extruded lane surfaces (z is purely visual)
# ============================================================
LANE_Z_BASE = 0.0       # road surface height
LANE_Z_RAISE = 0.8      # how much trajectories float above the road
LANE_Z_DEPTH = -0.3     # lane surface thickness for 3D extrusion

def draw_3d_lane_surface(ax, x_min, x_max, y_min, y_max, y_boundary, color_left, color_right):
    """Draw two lane surfaces with three prominent boundary lines (z is purely visual)."""
    # Lane 2 (lower y) surface
    verts_l2 = [(x_min, y_min, LANE_Z_DEPTH), (x_max, y_min, LANE_Z_DEPTH),
                (x_max, y_boundary, LANE_Z_DEPTH), (x_min, y_boundary, LANE_Z_DEPTH)]
    poly_l2 = Poly3DCollection([verts_l2], alpha=0.15, facecolor=color_left, edgecolor='none')
    ax.add_collection3d(poly_l2)

    # Lane 3 (higher y) surface
    verts_l3 = [(x_min, y_boundary, LANE_Z_DEPTH), (x_max, y_boundary, LANE_Z_DEPTH),
                (x_max, y_max, LANE_Z_DEPTH), (x_min, y_max, LANE_Z_DEPTH)]
    poly_l3 = Poly3DCollection([verts_l3], alpha=0.15, facecolor=color_right, edgecolor='none')
    ax.add_collection3d(poly_l3)

    # --- Three prominent lane lines ---
    # Line 1: Left road edge (solid yellow)
    ax.plot([x_min, x_max], [y_min, y_min], [LANE_Z_DEPTH, LANE_Z_DEPTH],
            color='#F1C40F', linestyle='-', linewidth=2.5, alpha=0.8)
    # Line 2: Center lane boundary (dashed white, thicker)
    ax.plot([x_min, x_max], [y_boundary, y_boundary], [LANE_Z_DEPTH, LANE_Z_DEPTH],
            color='#ECF0F1', linestyle='--', linewidth=2.5, alpha=0.9)
    # Line 3: Right road edge (solid yellow)
    ax.plot([x_min, x_max], [y_max, y_max], [LANE_Z_DEPTH, LANE_Z_DEPTH],
            color='#F1C40F', linestyle='-', linewidth=2.5, alpha=0.8)

def fill_3d_band(ax, xs, y_lower, y_upper, z, color, alpha=0.12):
    """Draw a 3D vertical band (confidence band floating above road)."""
    n = len(xs)
    verts = []
    for i in range(n - 1):
        quad = [(xs[i], y_lower[i], z), (xs[i+1], y_lower[i+1], z),
                (xs[i+1], y_upper[i+1], z), (xs[i], y_upper[i], z)]
        verts.append(quad)
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(poly)

# ============================================================
# 3D style helper
# ============================================================
def style_3d_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50', pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color='#555555', labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, color='#555555', labelpad=2)
    ax.tick_params(axis='x', which='major', labelsize=7, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=7, pad=0)
    # Completely clear z-axis — z is purely visual
    ax.set_zticks([])
    ax.set_zlim(-1, 2)
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_tick_params(labelleft=False, labelsize=0)
    ax.zaxis.label.set_visible(False)
    # Make z-axis pane and line fully transparent
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    # Clean x/y panes
    ax.xaxis.pane.set_facecolor('#f5f5f5')
    ax.yaxis.pane.set_facecolor('#f5f5f5')
    ax.xaxis.pane.set_edgecolor('#d8d8d8')
    ax.yaxis.pane.set_edgecolor('#d8d8d8')
    ax.xaxis.pane.set_linewidth(0.5)
    ax.yaxis.pane.set_linewidth(0.5)
    # Light grid only on bottom
    ax.grid(True, alpha=0.2, linewidth=0.3)

# ============================================================
# Plot 1: 2D Bird's-Eye View — Realistic 4-Lane Highway Scene
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('none')  # transparent background
ax1.patch.set_alpha(0)
x1_lo, x1_hi = float(v3_df['x'].min()) - 10, float(v0_df['x'].max()) + 10

# --- 4-lane geometry ---
# Lane 2 center ≈ 9.24 (V3, V6), Lane 3 center ≈ 12.90 (V0)
# Lane width ≈ 3.66m, extend to 4 lanes
lane_width = abs(lane3_center - lane2_center)
lane1_center = lane2_center - lane_width     # Lane 1 (leftmost, no vehicle)
lane4_center = lane3_center + lane_width     # Lane 4 (rightmost, no vehicle)

# Lane boundaries (3 dividers between 4 lanes)
boundary_12 = (lane1_center + lane2_center) / 2   # Lane 1 / Lane 2
boundary_23 = lane_boundary_y                       # Lane 2 / Lane 3 (existing)
boundary_34 = (lane3_center + lane4_center) / 2   # Lane 3 / Lane 4

# Road edges
road_y_min = lane1_center - lane_width / 2
road_y_max = lane4_center + lane_width / 2

# --- Wider confidence sigma for this plot only ---
PLOT1_SIGMA = 5.0

# No road fill - transparent background, only lane divider lines

# --- Helper: draw dashed line ---
def draw_dashed_line(ax, x_lo, x_hi, y, dash=6.0, gap=9.0, lw=2.5, alpha=0.85, color='#888888'):
    xp = x_lo
    while xp < x_hi:
        xe = min(xp + dash, x_hi)
        ax.plot([xp, xe], [y, y], color=color, linestyle='-',
                linewidth=lw, alpha=alpha, zorder=1, solid_capstyle='round')
        xp += dash + gap

# --- Road edge lines (dashed gray) ---
draw_dashed_line(ax1, x1_lo, x1_hi, road_y_min, dash=4.0, gap=4.0, lw=3.0, alpha=0.7, color='#aaaaaa')
draw_dashed_line(ax1, x1_lo, x1_hi, road_y_max, dash=4.0, gap=4.0, lw=3.0, alpha=0.7, color='#aaaaaa')

# --- Lane dividers (dashed) ---
draw_dashed_line(ax1, x1_lo, x1_hi, boundary_12, dash=6.0, gap=9.0, lw=2.5, alpha=0.7, color='#999999')
draw_dashed_line(ax1, x1_lo, x1_hi, boundary_23, dash=6.0, gap=9.0, lw=2.5, alpha=0.7, color='#999999')
draw_dashed_line(ax1, x1_lo, x1_hi, boundary_34, dash=6.0, gap=9.0, lw=2.5, alpha=0.7, color='#999999')

# --- Lane labels ---
for lane_name, lane_y in [('Lane 1', lane1_center),
                          ('Lane 2', lane2_center),
                          ('Lane 3', lane3_center),
                          ('Lane 4', lane4_center)]:
    ax1.text(x1_lo + 12, lane_y, lane_name,
             fontsize=7, color='#555555', fontweight='bold', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='#cccccc'))

# --- Helper: smooth band edges using Savitzky-Golay filter (avoids endpoint oscillation) ---
def _smooth_band(x_raw, y_raw, window=11, poly=2):
    from scipy.signal import savgol_filter
    y = np.asarray(y_raw)
    # Ensure window is odd and not larger than data
    w = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    if w < 3: return y
    w = w if w % 2 == 1 else w - 1
    p = min(poly, w - 2)
    return savgol_filter(y, window_length=w, polyorder=p)

# --- Confidence bands (soft glow with edge outline) ---
v0_x_band = predictions[0]['mean'][0]
v0_band_lower = predictions[0]['mean'][1] - PLOT1_SIGMA * predictions[0]['std'][1]
v0_band_upper = predictions[0]['mean'][1] + PLOT1_SIGMA * predictions[0]['std'][1]
# Smooth band edges for cleaner visual
v0_band_lower_s = _smooth_band(v0_x_band, v0_band_lower)
v0_band_upper_s = _smooth_band(v0_x_band, v0_band_upper)
ax1.fill_between(v0_x_band, v0_band_lower_s, v0_band_upper_s,
                  color=C_V0, alpha=0.28, zorder=2)
ax1.plot(v0_x_band, v0_band_lower_s, color=C_V0, linestyle='-',
         linewidth=1.0, alpha=0.50, zorder=3)
ax1.plot(v0_x_band, v0_band_upper_s, color=C_V0, linestyle='-',
         linewidth=1.0, alpha=0.50, zorder=3)

v3_x_band = predictions[3]['mean'][0]
v3_band_lower = predictions[3]['mean'][1] - PLOT1_SIGMA * predictions[3]['std'][1]
v3_band_upper = predictions[3]['mean'][1] + PLOT1_SIGMA * predictions[3]['std'][1]
# Smooth band edges for cleaner visual
v3_band_lower_s = _smooth_band(v3_x_band, v3_band_lower)
v3_band_upper_s = _smooth_band(v3_x_band, v3_band_upper)
ax1.fill_between(v3_x_band, v3_band_lower_s, v3_band_upper_s,
                  color=C_V3, alpha=0.28, zorder=2)
ax1.plot(v3_x_band, v3_band_lower_s, color=C_V3, linestyle='-',
         linewidth=1.0, alpha=0.50, zorder=3)
ax1.plot(v3_x_band, v3_band_upper_s, color=C_V3, linestyle='-',
         linewidth=1.0, alpha=0.50, zorder=3)

# --- Smooth trajectory curves (smoothing spline) ---

v0_spline = UnivariateSpline(v0_df['x'].values, v0_df['y'].values, s=0.15, k=3)
v0_x_curve = np.linspace(v0_df['x'].min(), v0_df['x'].max(), 200)
v0_y_curve = v0_spline(v0_x_curve)

v3_spline = UnivariateSpline(v3_df['x'].values, v3_df['y'].values, s=0.15, k=3)
v3_x_curve = np.linspace(v3_df['x'].min(), v3_df['x'].max(), 200)
v3_y_curve = v3_spline(v3_x_curve)

# Trajectory lines
ax1.plot(v0_x_curve, v0_y_curve, color=C_V0, linewidth=2.5, label='V0 predicted', zorder=4)
ax1.plot(v3_x_curve, v3_y_curve, color=C_V3, linewidth=2.5, label='V3 predicted', zorder=4)
ax1.plot(x_overtaking, y_overtaking, color=C_PLAN, linewidth=2.5, label='V6 planned', zorder=5)

# --- Realistic car models (body + windshield + roof) ---
CAR_LEN = 4.2
CAR_WID = 1.6

def draw_realistic_car(ax, x, y, heading, color, zorder=6):
    """Draw a realistic car: body, windshield, rear window, roof."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Car body — main rounded rectangle
    body_rect = np.array([
        [-CAR_LEN/2, -CAR_WID/2],
        [ CAR_LEN/2, -CAR_WID/2],
        [ CAR_LEN/2,  CAR_WID/2],
        [-CAR_LEN/2,  CAR_WID/2]])
    rbody = body_rect @ rot.T
    rbody[:, 0] += x; rbody[:, 1] += y
    ax.add_patch(Polygon(rbody, facecolor=color, edgecolor='#1a1a1a',
                         linewidth=1.2, alpha=0.92, zorder=zorder))

    # Windshield (front)
    ws_rect = np.array([
        [CAR_LEN * 0.08, -CAR_WID * 0.32],
        [CAR_LEN * 0.32, -CAR_WID * 0.32],
        [CAR_LEN * 0.32,  CAR_WID * 0.32],
        [CAR_LEN * 0.08,  CAR_WID * 0.32]])
    rws = ws_rect @ rot.T
    rws[:, 0] += x; rws[:, 1] += y
    ax.add_patch(Polygon(rws, facecolor='#B0CCE8', edgecolor='none',
                         alpha=0.45, zorder=zorder + 0.1))

    # Rear window
    rw_rect = np.array([
        [-CAR_LEN * 0.32, -CAR_WID * 0.32],
        [-CAR_LEN * 0.12, -CAR_WID * 0.32],
        [-CAR_LEN * 0.12,  CAR_WID * 0.32],
        [-CAR_LEN * 0.32,  CAR_WID * 0.32]])
    rrw = rw_rect @ rot.T
    rrw[:, 0] += x; rrw[:, 1] += y
    ax.add_patch(Polygon(rrw, facecolor='#B0CCE8', edgecolor='none',
                         alpha=0.35, zorder=zorder + 0.1))

    # Roof (slightly darker than body)
    roof_rect = np.array([
        [-CAR_LEN * 0.12, -CAR_WID * 0.28],
        [CAR_LEN * 0.08,  -CAR_WID * 0.28],
        [CAR_LEN * 0.08,   CAR_WID * 0.28],
        [-CAR_LEN * 0.12,  CAR_WID * 0.28]])
    rroof = roof_rect @ rot.T
    rroof[:, 0] += x; rroof[:, 1] += y
    darker = '#%02x%02x%02x' % (max(0, int(color[1:3], 16) - 30),
                                 max(0, int(color[3:5], 16) - 30),
                                 max(0, int(color[5:7], 16) - 30))
    ax.add_patch(Polygon(rroof, facecolor=darker, edgecolor='none',
                         alpha=0.5, zorder=zorder + 0.05))

# Place cars along trajectories
v0_headings = np.arctan(v0_spline.derivative()(v0_df['x'].values))
for idx in [5, len(v0_df)//3, 2*len(v0_df)//3, len(v0_df)-5]:
    draw_realistic_car(ax1, v0_df['x'].iloc[idx], v0_df['y'].iloc[idx],
                       v0_headings[idx], C_V0, zorder=6)

v3_headings = np.arctan(v3_spline.derivative()(v3_df['x'].values))
for idx in [5, len(v3_df)//3, 2*len(v3_df)//3, len(v3_df)-5]:
    draw_realistic_car(ax1, v3_df['x'].iloc[idx], v3_df['y'].iloc[idx],
                       v3_headings[idx], C_V3, zorder=6)

v6_headings = np.arctan2(np.gradient(y_overtaking), np.gradient(x_overtaking))
for idx in [5, len(x_overtaking)//3, 2*len(x_overtaking)//3, len(x_overtaking)-5]:
    draw_realistic_car(ax1, x_overtaking[idx], y_overtaking[idx],
                       v6_headings[idx], C_PLAN, zorder=7)

# Start / end markers
ax1.scatter(x_overtaking[0], y_overtaking[0], color=C_PLAN, s=60, marker='o',
            edgecolors='#fff', linewidths=1.0, zorder=8, label='Start')
ax1.scatter(x_overtaking[-1], y_overtaking[-1], color=C_PLAN, s=60, marker='s',
            edgecolors='#fff', linewidths=1.0, zorder=8, label='End')

# Overtaking point — gold diamond at optimal position
gap_values = gap_df['safe_upper'].values - gap_df['safe_lower'].values
max_gap_idx = np.argmax(gap_values)
optimal_idx = min(max_gap_idx, len(x_overtaking) - 1)
ax1.scatter(x_overtaking[optimal_idx], y_overtaking[optimal_idx],
            color='#FFD700', s=130, marker='D',
            edgecolors='#333', linewidths=1.0, zorder=8)
ax1.annotate('Overtaking Point', xy=(x_overtaking[optimal_idx], y_overtaking[optimal_idx]),
             xytext=(x_overtaking[optimal_idx] + 10, y_overtaking[optimal_idx] + 0.5),
             fontsize=6, fontweight='bold', color='#FFD700',
             arrowprops=dict(arrowstyle='->', color='#FFD700', lw=0.8),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a', alpha=0.85,
                       edgecolor='#FFD700', linewidth=0.6),
             zorder=9)

# Driving direction arrow
ax1.annotate('', xy=(x1_hi - 8, road_y_min - 0.8), xytext=(x1_lo + 8, road_y_min - 0.8),
             arrowprops=dict(arrowstyle='->', color='#888888', lw=1.8))
ax1.text((x1_lo + x1_hi) / 2, road_y_min - 1.2, 'Driving Direction',
         fontsize=7, color='#888888', ha='center', va='top', fontweight='bold')

ax1.set_xlim(x1_lo, x1_hi)
ax1.set_ylim(road_y_min - 1.0, road_y_max + 1.0)

# Style — light theme to match light background
ax1.set_title("2D Bird's-Eye View: Highway Overtaking Scene", fontsize=11,
              fontweight='bold', color='#2c3e50', pad=8)
ax1.set_xlabel('Longitudinal Position X (m)', fontsize=9, color='#555555', labelpad=4)
ax1.set_ylabel('Lateral Position Y (m)', fontsize=9, color='#555555', labelpad=4)
ax1.tick_params(colors='#666666', labelsize=7)
for spine in ax1.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(0.8)
ax1.grid(True, alpha=0.3, color='#cccccc', linewidth=0.5)
leg = ax1.legend(loc='upper right', fontsize=6, ncol=2, framealpha=0.9,
                 edgecolor='#cccccc', facecolor='white', labelcolor='#333333')

# ----- Plot 2: 3D Lateral Position vs Time -----
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
t2_lo, t2_hi = 0.0, 10.0

# Draw lane surfaces (time axis replaces x axis)
draw_3d_lane_surface(ax2, t2_lo, t2_hi, y_min, y_max, lane_boundary_y, C_V3, C_V0)

# V0 confidence band floating above road
fill_3d_band(ax2, t_all,
             predictions[0]['mean'][1] - CONF_SIGMA * predictions[0]['std'][1],
             predictions[0]['mean'][1] + CONF_SIGMA * predictions[0]['std'][1],
             LANE_Z_RAISE, C_V0, alpha=0.15)
ax2.plot(t_all, predictions[0]['mean'][1], LANE_Z_RAISE,
         color=C_V0, linewidth=2, label='V0 mean')

# V3 confidence band
fill_3d_band(ax2, t_all,
             predictions[3]['mean'][1] - CONF_SIGMA * predictions[3]['std'][1],
             predictions[3]['mean'][1] + CONF_SIGMA * predictions[3]['std'][1],
             LANE_Z_RAISE, C_V3, alpha=0.15)
ax2.plot(t_all, predictions[3]['mean'][1], LANE_Z_RAISE,
         color=C_V3, linewidth=2, label='V3 mean')

# V6 planned trajectory
ax2.plot(t_overtaking, y_overtaking, LANE_Z_RAISE + 0.1,
         color=C_PLAN, linewidth=2.8, label='V6 planned')

# Safe region band
fill_3d_band(ax2, gap_df['time'].values, gap_df['safe_lower'].values,
             gap_df['safe_upper'].values, LANE_Z_RAISE - 0.15, C_SAFE, alpha=0.10)

ax2.set_xlim(t2_lo, t2_hi)
ax2.set_ylim(y_min, y_max)
ax2.set_zlim(LANE_Z_DEPTH - 0.2, LANE_Z_RAISE + 0.5)
ax2.view_init(elev=28, azim=-55)
style_3d_ax(ax2, '3D Lateral Position vs Time', 'Time (s)', 'Y (m)')
ax2.legend(loc='upper left', fontsize=6, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 3: 3D Gap Analysis -----
ax3 = fig.add_subplot(gs[0, 2], projection='3d')
t3_lo, t3_hi = 0.0, 10.0

# Draw lane surfaces
draw_3d_lane_surface(ax3, t3_lo, t3_hi, y_min, y_max, lane_boundary_y, C_V3, C_V0)

# Safe region (prominent green band floating above road)
fill_3d_band(ax3, gap_df['time'].values, gap_df['safe_lower'].values,
             gap_df['safe_upper'].values, LANE_Z_RAISE, C_SAFE, alpha=0.25)

# V0 lower bound
ax3.plot(gap_df['time'].values, gap_df['v0_lower'].values, LANE_Z_RAISE,
         color=C_V0, linewidth=1.5, linestyle='--', label='V0 lower (μ-2σ)')
# V3 upper bound
ax3.plot(gap_df['time'].values, gap_df['v3_upper'].values, LANE_Z_RAISE,
         color=C_V3, linewidth=1.5, linestyle='--', label='V3 upper (μ+2σ)')
# V6 trajectory
ax3.plot(t_overtaking, y_overtaking, LANE_Z_RAISE + 0.1,
         color=C_PLAN, linewidth=2.8, label='V6 trajectory')

ax3.set_xlim(t3_lo, t3_hi)
ax3.set_ylim(y_min, y_max)
ax3.set_zlim(LANE_Z_DEPTH - 0.2, LANE_Z_RAISE + 0.5)
ax3.view_init(elev=28, azim=-55)
style_3d_ax(ax3, '3D Gap Analysis: Safe Region', 'Time (s)', 'Y (m)')
ax3.legend(loc='lower left', fontsize=6, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 4: Optuna tuning -----
ax4 = fig.add_subplot(gs[1, 0])
trials_v0 = [(t.number, t.value) for t in study_v0.trials if t.value is not None]
if trials_v0:
    nums, vals = zip(*trials_v0)
    ax4.scatter(nums, vals, color=C_V0, s=50, alpha=0.7, label='V0 trials', zorder=3)
    ax4.plot(nums, np.minimum.accumulate(vals), color=C_V0, linewidth=2, linestyle='-')

trials_v3 = [(t.number, t.value) for t in study_v3.trials if t.value is not None]
if trials_v3:
    nums, vals = zip(*trials_v3)
    ax4.scatter(nums, vals, color=C_V3, s=50, alpha=0.7, label='V3 trials', zorder=3)
    ax4.plot(nums, np.minimum.accumulate(vals), color=C_V3, linewidth=2, linestyle='-')

# Compute tight Y limits for Optuna
all_trial_vals = [v for _, v in trials_v0] + [v for _, v in trials_v3]
ax4.set_ylim(min(all_trial_vals) * 0.9, max(all_trial_vals) * 1.05)

style_ax(ax4, 'Optuna Hyperparameter Tuning', 'Trial', 'Loss')
ax4.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 5: Velocity estimation -----
ax5 = fig.add_subplot(gs[1, 1])
ax5.fill_between(t_overtaking,
                  inv_v_mean - CONF_SIGMA * inv_v_std,
                  inv_v_mean + CONF_SIGMA * inv_v_std,
                  color=C_INV_V, alpha=0.18)
ax5.plot(t_overtaking, inv_v_mean, color=C_INV_V, linewidth=2.5, label='Estimated')

# Tight Y limits
v_all = np.concatenate([inv_v_mean,
                          inv_v_mean - CONF_SIGMA * inv_v_std,
                          inv_v_mean + CONF_SIGMA * inv_v_std])
ax5.set_ylim(v_all.min() - 0.05, v_all.max() + 0.05)
style_ax(ax5, 'Inverse: Velocity Profile', 'Time (s)', 'Velocity (m/s)')
ax5.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 6: Heading estimation -----
ax6 = fig.add_subplot(gs[1, 2])
ax6.fill_between(t_overtaking,
                  inv_h_mean - CONF_SIGMA * inv_h_std,
                  inv_h_mean + CONF_SIGMA * inv_h_std,
                  color=C_INV_H, alpha=0.18)
ax6.plot(t_overtaking, inv_h_mean, color=C_INV_H, linewidth=2.5, label='Estimated')

# Tight Y limits
h_all = np.concatenate([inv_h_mean,
                         inv_h_mean - CONF_SIGMA * inv_h_std,
                         inv_h_mean + CONF_SIGMA * inv_h_std])
ax6.set_ylim(h_all.min() - 0.002, h_all.max() + 0.002)
style_ax(ax6, 'Inverse: Heading Profile', 'Time (s)', 'Heading (rad)')
ax6.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# ----- Distribution plots with KDE -----
from scipy.stats import gaussian_kde

time_indices = [20, 50, 80]  # t=2, 5, 8s
colors_dist = ['#3498DB', '#9B59B6', '#E74C3C']
labels_dist = ['t=2s', 't=5s', 't=8s']

# Compute acceleration samples for all MC passes
a_samples_all = np.array([np.gradient(inv_v_all[s, :], t_overtaking) for s in range(N_MC_SAMPLES)])

# ----- Plot 7: Velocity Distribution -----
ax7 = fig.add_subplot(gs[2, 0])
for idx, color, label in zip(time_indices, colors_dist, labels_dist):
    v_samples = inv_v_all[:, idx]
    # Histogram
    ax7.hist(v_samples, bins=25, alpha=0.35, color=color, density=True)
    # KDE overlay
    kde = gaussian_kde(v_samples)
    x_range = np.linspace(v_samples.min() - 0.05, v_samples.max() + 0.05, 200)
    ax7.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
    ax7.axvline(v_samples.mean(), color=color, linestyle='--', linewidth=1.2, alpha=0.7)

style_ax(ax7, 'Velocity Distribution', 'Velocity (m/s)', 'Density')
ax7.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 8: Heading Distribution -----
ax8 = fig.add_subplot(gs[2, 1])
for idx, color, label in zip(time_indices, colors_dist, labels_dist):
    h_samples = inv_h_all[:, idx]
    ax8.hist(h_samples, bins=25, alpha=0.35, color=color, density=True)
    kde = gaussian_kde(h_samples)
    x_range = np.linspace(h_samples.min() - 0.002, h_samples.max() + 0.002, 200)
    ax8.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
    ax8.axvline(h_samples.mean(), color=color, linestyle='--', linewidth=1.2, alpha=0.7)

style_ax(ax8, 'Heading Distribution', 'Heading (rad)', 'Density')
ax8.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# ----- Plot 9: Acceleration Distribution -----
ax9 = fig.add_subplot(gs[2, 2])
for idx, color, label in zip(time_indices, colors_dist, labels_dist):
    a_samples = a_samples_all[:, idx]
    ax9.hist(a_samples, bins=25, alpha=0.35, color=color, density=True)
    kde = gaussian_kde(a_samples)
    x_range = np.linspace(a_samples.min() - 0.05, a_samples.max() + 0.05, 200)
    ax9.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
    ax9.axvline(a_samples.mean(), color=color, linestyle='--', linewidth=1.2, alpha=0.7)

style_ax(ax9, 'Acceleration Distribution', 'Acceleration (m/s²)', 'Density')
ax9.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='#dddddd')

# Add suptitle
fig.suptitle('BPINN Overtaking System — Complete Analysis', fontsize=14, fontweight='bold',
             color='#2c3e50', y=0.98)

plt.savefig('results/bpinn_overtaking_results_v2.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# 9b. 2D Flat Version — Three plots with prominent lane lines
# ============================================================
print("\n  Generating 2D flat version...")

fig2, axes2d = plt.subplots(1, 3, figsize=(38, 6), facecolor='white')

def draw_2d_lane_lines(ax, x_lo, x_hi, y_min, y_max, y_boundary):
    """Draw three prominent lane lines + two lane fills in 2D."""
    # Lane fills
    ax.axhspan(y_min, y_boundary, color=C_V3, alpha=0.06, zorder=0)
    ax.axhspan(y_boundary, y_max, color=C_V0, alpha=0.06, zorder=0)
    # Line 1: Left edge (solid yellow)
    ax.axhline(y=y_min, color='#F1C40F', linestyle='-', linewidth=2.5, alpha=0.8, zorder=1)
    # Line 2: Center boundary (dashed white)
    ax.axhline(y=y_boundary, color='#7f8c8d', linestyle='--', linewidth=2.5, alpha=0.9, zorder=1)
    # Line 3: Right edge (solid yellow)
    ax.axhline(y=y_max, color='#F1C40F', linestyle='-', linewidth=2.5, alpha=0.8, zorder=1)

# --- 2D Plot 1: Top-Down ---
ax2d1 = axes2d[0]
draw_2d_lane_lines(ax2d1, x1_lo, x1_hi, y_min, y_max, lane_boundary_y)

ax2d1.fill_between(predictions[0]['mean'][0],
                    predictions[0]['mean'][1] - CONF_SIGMA * predictions[0]['std'][1],
                    predictions[0]['mean'][1] + CONF_SIGMA * predictions[0]['std'][1],
                    color=C_V0, alpha=0.28, zorder=2)
ax2d1.plot(v0_df['x'], v0_df['y'], color=C_V0, linewidth=2.2, label='V0 predicted', zorder=3)
# ax2d1.plot(predictions[0]['mean'][0], predictions[0]['mean'][1],
#            color=C_V0, linestyle=':', linewidth=1.0, alpha=0.5, zorder=2)

ax2d1.fill_between(predictions[3]['mean'][0],
                    predictions[3]['mean'][1] - CONF_SIGMA * predictions[3]['std'][1],
                    predictions[3]['mean'][1] + CONF_SIGMA * predictions[3]['std'][1],
                    color=C_V3, alpha=0.28, zorder=2)
ax2d1.plot(v3_df['x'], v3_df['y'], color=C_V3, linewidth=2.2, label='V3 predicted', zorder=3)
# ax2d1.plot(predictions[3]['mean'][0], predictions[3]['mean'][1],
#            color=C_V3, linestyle=':', linewidth=1.0, alpha=0.5, zorder=2)

ax2d1.plot(x_overtaking, y_overtaking, color=C_PLAN, linewidth=2.8, label='V6 planned', zorder=4)
ax2d1.scatter(x_overtaking[0], y_overtaking[0], color=C_PLAN, s=70, marker='o',
              edgecolors='#333', linewidths=0.6, zorder=5)
ax2d1.scatter(x_overtaking[-1], y_overtaking[-1], color=C_PLAN, s=70, marker='s',
              edgecolors='#333', linewidths=0.6, zorder=5)

ax2d1.set_xlim(x1_lo, x1_hi)
ax2d1.set_ylim(y_min, y_max)
style_ax(ax2d1, '2D Top-Down: Trajectories + Confidence Bands', 'X Position (m)', 'Y Position (m)')
ax2d1.legend(loc='lower right', fontsize=7, ncol=2, framealpha=0.8, edgecolor='#dddddd')

# --- 2D Plot 2: Lateral Position vs Time ---
ax2d2 = axes2d[1]
draw_2d_lane_lines(ax2d2, 0, 10, y_min, y_max, lane_boundary_y)

ax2d2.fill_between(t_all,
                    predictions[0]['mean'][1] - CONF_SIGMA * predictions[0]['std'][1],
                    predictions[0]['mean'][1] + CONF_SIGMA * predictions[0]['std'][1],
                    color=C_V0, alpha=0.28, zorder=2)
ax2d2.plot(t_all, predictions[0]['mean'][1], color=C_V0, linewidth=2, label='V0 mean', zorder=3)

ax2d2.fill_between(t_all,
                    predictions[3]['mean'][1] - CONF_SIGMA * predictions[3]['std'][1],
                    predictions[3]['mean'][1] + CONF_SIGMA * predictions[3]['std'][1],
                    color=C_V3, alpha=0.28, zorder=2)
ax2d2.plot(t_all, predictions[3]['mean'][1], color=C_V3, linewidth=2, label='V3 mean', zorder=3)

ax2d2.plot(t_overtaking, y_overtaking, color=C_PLAN, linewidth=2.8, label='V6 planned', zorder=4)
ax2d2.fill_between(gap_df['time'], gap_df['safe_lower'], gap_df['safe_upper'],
                   color=C_SAFE, alpha=0.10, zorder=1)

ax2d2.set_xlim(0, 10)
ax2d2.set_ylim(y_min, y_max)
style_ax(ax2d2, '2D Lateral Position vs Time', 'Time (s)', 'Y Position (m)')
ax2d2.legend(loc='upper right', fontsize=7, framealpha=0.8, edgecolor='#dddddd')

# --- 2D Plot 3: Gap Analysis ---
ax2d3 = axes2d[2]
draw_2d_lane_lines(ax2d3, 0, 10, y_min, y_max, lane_boundary_y)

ax2d3.fill_between(gap_df['time'], gap_df['safe_lower'], gap_df['safe_upper'],
                   color=C_SAFE, alpha=0.25, label='Safe region', zorder=2)
ax2d3.plot(gap_df['time'], gap_df['v0_lower'], color=C_V0, linewidth=1.5, linestyle='--',
           label='V0 lower (μ-2σ)', zorder=3)
ax2d3.plot(gap_df['time'], gap_df['v3_upper'], color=C_V3, linewidth=1.5, linestyle='--',
           label='V3 upper (μ+2σ)', zorder=3)
ax2d3.plot(t_overtaking, y_overtaking, color=C_PLAN, linewidth=2.8, label='V6 trajectory', zorder=4)

ax2d3.set_xlim(0, 10)
ax2d3.set_ylim(y_min, y_max)
style_ax(ax2d3, '2D Gap Analysis: Safe Region', 'Time (s)', 'Y Position (m)')
ax2d3.legend(loc='lower right', fontsize=6.5, framealpha=0.8, edgecolor='#dddddd')

fig2.suptitle('BPINN Overtaking — 2D Flat View (Three Lane Lines, Two Lanes)',
              fontsize=13, fontweight='bold', color='#2c3e50', y=1.02)
plt.tight_layout()
plt.savefig('results/bpinn_overtaking_2d_flat.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  2D flat version saved: bpinn_overtaking_2d_flat.png")

# ============================================================
# 10. Summary
# ============================================================
print("\n" + "=" * 70)
print("  SYSTEM COMPLETE - SUMMARY")
print("=" * 70)
print("""
  Stage 0 - Optuna Tuning:
    V0:  hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f
    V3:  hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f
    Inv: hidden=%s, dropout=%.3f, lr=%.5f, phys_w=%.3f

  Stage 1 - Forward Prediction:
    V0: σ_y = %.3f m
    V3: σ_y = %.3f m

  Stage 2 - Safe Overtaking:
    Safe region: t=[%.1f, %.1f]s, gap=%.2f~%.2f m
    Violations: %d/%d points
    Y range: %.2f ~ %.2f m

  Stage 3 - Inverse Estimation:
    Velocity:   %.2f ± %.2f m/s (%.1f ± %.1f km/h)
    Heading:    %.4f ± %.4f rad (%.2f ± %.2f deg)
    Acceleration: %.3f ± %.3f m/s²
""" % (
    best_v0['hidden_dims'], best_v0['dropout_rate'], best_v0['lr'], best_v0['physics_weight'],
    best_v3['hidden_dims'], best_v3['dropout_rate'], best_v3['lr'], best_v3['physics_weight'],
    best_inv['hidden_dims'], best_inv['dropout_rate'], best_inv['lr'], best_inv['physics_weight'],
    predictions[0]['std'][1].mean(),
    predictions[3]['std'][1].mean(),
    valid_gap['time'].min(), valid_gap['time'].max(),
    valid_gap['gap'].min(), valid_gap['gap'].max(),
    violations, len(t_overtaking),
    y_overtaking.min(), y_overtaking.max(),
    inv_v_mean.mean(), inv_v_mean.std(),
    inv_v_mean.mean() * 3.6, inv_v_mean.std() * 3.6,
    inv_h_mean.mean(), inv_h_mean.std(),
    np.degrees(inv_h_mean.mean()), np.degrees(inv_h_mean.std()),
    abs(inv_a).mean(), np.std(inv_a),
))

print("  Output files:")
print("    bpinn_overtaking_results_v2.png - 9-panel visualization")
print("    bpinn_overtaking_system_v2.py   - This script")
print("=" * 70)