# BPINN Overtaking System — Code Explanation

This document explains the **Bayesian Physics-Informed Neural Network (BPINN) overtaking system** project. The code explanations are written in English, with a clear structure to help Chinese-speaking readers navigate the project.

---

## 1. BPINN Introduction

**BPINN (Bayesian Physics-Informed Neural Network)** is a neural-network architecture that combines **Physics-Informed Neural Networks (PINN)** with **Bayesian inference** for uncertainty quantification.

### Core idea

- **Physics constraints.** The network is regularized so that its outputs obey the kinematic equations of vehicle motion. The LaTeX code for the equations is:

  `\frac{dx}{dt} = v \cos\theta`

  `\frac{dy}{dt} = v \sin\theta`

  where `\theta` is the heading angle. These derivatives are obtained via JAX automatic differentiation (`jax.grad`).
- **Neural network input / output.**
  - Input: time `t` (a single scalar, normalized).
  - Output: a 4-dimensional vector `[x, y, v, \theta]` representing longitudinal position, lateral position, velocity, and heading.
- **Bayesian uncertainty via MC Dropout.** Dropout layers are kept active at inference time. By running many stochastic forward passes (Monte Carlo sampling, e.g. 100 samples), the mean prediction `\mu` and standard deviation `\sigma` are computed:

  `\mu = \frac{1}{N}\sum_{i=1}^{N}\hat{y}_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - \mu)^2}`

  The standard deviation acts as an **epistemic uncertainty** estimate, giving confidence bands around the trajectory.

### Three stages of the system

1. **Forward prediction.** Train BPINNs on the observed trajectories of surrounding vehicles (V0, V3) and predict their future motion with uncertainty bands.
2. **Trajectory planning.** Use the confidence bands to find a safe overtaking corridor — a region strictly between the bands of the lead vehicle (V0) and the reference vehicle (V3) — and design a smooth planned trajectory (V6).
3. **Inverse parameter estimation.** Train a BPINN on the *planned* trajectory (x, y only) to inversely recover the underlying velocity, heading, and acceleration profiles, again with uncertainty distributions.

### Framework

The implementation uses **JAX** (automatic differentiation + JIT compilation), **Flax** (neural-network modules), **Optax** (Adam optimizer), **Optuna** (hyperparameter tuning), and **SciPy / Matplotlib** (interpolation and visualization).

---

## 1.5 Overtaking Algorithm Introduction

The overtaking algorithm is a **three-stage pipeline** that leverages BPINN uncertainty estimates to guarantee a collision-free lane-change maneuver.

### Stage 1: Forward Trajectory Prediction with Uncertainty

Given the observed trajectories of the two surrounding vehicles (V0 — slow lead, V3 — reference), a BPINN is trained for each vehicle. The network learns to map time `t` to the vehicle state `[x, y, v, \theta]`. During inference, **Monte Carlo Dropout** (100 stochastic forward passes) produces a predictive distribution. The mean `\mu(t)` and standard deviation `\sigma(t)` define a confidence band:

`y_{\text{upper}}(t) = \mu_y(t) + k \cdot \sigma_y(t), \quad y_{\text{lower}}(t) = \mu_y(t) - k \cdot \sigma_y(t)`

where `k` is the confidence multiplier (e.g., `k = 2` for ~95%, `k = 5` for the bird's-eye plot). The physics-informed constraint ensures that the predicted trajectory obeys the bicycle model:

`\frac{dx}{dt} = v \cos\theta, \quad \frac{dy}{dt} = v \sin\theta`

### Stage 2: Safe Corridor Extraction and Trajectory Planning

At each time step `t`, the algorithm extracts the **safe lateral corridor** as the gap between the confidence bands of V0 (above) and V3 (below):

`\text{safe\_lower}(t) = \mu_y^{V3}(t) + k \cdot \sigma_y^{V3}(t) + m`

`\text{safe\_upper}(t) = \mu_y^{V0}(t) - k \cdot \sigma_y^{V0}(t) - m`

where `m` is a safety margin (0.5 m). The gap is valid when:

`\text{gap}(t) = \text{safe\_upper}(t) - \text{safe\_lower}(t) > 0`

A planned trajectory `y_{\text{plan}}(t)` is then constructed using waypoints placed at the safe-center of valid time points, interpolated by a **monotonic PCHIP interpolator** (Piecewise Cubic Hermite Interpolating Polynomial) to avoid artificial oscillations, and clamped to the safe corridor:

`y_{\text{plan}}(t) = \text{clip}\left(\text{PCHIP}(\{t_i, y_i^{\text{wp}}\}, t),\; \text{safe\_lower}(t),\; \text{safe\_upper}(t)\right)`

The longitudinal position `x_{\text{plan}}(t)` is taken from V6's actual trajectory.

### Stage 3: Inverse Parameter Estimation

A second BPINN is trained on the **planned** trajectory (using only `x, y` as supervision) to inversely recover the underlying control parameters — velocity, heading, and acceleration. The inverse network uses the same physics constraints but with an x,y-only data loss:

`\mathcal{L}_{\text{inv}} = \mathcal{L}_{\text{data}}^{xy} + \lambda \cdot \mathcal{L}_{\text{physics}}`

where:

`\mathcal{L}_{\text{data}}^{xy} = \frac{1}{N} \sum_{i=1}^{N} \left[ (x_{\text{pred}} - x_{\text{true}})^2 + (y_{\text{pred}} - y_{\text{true}})^2 \right]`

`\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \left(\frac{dx}{dt} - v\cos\theta\right)^2 + \left(\frac{dy}{dt} - v\sin\theta\right)^2 \right]`

Again, MC Dropout provides **distributional estimates** of velocity `\hat{v}(t) \pm \sigma_v(t)`, heading `\hat{\theta}(t) \pm \sigma_\theta(t)`, and acceleration (derived as `a(t) = d\hat{v}/dt`).

### Algorithm Flow Summary

1. **Train** BPINN for V0 and V3 → predict `\mu(t), \sigma(t)` via MC Dropout
2. **Extract** safe corridor between confidence bands → `\text{safe\_lower}(t)`, `\text{safe\_upper}(t)`
3. **Plan** smooth trajectory `y_{\text{plan}}(t)` through the corridor via PCHIP interpolation
4. **Train inverse BPINN** on planned `(x, y)` → recover `\hat{v}, \hat{\theta}, \hat{a}` with uncertainty
5. **Validate** safety: count violations where `y_{\text{plan}}(t) \notin [\text{safe\_lower}(t), \text{safe\_upper}(t)]`

---

## 2. Dataset Introduction

The dataset is stored in `overtaking_scene_3vehicles.csv`. It contains **303 rows** in total, covering **3 vehicles** (V0, V3, V6) with **101 rows per vehicle**. The time range is **0.0 – 10.0 s** sampled at **0.1 s** intervals. The columns are:

`vehicle_id, time, x, y, velocity, accel_x, accel_y, heading, lane`

### Vehicle roles

- **V0** — Slow lead vehicle, traveling in **lane 3** (higher lateral Y), average velocity **26.25 m/s** (~94 km/h). This is the slow obstacle that the ego car needs to overtake.
- **V3** — Reference vehicle, traveling in **lane 2** (lower lateral Y), average velocity **28.10 m/s** (~101 km/h). This is the reference vehicle in the adjacent lane.
- **V6** — Overtaking vehicle (ego car), traveling in **lane 2**, average velocity **31.39 m/s** (~113 km/h). This is the vehicle that needs to plan a safe path between V0 and V3.

V0 travels in lane 3 (higher lateral Y) and is the slow obstacle to be overtaken. V3 travels in lane 2 (lower lateral Y) and is the reference. V6 is the ego vehicle that needs to plan a safe path between V0 and V3.

---

## 3. Code Explanation — bpinn_overtaking_main.py

> This corresponds to the file `bpinn_overtaking_system_v2.py`. It is the main script that runs the full BPINN pipeline and produces a 9-panel visualization.

### Section 0: Configuration

Defines global constants and hyperparameters:

```python
RNG_KEY = random.PRNGKey(42)
N_EPOCHS_TRAIN = 3000      # full training epochs (reduced for Kaggle)
N_EPOCHS_TUNING = 800      # shorter epochs for tuning (reduced for Kaggle)
N_MC_SAMPLES = 50          # Monte Carlo dropout samples (reduced for Kaggle)
CONF_SIGMA = 2.0           # confidence band = mean ± 2σ
MARGIN = 0.5               # safety margin in meters
N_OPTUNA_TRIALS = 8        # Optuna trials per model (reduced for Kaggle)
```

Also stores `DATA_PATH` (input CSV) and `OUTPUT_DIR` (output figures). The fixed RNG seed ensures reproducibility.

### Section 1: Data Loading

```python
df = pd.read_csv(DATA_PATH)
```

Reads the CSV and builds a `NORM` dictionary mapping each vehicle id (0, 3, 6) to its mean/std for `time`, `x`, `y`, `velocity`, `heading`. These statistics are used to **normalize** inputs and outputs before training (zero mean, unit variance), which stabilizes optimization. The per-vehicle dataframes `v0_df`, `v3_df`, `v6_df` are sorted by time for later use.

### Section 2: BPINN Model Definition

```python
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
```

- A Flax `nn.Module` with configurable `hidden_dims` (a tuple of layer widths) and `dropout_rate`.
- The `@nn.compact` decorator builds layers inline.
- Each hidden layer is `Dense -> Dropout -> tanh`.
- The output layer is `Dense(4)`, producing `[x, y, v, theta]`.
- The `training` flag controls dropout: when `training=True`, dropout is **stochastic** (used both during training and during MC-sampling inference); when `False`, dropout is disabled (used for the physics-loss forward pass, which must be deterministic for clean gradients).

A helper `make_apply_fn` wraps `model.apply` so it can be called with a plain params dict.

### Section 3: Loss Functions

Two loss components are combined.

**Physics loss** — enforces the kinematic constraints:

```python
def physics_loss_fn(params, model, t_batch, apply_fn):
    # dx/dt = v * cos(theta),  dy/dt = v * sin(theta)
    dx_dt = grad(lambda t: state_fn(t)[0])(t_single)
    dy_dt = grad(lambda t: state_fn(t)[1])(t_single)
    phys_x = dx_dt - v_pred * jnp.cos(h_pred)
    phys_y = dy_dt - v_pred * jnp.sin(h_pred)
    return phys_x**2 + phys_y**2
```

The network output `[x, y, v, \theta]` is treated as a differentiable function of `t`. `jax.grad` computes the time derivatives `\frac{dx}{dt}` and `\frac{dy}{dt}`, and the residual against `v\cos\theta` / `v\sin\theta` is squared:

`\mathcal{L}_{\text{physics}} = \frac{1}{N}\sum_{i=1}^{N}\left[\left(\frac{dx_i}{dt} - v_i\cos\theta_i\right)^2 + \left(\frac{dy_i}{dt} - v_i\sin\theta_i\right)^2\right]`

`vmap` parallelizes the computation across the batch. Dropout is off (`training=False`) here so the physics residual is smooth.

**Data loss** — supervised MSE against observed states:

`\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^{N}\left[(x_i - x_i^{\text{true}})^2 + (y_i - y_i^{\text{true}})^2 + (v_i - v_i^{\text{true}})^2 + (\theta_i - \theta_i^{\text{true}})^2\right]`

For the inverse problem, only `x, y` are used:

`\mathcal{L}_{\text{data}}^{xy} = \frac{1}{N}\sum_{i=1}^{N}\left[(x_i - x_i^{\text{true}})^2 + (y_i - y_i^{\text{true}})^2\right]`

`total_loss` returns `\mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}` plus an auxiliary dict for logging. `total_loss_xy` is the x,y-only variant for the inverse stage.

### Section 4: Training Function

```python
def train_bpinn(vehicle_id, t_train, x_train, ..., n_epochs):
    # normalize with NORM[vehicle_id]
    model = BPINN(hidden_dims=..., dropout_rate=...)
    optimizer = optax.adam(learning_rate=lr)

    @jit
    def train_step(params, opt_state, t_b, x_b, ...):
        loss_val, aux = total_loss(...)
        grads = jax.grad(...)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    for epoch in range(n_epochs):
        params, opt_state, loss_val, aux = train_step(...)
    return params, model, float(loss_val)
```

The function normalizes the data, initializes the model and Adam optimizer, and defines a `@jit`-compiled training step that computes the loss, gradients, and applies updates. The loop runs for `n_epochs`. JIT compilation makes each step fast.

### Section 5: Optuna Hyperparameter Tuning

```python
def objective(trial):
    hidden_dims_choice = trial.suggest_categorical('hidden_dims', [
        (32, 32, 32), (64, 64, 64), (128, 128, 128), (64, 128, 64)])
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    physics_weight = trial.suggest_float('physics_weight', 0.1, 5.0, log=True)
    ...train_bpinn(...) with N_EPOCHS_TUNING ...
    return final_loss
```

- Uses Optuna's **TPE sampler** (Tree-structured Parzen Estimator) with `direction='minimize'`.
- Runs **8 trials** (`N_OPTUNA_TRIALS`) for each of three studies:
  1. `study_v0` — tuning for V0 (full-state loss).
  2. `study_v3` — tuning for V3 (full-state loss).
  3. `study_inv` — tuning for the inverse problem (x,y-only loss, using V6's actual trajectory as the target during tuning, with `physics_weight` range up to 10.0).
- Each trial trains a quick model (`N_EPOCHS_TUNING = 800`) and returns the final loss. The best hyperparameters (`best_v0`, `best_v3`, `best_inv`) are reused in the later stages.

### Section 6: Stage 1 — Forward BPINN Prediction

Trains V0 and V3 with their **best** hyperparameters for the full `N_EPOCHS_TRAIN = 3000` epochs. Then runs **MC Dropout**:

```python
for i in range(N_MC_SAMPLES):           # 50 stochastic passes
    rng = random.fold_in(rng_key, i)
    out = model.apply({'params': params}, t_n, training=True, rngs={'dropout': rng})
    all_preds.append(out)
mean_pred = jnp.mean(all_preds, axis=0)
std_pred  = jnp.std(all_preds, axis=0)
```

Keeping `training=True` keeps dropout active, so each pass yields a slightly different prediction. The mean and standard deviation across the 100 samples give the predicted trajectory and its uncertainty:

`\mu(t) = \frac{1}{N}\sum_{i=1}^{N}\hat{y}_i(t), \quad \sigma(t) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i(t) - \mu(t)\right)^2}`

Results are **denormalized** back to physical units and stored in `predictions[vid]`.

### Section 7: Stage 2 — Safe Overtaking Trajectory

Performs a **gap analysis** at every time step `t`:

`y_{\text{lower}}^{V0}(t) = \mu_y^{V0}(t) - k \cdot \sigma_y^{V0}(t) \quad \text{(lower edge of V0 band)}`

`y_{\text{upper}}^{V3}(t) = \mu_y^{V3}(t) + k \cdot \sigma_y^{V3}(t) \quad \text{(upper edge of V3 band)}`

`\text{safe\_lower}(t) = y_{\text{upper}}^{V3}(t) + m, \quad \text{safe\_upper}(t) = y_{\text{lower}}^{V0}(t) - m`

`\text{gap}(t) = \text{safe\_upper}(t) - \text{safe\_lower}(t)`

The safe corridor is the lateral region strictly between V3's upper confidence edge and V0's lower confidence edge, shrunk by a safety margin `m = 0.5` m. Time points where `\text{gap}(t) > 0` form the **valid overtaking window**. If no safe region exists, the margin is removed as a fallback.

A planned trajectory is built from waypoints (start/end follow V6's actual lane, middle waypoints use the safe-center), interpolated with a **monotonic PchipInterpolator** (Piecewise Cubic Hermite, no artificial wiggles), and clamped into the safe region at every point:

`y_{\text{plan}}(t) = \text{clip}\left(\text{PCHIP}(\{t_i, y_i^{\text{wp}}\}, t),\; \text{safe\_lower}(t),\; \text{safe\_upper}(t)\right)`

The longitudinal `x` comes from V6's actual trajectory via `interp1d`. A violation count verifies the path stays inside the corridor.

### Section 8: Stage 3 — Inverse BPINN

Trains a BPINN on the **planned** trajectory using the best inverse hyperparameters and the x,y-only loss (`\mathcal{L}_{\text{data}}^{xy}`) for `N_{\text{epochs}}` epochs. Then MC Dropout is run again (100 samples) to obtain distributions of velocity and heading:

`\hat{v}(t) = \frac{1}{N}\sum_{i=1}^{N}\hat{v}_i(t), \quad \sigma_v(t) = \text{std}\left[\{\hat{v}_i(t)\}_{i=1}^{N}\right]`

`\hat{\theta}(t) = \frac{1}{N}\sum_{i=1}^{N}\hat{\theta}_i(t), \quad \sigma_\theta(t) = \text{std}\left[\{\hat{\theta}_i(t)\}_{i=1}^{N}\right]`

Acceleration is derived numerically:

`\hat{a}(t) = \frac{d\hat{v}(t)}{dt}`

This "inverts" the planned path to recover the implied velocity, heading, and acceleration profiles — each with an uncertainty band. Distribution statistics (mean ± std in m/s, rad, and m/s²) are printed.

### Section 9: Visualization (9-panel figure)

Creates a 3×3 grid figure (`bpinn_overtaking_results_v2.png`) plus a flat 2D three-panel figure (`bpinn_overtaking_2d_flat.png`).

The 9 panels are:

1. **2D Bird's-Eye View** — a realistic 4-lane highway scene with dashed lane dividers, confidence bands (smoothed with Savitzky-Golay), smoothed trajectory splines (`UnivariateSpline`), and detailed car models (body, windshield, rear window, roof, rotated by heading). A gold diamond marks the optimal overtaking point.
2. **3D Lateral Position vs Time** — 3D lane surfaces (`Poly3DCollection`) with floating confidence bands; the V6 planned trajectory is drawn above the road.
3. **3D Gap Analysis** — highlights the green safe-region band bounded by the V0 lower (`\mu - 2\sigma`) and V3 upper (`\mu + 2\sigma`) dashed lines.
4. **Optuna Hyperparameter Tuning** — scatter of trial losses with cumulative-minimum curves for V0 and V3.
5. **Inverse: Velocity Profile** — estimated velocity with `\pm 2\sigma` confidence band.
6. **Inverse: Heading Profile** — estimated heading with `\pm 2\sigma` confidence band.
7. **Velocity Distribution** — histograms + KDE curves at t = 2, 5, 8 s from the MC samples.
8. **Heading Distribution** — histograms + KDE for heading.
9. **Acceleration Distribution** — histograms + KDE for acceleration (derived by `np.gradient` of each MC velocity sample).

A final **summary** prints the best hyperparameters, uncertainty levels, safe-region time window, violation count, and inverse-estimation statistics.

---

## 4. Code Explanation — vehicle_comparison_4plots.py

This script produces a **2×2 comparison figure** (`vehicle_comparison_4plots.png`) of all three vehicles across four dimensions: X position, Y position, Velocity, and Heading. It does **not** train any neural network — the bands are synthetic.

Key points:

- **Fixed confidence half-widths** tuned so all four panels look visually uniform:
  ```python
  CI_HALF_WIDTH = {'x': 15.0, 'y': 0.45, 'velocity': 0.35, 'heading': 0.009}
  ```
- **Jagged band edges.** The `jagged_edge()` function builds irregular, asymmetric band boundaries by mixing a low-frequency sine wave (independent phase per side) with Gaussian jitter. The noise amplitude is proportional to the half-width `w`, so the mean line always stays inside the band:

  `y_{\text{upper}} = \mu + w + A\sin(\omega t + \phi_u) + \epsilon_u`

  `y_{\text{lower}} = \mu - w + A\sin(\omega t + \phi_l) + \epsilon_l`

  where `A = 0.6 \cdot w \cdot r`, `\epsilon \sim \mathcal{N}(0, 0.4 \cdot w \cdot r)`, and `r` is the jitter ratio. `add_ci_band()` then calls `fill_between` to shade the region.
- **Band rules per panel:**
  - X and Y plots: V6 (yellow) has **no band**; V0 (red) and V3 (blue) have bands.
  - Velocity and Heading plots: **all three** vehicles have bands.
- Start/end markers (circle / square) are drawn on each line. A shared `style_ax()` helper applies consistent titles, grid, and spine styling.

---

## 5. Code Explanation — vehicle_comparison_bpinn.py (vehicle_comparison_with_confidence.py)

This script is the **BPINN-trained** counterpart to Section 4. It trains a real BPINN for each of V0, V3, and V6 and plots genuine mean `\pm 4\sigma` confidence bands.

Key points:

- **Fixed hyperparameters** (no Optuna tuning here):
  ```python
  HIDDEN_DIMS = (128, 128, 128)
  DROPOUT_RATE = 0.15
  LR = 5e-4
  PHYSICS_WEIGHT = 1.0
  N_EPOCHS = 3000
  N_MC_SAMPLES = 50
  CONF_SIGMA = 4.0   # ±4σ ≈ 99.99%
  ```
- Uses the **same BPINN architecture and loss functions** as the main script (Dense+Dropout+tanh stack, physics loss with `\frac{dx}{dt} = v\cos\theta`, `\frac{dy}{dt} = v\sin\theta`, MSE data loss).
- After training, MC Dropout (100 passes) produces `mean` and `std` arrays. The `std` boundaries are smoothed with a **Savitzky-Golay filter** (`window_length=7, polyorder=3`) to remove sampling noise.
- The 2×2 figure (`vehicle_comparison_with_confidence.png`) shows X, Y, Velocity, and Heading. **All three vehicles** carry confidence bands (`\text{fill\_between}(t, \mu - 4\sigma, \mu + 4\sigma)`), the mean prediction line, and start/end scatter markers. Because these bands come from the trained model, their widths vary along time and differ between vehicles — unlike the uniform synthetic bands in Section 4.

---

## 6. Code Explanation — overtaking_3d_diagram.py

> This corresponds to the file `overtaking_3d_diagram_v4.py`. It renders a **two-panel 3D isometric diagram** (`overtaking_3d_diagram.png`) contrasting the blocked and safe overtaking states. It uses hand-rolled isometric projection rather than Matplotlib's 3D engine.

Key points:

- **Isometric projection.** A custom `project(x, y, z)` function maps 3D world coordinates to 2D screen coordinates using two rotation angles (`A = 25°`, `B = 35°`). All road, car, and band geometry is drawn with 2D `Polygon` patches positioned via this projection.
- **Road.** Only two lanes (L2 and L3) are drawn on a light-gray surface with a dashed white center divider and gray edge lines. `road_len = 90`, lane width `lw = 5.5`.
- **3D cars.** `draw_car_iso()` builds a detailed vehicle: ground shadow, body box (four side faces + top), cabin with tinted window faces, four wheels, yellow headlights, a rotated label badge, and a direction arrow. Everything is rotated by the car's heading.
- **Confidence bands.** `draw_band_iso()` extrudes a band as a strip of quadrilateral polygons between an upper and lower edge, with outline strokes.
- **Left panel — "Before Overtaking".** V3 (blue) and V0 (red) are placed close together in Y so their confidence bands **overlap**. V6 (yellow) sits behind, blocked. A red **"No Path" X** marker is drawn where the bands overlap, with a "No Path" label.
- **Right panel — "During Overtaking".** V3 and V0 are moved to their proper lane centers (far apart in Y), so a **safe corridor** (yellow translucent strip) opens between the bands. V6 is placed in the middle, and a dashed yellow **planned path** with an arrow runs through the corridor.
- A shared legend on the right panel identifies the V0 band, V3 band, safe corridor, and planned path. A bold suptitle "BPINN-Based Autonomous Overtaking System" spans both panels.

---

## Summary

| File | Purpose | Key output |
|---|---|---|
| `bpinn_overtaking_main.py` | Full BPINN pipeline: tuning → forward prediction → trajectory planning → inverse estimation → visualization | 9-panel + 3-panel figures via `plt.show()` |
| `vehicle_comparison_4plots.py` | 4-panel comparison with **synthetic** jagged 95% confidence bands | 4-panel figure via `plt.show()` |
| `vehicle_comparison_bpinn.py` | 4-panel comparison with **BPINN-trained** `\pm 4\sigma` confidence bands | 4-panel figure via `plt.show()` |
| `overtaking_3d_diagram.py` | Two-panel 3D isometric diagram: blocked vs. safe overtaking | Two-panel figure via `plt.show()` |
