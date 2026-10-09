"""
Overtaking Scene Analysis for BPINN Trajectory Prediction
-----------------------------------------------------------
This script identifies and visualizes a 3-vehicle overtaking scenario
suitable for BPINN-based trajectory prediction and overtaking system design.

Selected scene (adjacent-lane overtaking on a curved highway):
  - V0 (slow vehicle, outer lane): being overtaken (被超车辆)
  - V6 (fast vehicle, inner lane): overtaking V0 (超车车辆)
  - V3 (medium-speed vehicle, inner lane ahead): traffic context (前方参考车)

Key facts:
  - V0: 26.2 m/s (94.5 km/h) - slowest, in outer lane
  - V6: 31.4 m/s (113.0 km/h) - fast, catches up and passes V0
  - V3: 28.1 m/s (101.2 km/h) - medium speed, in same lane as V6 but ahead
  - Speed difference (V6 vs V0): 5.2 m/s (18.7 km/h)
  - Natural overtaking: V6 starts behind V0 and passes it at t≈3.5s
  - Curved road with sinusoidal profile - adds realism for BPINN study

Time window: 0.0s ~ 10.0s
X range: ~60m ~ 420m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load data
FILE_PATH = 'data/highway_trajectory_all_vehicles.csv'
df = pd.read_csv(FILE_PATH)

# ============================================================
# Scene Configuration
# ============================================================
SELECTED_VEHICLES = [0, 6, 3]  # V0: slow/被超, V6: fast/超车, V3: medium/前方参考
VEHICLE_INFO = {
    0: {'label': 'V0 - Slow / Lead (Outer Lane)', 'color': '#e74c3c', 'role': '被超车辆'},
    6: {'label': 'V6 - Fast / Overtaking (Inner Lane)', 'color': '#27ae60', 'role': '超车车辆'},
    3: {'label': 'V3 - Medium / Reference (Inner Lane Ahead)', 'color': '#2980b9', 'role': '参考车辆'},
}
T_START = 0.0
T_END = 10.0

# Extract window
window = df[(df['time'] >= T_START) & (df['time'] <= T_END) &
            (df['vehicle_id'].isin(SELECTED_VEHICLES))].copy()

# ============================================================
# Compute overtaking metrics
# ============================================================
v0 = window[window['vehicle_id'] == 0].sort_values('time').reset_index(drop=True)
v6 = window[window['vehicle_id'] == 6].sort_values('time').reset_index(drop=True)
v3 = window[window['vehicle_id'] == 3].sort_values('time').reset_index(drop=True)

# Merge for gap calculation (V6 vs V0 = overtaking pair)
merged = pd.merge(v0[['time', 'x', 'y', 'velocity']],
                  v6[['time', 'x', 'y', 'velocity']],
                  on='time', suffixes=('_V0', '_V6'))
merged = pd.merge(merged,
                  v3[['time', 'x', 'y', 'velocity']].rename(
                      columns={'x': 'x_V3', 'y': 'y_V3', 'velocity': 'velocity_V3'}),
                  on='time')

merged['gap_V6_V0'] = merged['x_V6'] - merged['x_V0']  # +: V6 ahead
merged['y_gap_V6_V0'] = abs(merged['y_V6'] - merged['y_V0'])
merged['gap_V6_V3'] = merged['x_V6'] - merged['x_V3']

# Find overtaking point
cross_idx = np.where(np.diff(np.sign(merged['gap_V6_V0'])) != 0)[0]
t_overtake = None
x_overtake = None
y0_overtake = None
y6_overtake = None

if len(cross_idx) > 0:
    idx = cross_idx[0]
    t1, t2 = merged.loc[idx, 'time'], merged.loc[idx + 1, 'time']
    g1, g2 = merged.loc[idx, 'gap_V6_V0'], merged.loc[idx + 1, 'gap_V6_V0']
    t_overtake = t1 + (0 - g1) * (t2 - t1) / (g2 - g1)
    frac = (t_overtake - t1) / (t2 - t1)
    x_overtake = merged.loc[idx, 'x_V0'] + frac * (merged.loc[idx + 1, 'x_V0'] - merged.loc[idx, 'x_V0'])
    y0_overtake = merged.loc[idx, 'y_V0'] + frac * (merged.loc[idx + 1, 'y_V0'] - merged.loc[idx, 'y_V0'])
    y6_overtake = merged.loc[idx, 'y_V6'] + frac * (merged.loc[idx + 1, 'y_V6'] - merged.loc[idx, 'y_V6'])

# Print analysis
print("=" * 65)
print("  3-VEHICLE OVERTAKING SCENE ANALYSIS (for BPINN Study)")
print("=" * 65)
print("  Time window: %.1f ~ %.1f s" % (T_START, T_END))
print("  Vehicles: V0 (被超), V6 (超车), V3 (参考)")
print("-" * 65)

for vid in SELECTED_VEHICLES:
    vdf = window[window['vehicle_id'] == vid]
    info = VEHICLE_INFO[vid]
    lane = vdf['lane'].iloc[0]
    v_mean = vdf['velocity'].mean()
    print()
    print("  Vehicle %d: %s" % (vid, info['role']))
    print("    Lane (Frenet): %d" % lane)
    print("    Speed: %.2f m/s (%.1f km/h)" % (v_mean, v_mean * 3.6))
    print("    X range: [%.1f, %.1f] m" % (vdf['x'].min(), vdf['x'].max()))
    print("    Y range: [%.2f, %.2f] m" % (vdf['y'].min(), vdf['y'].max()))

print()
print("-" * 65)
print("  Overtaking Event (V6 passing V0)")
print("-" * 65)
if t_overtake is not None:
    print("  Time:         t = %.2f s" % t_overtake)
    print("  Position:     x ≈ %.1f m" % x_overtake)
    print("  V0 Y at overtake:  %.2f m" % y0_overtake)
    print("  V6 Y at overtake:  %.2f m" % y6_overtake)
    print("  Lateral gap:  %.2f m (adjacent lane)" % abs(y0_overtake - y6_overtake))
    print("  Speed diff:   %.2f m/s (%.1f km/h)" % (
        v6['velocity'].mean() - v0['velocity'].mean(),
        (v6['velocity'].mean() - v0['velocity'].mean()) * 3.6))

    # V3 position at overtaking
    v3_row = v3.iloc[(v3['time'] - t_overtake).abs().argsort().iloc[0]]
    print()
    print("  V3 position at overtaking moment:")
    print("    x = %.1f m (%+.1f m relative to V0)" % (
        v3_row['x'], v3_row['x'] - x_overtake))
    print("    y = %.2f m (%.2f m from V6)" % (
        v3_row['y'], abs(v3_row['y'] - y6_overtake)))

print()
print("=" * 65)
print("  Why this scene is good for BPINN overtaking study:")
print("  1. Natural overtaking: V6 catches up to V0 from behind")
print("  2. Adjacent lane pass: realistic highway overtaking maneuver")
print("  3. Curved road: tests BPINN on non-straight trajectories")
print("  4. 3-vehicle interaction: V6 must consider both V0 and V3")
print("  5. Speed variation: small velocity fluctuations add realism")
print("=" * 65)

# ============================================================
# Plot 1: Comprehensive 4-panel analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Subplot 1: X-Y top-down trajectories ---
ax1 = axes[0, 0]
for vid in SELECTED_VEHICLES:
    vdf = window[window['vehicle_id'] == vid].sort_values('time')
    info = VEHICLE_INFO[vid]
    ax1.plot(vdf['x'], vdf['y'], color=info['color'], linewidth=2.5,
             alpha=0.85, label=info['label'])
    # Start point
    ax1.scatter(vdf.iloc[0]['x'], vdf.iloc[0]['y'], color=info['color'],
                s=120, marker='o', edgecolors='black', linewidths=1, zorder=5)
    # End point
    ax1.scatter(vdf.iloc[-1]['x'], vdf.iloc[-1]['y'], color=info['color'],
                s=120, marker='^', edgecolors='black', linewidths=1, zorder=5)
    ax1.annotate('V%d' % vid,
                 xy=(vdf.iloc[0]['x'], vdf.iloc[0]['y']),
                 xytext=(-15, 10), textcoords='offset points',
                 fontsize=9, color=info['color'], fontweight='bold')

# Mark overtaking point
if t_overtake is not None and x_overtake is not None:
    ax1.scatter([x_overtake, x_overtake], [y0_overtake, y6_overtake],
                color='gold', s=200, marker='*',
                edgecolors='black', linewidths=1.5, zorder=10)
    ax1.annotate('Overtaking\n(t=%.1fs)' % t_overtake,
                 xy=(x_overtake, (y0_overtake + y6_overtake) / 2),
                 xytext=(30, 0), textcoords='offset points',
                 fontsize=9, color='black', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='darkorange'))

ax1.set_xlabel('X Position (m)', fontsize=11)
ax1.set_ylabel('Y Position (m)', fontsize=11)
ax1.set_title('Top-Down View: 3-Vehicle Overtaking Scene', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='box')

# --- Subplot 2: X position vs time ---
ax2 = axes[0, 1]
for vid in SELECTED_VEHICLES:
    vdf = window[window['vehicle_id'] == vid].sort_values('time')
    info = VEHICLE_INFO[vid]
    ax2.plot(vdf['time'], vdf['x'], color=info['color'], linewidth=2,
             label='V%d (%.1f m/s)' % (vid, vdf['velocity'].mean()))

if t_overtake is not None:
    ax2.axvline(x=t_overtake, color='gold', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.scatter([t_overtake], [x_overtake], color='gold', s=100, marker='*',
                edgecolors='black', linewidths=1, zorder=5)
    ax2.annotate('Overtaking', xy=(t_overtake, x_overtake),
                 xytext=(10, -30), textcoords='offset points',
                 fontsize=9, color='darkorange', fontweight='bold')

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('X Position (m)', fontsize=11)
ax2.set_title('Longitudinal Position vs Time', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')

# --- Subplot 3: Gap between V6 and V0 (overtaking pair) ---
ax3 = axes[1, 0]
ax3.plot(merged['time'], merged['gap_V6_V0'], color=VEHICLE_INFO[6]['color'],
         linewidth=2.5, label='Longitudinal Gap (V6 - V0)')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.fill_between(merged['time'], merged['gap_V6_V0'], 0,
                 where=merged['gap_V6_V0'] < 0,
                 alpha=0.2, color=VEHICLE_INFO[0]['color'], label='V6 behind V0')
ax3.fill_between(merged['time'], merged['gap_V6_V0'], 0,
                 where=merged['gap_V6_V0'] >= 0,
                 alpha=0.2, color=VEHICLE_INFO[6]['color'], label='V6 ahead of V0')

# Lateral gap (secondary axis)
ax3_twin = ax3.twinx()
ax3_twin.plot(merged['time'], merged['y_gap_V6_V0'], color='purple',
              linewidth=1.5, linestyle='--', alpha=0.7, label='Lateral Gap |Y6-Y0|')
ax3_twin.set_ylabel('Lateral Gap (m)', fontsize=10, color='purple')
ax3_twin.tick_params(axis='y', labelcolor='purple')

if t_overtake is not None:
    ax3.axvline(x=t_overtake, color='gold', linestyle='--', linewidth=1.5)
    ax3.scatter([t_overtake], [0], color='gold', s=100, marker='*',
                edgecolors='black', linewidths=1, zorder=5)
    ax3.annotate('Overtaking\nPoint', xy=(t_overtake, 0),
                 xytext=(15, 40), textcoords='offset points',
                 fontsize=9, color='darkorange', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkorange'))

ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Longitudinal Gap (m)\nV6 - V0', fontsize=11)
ax3.set_title('Gap Evolution: V6 (Overtaking) vs V0 (Lead)', fontsize=12, fontweight='bold')

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

ax3.grid(True, alpha=0.3, linestyle='--')

# --- Subplot 4: Velocity comparison ---
ax4 = axes[1, 1]
for vid in SELECTED_VEHICLES:
    vdf = window[window['vehicle_id'] == vid].sort_values('time')
    info = VEHICLE_INFO[vid]
    ax4.plot(vdf['time'], vdf['velocity'], color=info['color'], linewidth=2,
             label='V%d (avg: %.1f m/s)' % (vid, vdf['velocity'].mean()))

# Mark overtaking time
if t_overtake is not None:
    ax4.axvline(x=t_overtake, color='gold', linestyle='--', linewidth=1.5, alpha=0.7)

ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Velocity (m/s)', fontsize=11)
ax4.set_title('Velocity Profiles', fontsize=12, fontweight='bold')
ax4.legend(loc='center right', fontsize=9)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/overtaking_scene_analysis.png',
            dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 2: Scene schematic for BPINN setup
# ============================================================
fig2, ax = plt.subplots(figsize=(15, 7))

# Get Y bounds
all_y = window['y'].values
y_min, y_max = all_y.min() - 1.5, all_y.max() + 1.5
x_min, x_max = window['x'].min() - 30, window['x'].max() + 30

# Road background
ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           facecolor='#34495e', alpha=0.1, edgecolor='none'))

# Draw approximate lane centers by fitting a curve through vehicle paths
# For inner lane (V3, V6) and outer lane (V0)
x_smooth = np.linspace(x_min + 20, x_max - 20, 200)

# Outer lane (V0 path)
from scipy.interpolate import interp1d
v0_sorted = v0.sort_values('x')
f_v0 = interp1d(v0_sorted['x'], v0_sorted['y'], kind='cubic',
                bounds_error=False, fill_value='extrapolate')
ax.plot(x_smooth, f_v0(x_smooth), color=VEHICLE_INFO[0]['color'],
        linewidth=1, alpha=0.3, linestyle=':')

# Inner lane (V3 path - as reference)
v3_sorted = v3.sort_values('x')
f_v3 = interp1d(v3_sorted['x'], v3_sorted['y'], kind='cubic',
                bounds_error=False, fill_value='extrapolate')
ax.plot(x_smooth, f_v3(x_smooth), color=VEHICLE_INFO[3]['color'],
        linewidth=1, alpha=0.3, linestyle=':')

# Lane labels
ax.text(x_min + 10, f_v0(x_min + 50), 'Outer Lane (V0)',
        fontsize=9, color=VEHICLE_INFO[0]['color'], fontweight='bold', va='bottom')
ax.text(x_min + 10, f_v3(x_min + 50), 'Inner Lane (V3, V6)',
        fontsize=9, color=VEHICLE_INFO[3]['color'], fontweight='bold', va='bottom')

# Draw vehicle positions at key time points
key_times = [T_START, t_overtake if t_overtake else 3.5, T_END]
time_labels = ['Start', 'Overtaking', 'End']
markers = ['o', '*', 's']
sizes = [180, 220, 180]

for vid in SELECTED_VEHICLES:
    vdf = window[window['vehicle_id'] == vid].sort_values('time').reset_index(drop=True)
    info = VEHICLE_INFO[vid]

    # Full trajectory as thin dashed line
    ax.plot(vdf['x'], vdf['y'], color=info['color'],
            linewidth=1.5, alpha=0.5, linestyle='--')

    # Key points
    for i, t_key in enumerate(key_times):
        idx = (vdf['time'] - t_key).abs().argsort().iloc[0]
        row = vdf.iloc[idx]
        ax.scatter(row['x'], row['y'], color=info['color'], s=sizes[i],
                   marker=markers[i], edgecolors='black', linewidths=1.2, zorder=5 + i)
        # Label
        y_offset = 15 if vid == 0 else -18
        ax.annotate('V%d\n(%s)' % (vid, time_labels[i]),
                    xy=(row['x'], row['y']),
                    xytext=(0, y_offset), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color=info['color'],
                    ha='center', va='bottom' if y_offset > 0 else 'top')

# Direction arrow
ax.annotate('', xy=(x_max - 20, y_min + 0.5),
            xytext=(x_min + 20, y_min + 0.5),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.text((x_min + x_max) / 2, y_min + 0.8, 'Driving Direction →',
        fontsize=10, color='#2c3e50', ha='center', va='bottom')

# Legend
custom_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=VEHICLE_INFO[0]['color'],
           markersize=10, markeredgecolor='black', label='V0 - Slow / Lead Vehicle'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=VEHICLE_INFO[6]['color'],
           markersize=10, markeredgecolor='black', label='V6 - Overtaking Vehicle'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=VEHICLE_INFO[3]['color'],
           markersize=10, markeredgecolor='black', label='V3 - Reference Vehicle'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markersize=12, markeredgecolor='black', label='Overtaking Point'),
]
ax.legend(handles=custom_legend, loc='upper right', fontsize=9,
          title='Scene Legend', title_fontsize=10)

ax.set_xlabel('X Position (m)', fontsize=11)
ax.set_ylabel('Y Position (m)', fontsize=11)
ax.set_title(
    '3-Vehicle Overtaking Scene for BPINN Study\n'
    'V0 (slow/outer) + V6 (fast/inner, overtaking) + V3 (medium/inner, ahead)',
    fontsize=12, fontweight='bold')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min - 0.5, y_max + 0.5)
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('results/overtaking_scene_diagram.png',
            dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 3: Extracted scene data CSV for BPINN use
# ============================================================
# Save the 3-vehicle subset as a clean CSV for easy loading
output_df = window.sort_values(['vehicle_id', 'time']).reset_index(drop=True)
output_path = 'data/overtaking_scene_3vehicles.csv'
output_df.to_csv(output_path, index=False)

print("\nScene data saved to: %s" % output_path)
print("  Rows: %d, Columns: %s" % (len(output_df), list(output_df.columns)))
print("\nAll plots saved successfully!")
