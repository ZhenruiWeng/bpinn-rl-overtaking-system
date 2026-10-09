"""
2D Bird's-Eye View of 3-Vehicle Overtaking Scene
--------------------------------------------------
Visualizes V0, V6, V3 trajectories on a 2D top-down view
with stretched Y-axis for better lateral visibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# Load scene data
DATA_PATH = 'data/overtaking_scene_3vehicles.csv'
df = pd.read_csv(DATA_PATH)

# Vehicle configuration
VEHICLES = [0, 6, 3]
VEHICLE_INFO = {
    0: {'name': 'V0', 'role': 'Slow / Lead', 'color': '#e74c3c', 'lane_side': 'outer'},
    6: {'name': 'V6', 'role': 'Overtaking', 'color': '#27ae60', 'lane_side': 'inner'},
    3: {'name': 'V3', 'role': 'Reference', 'color': '#2980b9', 'lane_side': 'inner'},
}

# Find overtaking point
v0 = df[df['vehicle_id'] == 0].sort_values('time').reset_index(drop=True)
v6 = df[df['vehicle_id'] == 6].sort_values('time').reset_index(drop=True)
v3 = df[df['vehicle_id'] == 3].sort_values('time').reset_index(drop=True)

merged = pd.merge(v0[['time', 'x', 'y']], v6[['time', 'x', 'y']],
                  on='time', suffixes=('_V0', '_V6'))
merged['gap'] = merged['x_V6'] - merged['x_V0']
cross_idx = np.where(np.diff(np.sign(merged['gap'])) != 0)[0]

t_overtake = None
x_overtake = None
y0_overtake = None
y6_overtake = None

if len(cross_idx) > 0:
    idx = cross_idx[0]
    t1, t2 = merged.loc[idx, 'time'], merged.loc[idx + 1, 'time']
    g1, g2 = merged.loc[idx, 'gap'], merged.loc[idx + 1, 'gap']
    t_overtake = t1 + (0 - g1) * (t2 - t1) / (g2 - g1)
    frac = (t_overtake - t1) / (t2 - t1)
    x_overtake = merged.loc[idx, 'x_V0'] + frac * (merged.loc[idx + 1, 'x_V0'] - merged.loc[idx, 'x_V0'])
    y0_overtake = merged.loc[idx, 'y_V0'] + frac * (merged.loc[idx + 1, 'y_V0'] - merged.loc[idx, 'y_V0'])
    y6_overtake = merged.loc[idx, 'y_V6'] + frac * (merged.loc[idx + 1, 'y_V6'] - merged.loc[idx, 'y_V6'])

# ============================================================
# Create 2D Bird's-Eye View
# ============================================================
fig, ax = plt.subplots(figsize=(18, 10))

# --- Road background ---
x_min, x_max = df['x'].min() - 30, df['x'].max() + 30
y_min, y_max = df['y'].min() - 4, df['y'].max() + 4

# Asphalt
ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                       facecolor='#2c3e50', alpha=0.15, edgecolor='none', zorder=0))

# Lane markings (estimate from vehicle paths)
# Compute approximate lane centerlines by interpolation
from scipy.interpolate import interp1d

x_smooth = np.linspace(x_min + 20, x_max - 20, 500)

# Outer lane center (V0 path)
v0_sorted = v0.sort_values('x')
f_v0 = interp1d(v0_sorted['x'], v0_sorted['y'], kind='cubic',
                bounds_error=False, fill_value='extrapolate')

# Inner lane center (average of V3 and V6 paths)
v3_sorted = v3.sort_values('x')
f_v3 = interp1d(v3_sorted['x'], v3_sorted['y'], kind='cubic',
                bounds_error=False, fill_value='extrapolate')
v6_sorted = v6.sort_values('x')
f_v6 = interp1d(v6_sorted['x'], v6_sorted['y'], kind='cubic',
                bounds_error=False, fill_value='extrapolate')

# Estimate lane width from lateral gap
lane_width = np.mean(np.abs(f_v3(x_smooth) - f_v0(x_smooth))) * 0.8

# Draw lane boundaries
# Edge lines (solid)
ax.plot(x_smooth, f_v0(x_smooth) + lane_width/2,
        color='white', linewidth=3, linestyle='-', zorder=1, alpha=0.9)
ax.plot(x_smooth, (f_v3(x_smooth) + f_v6(x_smooth))/2 - lane_width/2,
        color='white', linewidth=3, linestyle='-', zorder=1, alpha=0.9)

# Center dashed line between lanes
ax.plot(x_smooth, (f_v0(x_smooth) + f_v3(x_smooth) + f_v6(x_smooth))/3,
        color='white', linewidth=2, linestyle='--', dashes=(15, 10), zorder=1, alpha=0.7)

# Inner lane center dashed line
ax.plot(x_smooth, (f_v3(x_smooth) + f_v6(x_smooth))/2,
        color='white', linewidth=1.5, linestyle=':', zorder=1, alpha=0.5)

# Lane labels
ax.text(x_min + 15, f_v0(x_min + 40) + lane_width/2 + 0.8,
        'Outer Lane  (V0)', fontsize=11, fontweight='bold', color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax.text(x_min + 15, (f_v3(x_min + 40) + f_v6(x_min + 40))/2,
        'Inner Lane  (V3, V6)', fontsize=11, fontweight='bold', color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# --- Draw vehicle trajectories ---
for vid in VEHICLES:
    vdf = df[df['vehicle_id'] == vid].sort_values('time')
    info = VEHICLE_INFO[vid]

    # Trajectory line
    ax.plot(vdf['x'], vdf['y'], color=info['color'],
            linewidth=3, alpha=0.8, zorder=3,
            label='%s - %s' % (info['name'], info['role']))

    # Trajectory glow effect
    ax.plot(vdf['x'], vdf['y'], color=info['color'],
            linewidth=8, alpha=0.15, zorder=2)

# --- Draw vehicle shapes at key time steps ---
key_times = [0.0, 2.0, t_overtake if t_overtake else 3.5, 6.0, 9.0]
key_labels = ['t=0s', 't=2s', 'Overtake', 't=6s', 't=9s']

CAR_LENGTH = 6.0  # meters
CAR_WIDTH = 2.0   # meters

def draw_vehicle(ax, x, y, heading, color, label='', alpha=1.0):
    """Draw a simplified car shape as a rectangle with heading angle."""
    # Convert heading to degrees for rotation
    heading_deg = np.degrees(heading)

    # Draw car body
    car = FancyBboxPatch(
        (x - CAR_LENGTH/2, y - CAR_WIDTH/2),
        CAR_LENGTH, CAR_WIDTH,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='black', linewidth=1.2,
        alpha=alpha, zorder=5
    )
    # Apply rotation
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(x, y, heading_deg) + ax.transData
    car.set_transform(t)
    ax.add_patch(car)

    # Direction indicator (front of car)
    front_x = x + CAR_LENGTH/2 * np.cos(heading)
    front_y = y + CAR_LENGTH/2 * np.sin(heading)
    ax.plot([x, front_x], [y, front_y], color='white',
            linewidth=2, alpha=0.8, zorder=6)

    # Label
    if label:
        ax.annotate(label, xy=(x, y + CAR_WIDTH/2 + 0.5),
                    fontsize=8, fontweight='bold', color=color,
                    ha='center', va='bottom', zorder=7)

for i, t_key in enumerate(key_times):
    for vid in VEHICLES:
        vdf = df[df['vehicle_id'] == vid].sort_values('time').reset_index(drop=True)
        idx = (vdf['time'] - t_key).abs().argsort().iloc[0]
        row = vdf.iloc[idx]

        if abs(row['time'] - t_key) < 0.15:
            info = VEHICLE_INFO[vid]
            alpha = 0.9 if i == 2 else 0.55
            label = info['name'] if i == 2 else ''
            draw_vehicle(ax, row['x'], row['y'], row['heading'],
                         info['color'], label=label, alpha=alpha)

# --- Highlight overtaking point ---
if t_overtake is not None and x_overtake is not None:
    ax.scatter([x_overtake], [(y0_overtake + y6_overtake)/2],
               color='gold', s=400, marker='*',
               edgecolors='black', linewidths=2, zorder=10)
    ax.annotate('OVERTAKING POINT\nt=%.2fs  x≈%.0fm' % (t_overtake, x_overtake),
                xy=(x_overtake, (y0_overtake + y6_overtake)/2),
                xytext=(0, 60), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkorange',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

# --- Direction arrow ---
ax.annotate('', xy=(x_max - 40, y_min + 1.0),
            xytext=(x_min + 40, y_min + 1.0),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
ax.text((x_min + x_max)/2, y_min + 1.3, 'Driving Direction →',
        fontsize=11, fontweight='bold', color='#2c3e50',
        ha='center', va='bottom')

# --- Axes configuration ---
ax.set_xlabel('Longitudinal Position  X  (m)', fontsize=13, fontweight='bold')
ax.set_ylabel('Lateral Position  Y  (m)', fontsize=13, fontweight='bold')
ax.set_title('2D Bird\'s-Eye View: 3-Vehicle Overtaking Scene\n'
             'V0 (slow/outer) ← V6 (fast/inner overtaking) → V3 (medium/inner ahead)',
             fontsize=15, fontweight='bold', pad=15)

# Stretch Y-axis (no equal aspect)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)

# Legend
custom_legend = [
    Line2D([0], [0], color=VEHICLE_INFO[0]['color'], linewidth=3,
           label='V0 - Slow Lead Vehicle (Outer Lane)'),
    Line2D([0], [0], color=VEHICLE_INFO[6]['color'], linewidth=3,
           label='V6 - Overtaking Vehicle (Inner Lane)'),
    Line2D([0], [0], color=VEHICLE_INFO[3]['color'], linewidth=3,
           label='V3 - Reference Vehicle (Inner Lane)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
           markersize=14, markeredgecolor='black', label='Overtaking Point'),
]
ax.legend(handles=custom_legend, loc='upper right', fontsize=10,
          title='Vehicle Legend', title_fontsize=11, framealpha=0.9)

# Mile markers on X axis
for x_mark in range(int(x_min/50)*50, int(x_max), 50):
    if x_min < x_mark < x_max:
        ax.axvline(x=x_mark, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

plt.tight_layout()
output_path = 'results/birdview_2d_overtaking.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print("2D Bird's-Eye View saved to:")
print("  %s" % output_path)
print()
print("Scene summary:")
print("  X range: %.0f ~ %.0f m" % (x_min, x_max))
print("  Y range: %.1f ~ %.1f m (stretched for visibility)" % (y_min, y_max))
print("  Y-axis aspect: stretched (not equal scale)")
print("  Key time points: t=0s, 2s, overtake(%.1fs), 6s, 9s" % (t_overtake or 3.5))
