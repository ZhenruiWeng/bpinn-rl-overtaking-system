# ============================================================
# Dynamic Overtaking Scenario Simulation
# Three-phase dynamic gap scenario for rolling-window BPINN+RL
#
# Phase 1 (t=0~3s):  Gap opens  → V6 starts overtaking
# Phase 2 (t=3~6s):  Gap closes → V6 aborts, returns to lane
# Phase 3 (t=6~10s): Gap opens  → V6 successfully overtakes
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import PchipInterpolator
import csv

np.random.seed(42)

# ============================================================
# 1. Simulation Parameters
# ============================================================
DT = 0.1          # time step (s)
T_TOTAL = 10.0    # total simulation time (s)
N_STEPS = int(T_TOTAL / DT)

# Lane geometry (4 lanes)
LANE_WIDTH = 4.0
LANE_CENTERS = [3.0, 7.0, 11.0, 15.0]  # Lane 1~4 center Y
ROAD_Y_MIN = 1.0
ROAD_Y_MAX = 17.0

# Vehicle parameters
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8

# ============================================================
# 2. Vehicle Trajectory Generation
# ============================================================
# V0: Red, slow lead vehicle, Lane 3 (y=11)
#   - Speed varies to create dynamic gap
#   - Phase 1: v=26 m/s (slow, gap grows because V3 is faster)
#   - Phase 2: v=30 m/s (speeds up, catches up to V3, gap closes)
#   - Phase 3: v=25 m/s (slows down, gap opens again)

# V3: Blue, reference vehicle, Lane 2 (y=7)
#   - Relatively constant speed ~28 m/s
#   - Slight lateral drift

# V6: Yellow, ego vehicle, Lane 2 (y=7) → Lane 3 → Lane 2
#   - Phase 1: Starts overtaking, moves from Lane 2 to Lane 3
#   - Phase 2: Aborts, returns from Lane 3 to Lane 2
#   - Phase 3: Overtakes successfully, Lane 2 → Lane 3 → Lane 2

t = np.arange(0, T_TOTAL + DT, DT)
N = len(t)

# --- V0: speed profile with 3 phases ---
# Phase 1: v=31 m/s (fast, gap to V3 grows)
# Phase 2: v=22 m/s (slow, V3 catches up, gap closes below threshold)
# Phase 3: v=32 m/s (fast again, gap opens for successful overtake)
v0_speed = np.zeros(N)
for i, ti in enumerate(t):
    if ti < 3.0:
        v0_speed[i] = 31.0 + 0.3 * np.sin(ti * 1.5)
    elif ti < 6.0:
        # Smooth transition to slow speed
        transition = (ti - 3.0) / 3.0
        v0_speed[i] = 31.0 - 9.0 * transition + 0.3 * np.sin(ti * 1.5)
    else:
        # Smooth transition back to fast
        transition = (ti - 6.0) / 4.0
        v0_speed[i] = 22.0 + 10.0 * transition + 0.3 * np.sin(ti * 1.5)

v0_speed += np.random.normal(0, 0.15, N)

# --- V3: relatively constant speed ~28 m/s ---
v3_speed = np.zeros(N)
for i, ti in enumerate(t):
    v3_speed[i] = 28.0 + 0.3 * np.sin(ti * 2.0)

v3_speed += np.random.normal(0, 0.12, N)

# --- V6: ego vehicle speed ---
# Phase 1: accelerating (going for overtake)
# Phase 2: decelerating (aborting, falling back)
# Phase 3: accelerating (second attempt, successful)
v6_speed = np.zeros(N)
for i, ti in enumerate(t):
    if ti < 3.0:
        v6_speed[i] = 30.0 + 1.5 * (ti / 3.0)
    elif ti < 6.0:
        v6_speed[i] = 31.5 - 5.0 * (ti - 3.0) / 3.0
    else:
        v6_speed[i] = 26.5 + 4.0 * (ti - 6.0) / 4.0

v6_speed += np.random.normal(0, 0.1, N)

# --- Compute positions by integration ---
# V0 starts ahead in Lane 3
v0_x = np.zeros(N)
v0_y = np.zeros(N)
v0_x[0] = 200.0
v0_y[0] = LANE_CENTERS[2]  # Lane 3
v0_heading = np.zeros(N)

for i in range(1, N):
    v0_x[i] = v0_x[i-1] + v0_speed[i-1] * DT
    v0_y[i] = LANE_CENTERS[2] + 0.3 * np.sin(t[i] * 0.8)
    v0_heading[i] = np.arctan2(v0_y[i] - v0_y[i-1], v0_x[i] - v0_x[i-1]) if abs(v0_x[i] - v0_x[i-1]) > 1e-6 else 0.0

# V3 starts behind V0 in Lane 2
v3_x = np.zeros(N)
v3_y = np.zeros(N)
v3_x[0] = 150.0
v3_y[0] = LANE_CENTERS[1]  # Lane 2
v3_heading = np.zeros(N)

for i in range(1, N):
    v3_x[i] = v3_x[i-1] + v3_speed[i-1] * DT
    v3_y[i] = LANE_CENTERS[1] + 0.4 * np.sin(t[i] * 1.2)
    v3_heading[i] = np.arctan2(v3_y[i] - v3_y[i-1], v3_x[i] - v3_x[i-1]) if abs(v3_x[i] - v3_x[i-1]) > 1e-6 else 0.0

# V6 starts behind V3 in Lane 2
v6_x = np.zeros(N)
v6_y = np.zeros(N)
v6_x[0] = 100.0
v6_y[0] = LANE_CENTERS[1]  # Lane 2
v6_heading = np.zeros(N)

# V6 trajectory: 3-phase lane changes
# Phase 1: Lane 2 → Lane 3 (start overtake)
# Phase 2: Lane 3 → Lane 2 (abort)
# Phase 3: Lane 2 → Lane 3 → Lane 2 (complete overtake)
for i in range(1, N):
    v6_x[i] = v6_x[i-1] + v6_speed[i-1] * DT
    
    ti = t[i]
    if ti < 0.5:
        # Initial: stay in Lane 2
        v6_y[i] = LANE_CENTERS[1]
    elif ti < 3.0:
        # Phase 1: Move from Lane 2 to Lane 3 (smooth)
        progress = (ti - 0.5) / 2.5
        v6_y[i] = LANE_CENTERS[1] + (LANE_CENTERS[2] - LANE_CENTERS[1]) * progress
    elif ti < 3.5:
        # Transition: stay at Lane 3 briefly
        v6_y[i] = LANE_CENTERS[2]
    elif ti < 6.0:
        # Phase 2: Abort, return from Lane 3 to Lane 2
        progress = (ti - 3.5) / 2.5
        v6_y[i] = LANE_CENTERS[2] + (LANE_CENTERS[1] - LANE_CENTERS[2]) * progress
    elif ti < 6.5:
        # Stay in Lane 2 briefly
        v6_y[i] = LANE_CENTERS[1]
    elif ti < 8.5:
        # Phase 3: Move from Lane 2 to Lane 3 (overtake)
        progress = (ti - 6.5) / 2.0
        v6_y[i] = LANE_CENTERS[1] + (LANE_CENTERS[2] - LANE_CENTERS[1]) * progress
    elif ti < 9.0:
        # Stay in Lane 3 (passing V3)
        v6_y[i] = LANE_CENTERS[2]
    else:
        # Return to Lane 2
        progress = (ti - 9.0) / 1.0
        v6_y[i] = LANE_CENTERS[2] + (LANE_CENTERS[1] - LANE_CENTERS[2]) * min(progress, 1.0)
    
    v6_heading[i] = np.arctan2(v6_y[i] - v6_y[i-1], v6_x[i] - v6_x[i-1]) if abs(v6_x[i] - v6_x[i-1]) > 1e-6 else 0.0

# Add small noise to positions
v0_x += np.random.normal(0, 0.1, N)
v0_y += np.random.normal(0, 0.05, N)
v3_x += np.random.normal(0, 0.1, N)
v3_y += np.random.normal(0, 0.05, N)

# ============================================================
# 3. Compute Dynamic Gap
# ============================================================
# Gap = longitudinal distance between V0 and V3
gap_v0_v3 = v0_x - v3_x  # positive means V0 is ahead

# Gap threshold for overtaking
GAP_THRESHOLD = 30.0  # meters (minimum gap for safe overtake)

# ============================================================
# 4. Save to CSV
# ============================================================
csv_path = "dynamic_overtaking_scenario.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'vehicle_id', 'x', 'y', 'velocity', 'heading'])
    for i in range(N):
        writer.writerow([f'{t[i]:.1f}', 'V0', f'{v0_x[i]:.2f}', f'{v0_y[i]:.2f}',
                       f'{v0_speed[i]:.2f}', f'{v0_heading[i]:.4f}'])
        writer.writerow([f'{t[i]:.1f}', 'V3', f'{v3_x[i]:.2f}', f'{v3_y[i]:.2f}',
                       f'{v3_speed[i]:.2f}', f'{v3_heading[i]:.4f}'])
        writer.writerow([f'{t[i]:.1f}', 'V6', f'{v6_x[i]:.2f}', f'{v6_y[i]:.2f}',
                       f'{v6_speed[i]:.2f}', f'{v6_heading[i]:.4f}'])

print(f"Scenario data saved to {csv_path}")
print(f"Total rows: {N * 3}, Time steps: {N}, Duration: {T_TOTAL}s")

# ============================================================
# 5. Visualization
# ============================================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('white')

# ---- Panel 1: Bird's-eye view (full scenario) ----
ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.40])

# Draw lanes
for j, ly in enumerate(LANE_CENTERS):
    ax1.axhline(y=ly, color='#CCCCCC', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.axhline(y=ROAD_Y_MIN, color='#888888', linewidth=1.5)
ax1.axhline(y=ROAD_Y_MAX, color='#888888', linewidth=1.5)

# Phase backgrounds
ax1.axvspan(v6_x[0], v6_x[30], alpha=0.06, color='green', label='Phase 1: Gap Opens')
ax1.axvspan(v6_x[30], v6_x[60], alpha=0.06, color='red', label='Phase 2: Gap Closes')
ax1.axvspan(v6_x[60], v6_x[-1], alpha=0.06, color='green', label='Phase 3: Overtake')

# Plot trajectories
ax1.plot(v0_x, v0_y, color='#E74C3C', linewidth=2, label='V0 (actual)', zorder=3)
ax1.plot(v3_x, v3_y, color='#3498DB', linewidth=2, label='V3 (actual)', zorder=3)
ax1.plot(v6_x, v6_y, color='#D4A017', linewidth=2.5, label='V6 (planned)', zorder=4)

# Draw vehicles at key timestamps
def draw_car(ax, x, y, color, label=None, angle=0):
    rect = mpatches.FancyBboxPatch(
        (x - CAR_LENGTH/2, y - CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor='black',
        linewidth=0.5, alpha=0.8, zorder=5
    )
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white', zorder=6)

# Draw cars at t=0, 3, 6, 10
key_times = [0, 30, 60, -1]
for idx in key_times:
    draw_car(ax1, v0_x[idx], v0_y[idx], '#E74C3C', 'V0')
    draw_car(ax1, v3_x[idx], v3_y[idx], '#3498DB', 'V3')
    draw_car(ax1, v6_x[idx], v6_y[idx], '#D4A017', 'V6')

# Mark overtaking points
ax1.plot(v6_x[15], v6_y[15], 'D', color='#D4A017', markersize=10, zorder=6)
ax1.annotate('Start Overtake', (v6_x[15], v6_y[15]),
            textcoords="offset points", xytext=(10, 15), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#D4A017'))

ax1.plot(v6_x[35], v6_y[35], 'X', color='#E74C3C', markersize=10, zorder=6)
ax1.annotate('Abort!', (v6_x[35], v6_y[35]),
            textcoords="offset points", xytext=(10, -15), fontsize=8,
            color='#E74C3C', arrowprops=dict(arrowstyle='->', color='#E74C3C'))

ax1.plot(v6_x[75], v6_y[75], 'D', color='#27AE60', markersize=10, zorder=6)
ax1.annotate('Overtake Success', (v6_x[75], v6_y[75]),
            textcoords="offset points", xytext=(10, 15), fontsize=8,
            color='#27AE60', arrowprops=dict(arrowstyle='->', color='#27AE60'))

ax1.set_xlabel('Longitudinal Position X (m)', fontsize=11)
ax1.set_ylabel('Lateral Position Y (m)', fontsize=11)
ax1.set_title('Bird\'s-Eye View — Three-Phase Dynamic Overtaking Scenario', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax1.set_ylim([1, 17])
ax1.set_aspect('auto')
ax1.grid(True, alpha=0.2)

# ---- Panel 2: Gap dynamics ----
ax2 = fig.add_axes([0.52, 0.55, 0.42, 0.40])

ax2.fill_between(t, 0, gap_v0_v3, where=(gap_v0_v3 >= GAP_THRESHOLD),
                 alpha=0.15, color='green', label='Safe to overtake')
ax2.fill_between(t, 0, gap_v0_v3, where=(gap_v0_v3 < GAP_THRESHOLD),
                 alpha=0.15, color='red', label='Unsafe')

ax2.plot(t, gap_v0_v3, color='#004B8D', linewidth=2.5, label='Gap V0-V3')
ax2.axhline(y=GAP_THRESHOLD, color='#E74C3C', linewidth=1.5, linestyle='--',
            label=f'Threshold ({GAP_THRESHOLD}m)')

# Mark phases
ax2.axvspan(0, 3, alpha=0.05, color='green')
ax2.axvspan(3, 6, alpha=0.05, color='red')
ax2.axvspan(6, 10, alpha=0.05, color='green')

# Annotations
ax2.annotate('Phase 1: Gap Opens\n→ Start Overtake', xy=(1.5, gap_v0_v3[15]),
            fontsize=9, ha='center', color='#27AE60', fontweight='bold')
ax2.annotate('Phase 2: Gap Closes\n→ Abort!', xy=(4.5, gap_v0_v3[45]),
            fontsize=9, ha='center', color='#E74C3C', fontweight='bold')
ax2.annotate('Phase 3: Gap Opens\n→ Overtake Success!', xy=(8, gap_v0_v3[80]),
            fontsize=9, ha='center', color='#27AE60', fontweight='bold')

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Longitudinal Gap V0-V3 (m)', fontsize=11)
ax2.set_title('Dynamic Gap Between V0 and V3', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# ---- Panel 3: Speed comparison ----
ax3 = fig.add_axes([0.05, 0.08, 0.28, 0.38])

ax3.plot(t, v0_speed, color='#E74C3C', linewidth=2, label='V0 (slow lead)')
ax3.plot(t, v3_speed, color='#3498DB', linewidth=2, label='V3 (reference)')
ax3.plot(t, v6_speed, color='#D4A017', linewidth=2, label='V6 (ego)')

ax3.axvspan(0, 3, alpha=0.04, color='green')
ax3.axvspan(3, 6, alpha=0.04, color='red')
ax3.axvspan(6, 10, alpha=0.04, color='green')

ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Speed (m/s)', fontsize=10)
ax3.set_title('Vehicle Speeds', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ---- Panel 4: V6 lateral position (lane changes) ----
ax4 = fig.add_axes([0.38, 0.08, 0.28, 0.38])

ax4.plot(t, v6_y, color='#D4A017', linewidth=2.5, label='V6 Y position')
for ly in LANE_CENTERS:
    ax4.axhline(y=ly, color='#CCCCCC', linewidth=0.8, linestyle='--', alpha=0.5)

ax4.axhspan(LANE_CENTERS[1]-1, LANE_CENTERS[1]+1, alpha=0.05, color='blue')
ax4.axhspan(LANE_CENTERS[2]-1, LANE_CENTERS[2]+1, alpha=0.05, color='red')

# Phase annotations
ax4.axvspan(0, 3, alpha=0.04, color='green')
ax4.axvspan(3, 6, alpha=0.04, color='red')
ax4.axvspan(6, 10, alpha=0.04, color='green')

ax4.annotate('→ Lane 3', xy=(1.5, 10), fontsize=8, color='#D4A017', fontweight='bold')
ax4.annotate('← Lane 2', xy=(4.5, 8), fontsize=8, color='#E74C3C', fontweight='bold')
ax4.annotate('→ Lane 3 → Lane 2', xy=(7.5, 10), fontsize=8, color='#27AE60', fontweight='bold')

ax4.set_xlabel('Time (s)', fontsize=10)
ax4.set_ylabel('Lateral Y (m)', fontsize=10)
ax4.set_title('V6 Lane Changes (3-Phase)', fontsize=12, fontweight='bold')
ax4.set_ylim([4, 14])
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)

# ---- Panel 5: Gap + V6 action timeline ----
ax5 = fig.add_axes([0.71, 0.08, 0.25, 0.38])

# Action timeline
actions = np.zeros(N)
for i, ti in enumerate(t):
    if ti < 0.5:
        actions[i] = 0  # hold
    elif ti < 3.0:
        actions[i] = 1  # start overtake
    elif ti < 3.5:
        actions[i] = 2  # at Lane 3
    elif ti < 6.0:
        actions[i] = 3  # abort
    elif ti < 6.5:
        actions[i] = 0  # hold
    elif ti < 8.5:
        actions[i] = 1  # overtake
    elif ti < 9.0:
        actions[i] = 2  # at Lane 3
    else:
        actions[i] = 4  # return

colors_map = {0: '#888888', 1: '#F39C12', 2: '#27AE60', 3: '#E74C3C', 4: '#3498DB'}
for i in range(N-1):
    ax5.barh(y=0, width=DT, left=t[i], height=0.5,
             color=colors_map[int(actions[i])], alpha=0.8)

ax5_twin = ax5.twinx()
ax5_twin.plot(t, gap_v0_v3, color='#004B8D', linewidth=2)
ax5_twin.axhline(y=GAP_THRESHOLD, color='#E74C3C', linestyle='--', linewidth=1)

ax5.set_yticks([])
ax5.set_xlabel('Time (s)', fontsize=10)
ax5.set_title('Action Timeline & Gap', fontsize=12, fontweight='bold')
ax5_twin.set_ylabel('Gap (m)', fontsize=10, color='#004B8D')

# Legend for actions
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#888888', label='Hold'),
    Patch(facecolor='#F39C12', label='Start Overtake'),
    Patch(facecolor='#27AE60', label='At Lane 3'),
    Patch(facecolor='#E74C3C', label='Abort'),
    Patch(facecolor='#3498DB', label='Return'),
]
ax5.legend(handles=legend_elements, fontsize=7, loc='upper left', ncol=2)

plt.savefig('dynamic_scenario_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n=== Scenario Summary ===")
print(f"Phase 1 (t=0~3s):   Gap = {gap_v0_v3[0]:.1f}m → {gap_v0_v3[30]:.1f}m  (opens)")
print(f"Phase 2 (t=3~6s):   Gap = {gap_v0_v3[30]:.1f}m → {gap_v0_v3[60]:.1f}m  (closes)")
print(f"Phase 3 (t=6~10s):  Gap = {gap_v0_v3[60]:.1f}m → {gap_v0_v3[-1]:.1f}m  (opens)")
print(f"\nV0 avg speed: {np.mean(v0_speed):.1f} m/s")
print(f"V3 avg speed: {np.mean(v3_speed):.1f} m/s")
print(f"V6 avg speed: {np.mean(v6_speed):.1f} m/s")
print(f"\nCSV saved: {csv_path} ({N*3} rows)")
print(f"Figure saved: dynamic_scenario_overview.png")
