"""
3D Overtaking Scenario - Two-Panel: Before & During Overtaking
===============================================================
Left:  Before - V0 and V3 block V6, confidence bands overlap (no path)
Right: During - V6 passes between V0 and V3, trajectory sandwiched by bands
White background, only middle two lanes (L2, L3)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection

OUTPUT_DIR = '.'

# ============================================================
# Isometric projection
# ============================================================
A = np.radians(25)
B = np.radians(35)
COS_A, SIN_A = np.cos(A), np.sin(A)
COS_B, SIN_B = np.cos(B), np.sin(B)

def project(x, y, z=0):
    sx = x * COS_A - y * SIN_A
    sy = x * SIN_A + y * COS_A * COS_B + z * SIN_B
    return sx, sy

def project_arr(pts3d):
    pts = np.array(pts3d)
    sx = pts[:,0] * COS_A - pts[:,1] * SIN_A
    sy = pts[:,0] * SIN_A + pts[:,1] * COS_A * COS_B + pts[:,2] * SIN_B
    return list(zip(sx, sy))

# ============================================================
# Colors
# ============================================================
C_V0 = '#FF4444'; C_V0_DARK = '#CC2222'
C_V3 = '#3399FF'; C_V3_DARK = '#2266CC'
C_V6 = '#F4C430'; C_V6_DARK = '#D4A017'
C_PATH = '#E6A800'      # Yellow for safe corridor and planned path
C_ROAD = '#e8e8e8'       # Light gray road for white background
C_EDGE = '#888888'       # Gray edge lines
C_WINDOW = '#88ccee'
C_TIRE = '#1a1a1a'
C_TEXT = '#222222'       # Dark text for white background

# ============================================================
# Road parameters - only 2 lanes (L2, L3)
# ============================================================
road_len = 90
lw = 5.5
rw = 2 * lw             # Only 2 lanes
y_min = -rw / 2
y_max = rw / 2
lane_y = [y_min + 0.5*lw, y_min + 1.5*lw]  # L2, L3

# ============================================================
# Draw Road
# ============================================================
def draw_road(ax):
    # Surface
    road_corners = [(0, y_min, 0), (road_len, y_min, 0), (road_len, y_max, 0), (0, y_max, 0)]
    ax.add_patch(Polygon(project_arr(road_corners), closed=True, facecolor=C_ROAD, edgecolor='none', zorder=1))
    # Lane divider (single, between L2 and L3)
    yl = y_min + lw
    for s in range(0, road_len, 7):
        e = min(s + 3.5, road_len)
        p1 = project(s, yl, 0.02); p2 = project(e, yl, 0.02)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', lw=2.0, alpha=0.8, zorder=2, solid_capstyle='round')
    # Edges
    for ye in [y_min, y_max]:
        p1 = project(0, ye, 0.03); p2 = project(road_len, ye, 0.03)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_EDGE, lw=2.5, alpha=0.85, zorder=2, solid_capstyle='round')
    # Lane labels
    for i, lab in enumerate(['L2', 'L3']):
        yc = y_min + (i + 0.5) * lw
        p = project(5, yc, 0.05)
        ax.text(p[0], p[1], lab, fontsize=13, color='#999', fontweight='bold', ha='center', va='center', zorder=3)

# ============================================================
# Draw Car
# ============================================================
def draw_car_iso(ax, cx, cy, heading, color, color_dark, label, label_dy=3.5):
    ch, sh = np.cos(heading), np.sin(heading)
    def rot(x, y): return x * ch - y * sh, x * sh + y * ch
    L, W, H = 5.0, 2.0, 1.6

    # Shadow
    shadow_pts = []
    for dx, dy in [(-L/2-0.3, -W/2-0.3), (L/2+0.3, -W/2-0.3), (L/2+0.3, W/2+0.3), (-L/2-0.3, W/2+0.3)]:
        rx, ry = rot(dx, dy)
        shadow_pts.append(project(cx + rx, cy + ry, 0.02))
    ax.add_patch(Polygon(shadow_pts, closed=True, facecolor='black', alpha=0.15, edgecolor='none', zorder=3))

    # Body
    body_h = H * 0.55
    corners = {}
    for dx in [-L/2, L/2]:
        for dy in [-W/2, W/2]:
            for dz in [0, body_h]:
                rx, ry = rot(dx, dy)
                corners[(dx, dy, dz)] = project(cx + rx, cy + ry, dz + 0.05)

    for face_pts, fc, z in [
        ([(-L/2,-W/2,0),(L/2,-W/2,0),(L/2,-W/2,body_h),(-L/2,-W/2,body_h)], color_dark, 4),
        ([(L/2,-W/2,0),(L/2,W/2,0),(L/2,W/2,body_h),(L/2,-W/2,body_h)], color_dark, 5),
        ([(-L/2,W/2,0),(L/2,W/2,0),(L/2,W/2,body_h),(-L/2,W/2,body_h)], color, 5),
        ([(-L/2,-W/2,body_h),(L/2,-W/2,body_h),(L/2,W/2,body_h),(-L/2,W/2,body_h)], color, 6),
    ]:
        face = [corners[p] for p in face_pts]
        ax.add_patch(Polygon(face, closed=True, facecolor=fc, edgecolor='#111', lw=0.5, zorder=z))

    # Cabin
    cL, cW = L * 0.5, W * 0.78
    cabin_h = H; cabin_base = body_h
    cc = {}
    for dx in [-cL/2, cL/2]:
        for dy in [-cW/2, cW/2]:
            for dz in [cabin_base, cabin_h]:
                rx, ry = rot(dx, dy)
                cc[(dx, dy, dz)] = project(cx + rx, cy + ry, dz + 0.05)

    for face_pts, fc, z, a in [
        ([(-cL/2,-cW/2,cabin_base),(cL/2,-cW/2,cabin_base),(cL/2,-cW/2,cabin_h),(-cL/2,-cW/2,cabin_h)], '#222', 6, 1),
        ([(cL/2,-cW/2,cabin_base),(cL/2,cW/2,cabin_base),(cL/2,cW/2,cabin_h),(cL/2,-cW/2,cabin_h)], C_WINDOW, 7, 0.7),
        ([(-cL/2,cW/2,cabin_base),(cL/2,cW/2,cabin_base),(cL/2,cW/2,cabin_h),(-cL/2,cW/2,cabin_h)], '#333', 7, 1),
        ([(-cL/2,-cW/2,cabin_h),(cL/2,-cW/2,cabin_h),(cL/2,cW/2,cabin_h),(-cL/2,cW/2,cabin_h)], '#444', 8, 1),
        ([(-cL/2,-cW/2,cabin_base),(-cL/2,cW/2,cabin_base),(-cL/2,cW/2,cabin_h),(-cL/2,-cW/2,cabin_h)], C_WINDOW, 6, 0.5),
    ]:
        face = [cc[p] for p in face_pts]
        ax.add_patch(Polygon(face, closed=True, facecolor=fc, edgecolor='#111', lw=0.4, alpha=a, zorder=z))

    # Wheels
    wheel_r = 0.35
    for wx, wy in [(-L/2.5, -W/2), (-L/2.5, W/2), (L/2.5, -W/2), (L/2.5, W/2)]:
        rx, ry = rot(wx, wy)
        wp = project(cx + rx, cy + ry, 0.05)
        ax.add_patch(plt.Circle(wp, wheel_r, facecolor=C_TIRE, edgecolor='#444', lw=0.5, zorder=4))

    # Headlights
    for hy in [-W/4, W/4]:
        rx, ry = rot(L/2, hy)
        hp = project(cx + rx, cy + ry, body_h * 0.5 + 0.05)
        ax.scatter([hp[0]], [hp[1]], color='#ffeb3b', s=25, zorder=8, edgecolor='none')

    # Label
    lp = project(cx, cy + label_dy, H + 0.8)
    ax.text(lp[0], lp[1], label, fontsize=13, color=color, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color, lw=1.5), zorder=10)

    # Direction arrow
    arrow_start = project(cx, cy, H + 0.3)
    ax_end = project(cx + 5 * ch, cy + 5 * sh, H + 0.3)
    ax.annotate('', xy=ax_end, xytext=arrow_start, arrowprops=dict(arrowstyle='->', color=color, lw=2.5), zorder=10)

# ============================================================
# Draw Confidence Band
# ============================================================
def draw_band_iso(ax, xs, yc, width, z, color, alpha=0.25):
    yu = yc + width; yl = yc - width
    verts = []
    for i in range(len(xs) - 1):
        quad = [project(xs[i], yu[i], z), project(xs[i+1], yu[i+1], z),
                project(xs[i+1], yl[i+1], z), project(xs[i], yl[i], z)]
        verts.append(quad)
    ax.add_collection(PolyCollection(verts, facecolor=color, alpha=alpha, edgecolor='none', zorder=2.5))
    pu = [project(xs[i], yu[i], z) for i in range(len(xs))]
    pl = [project(xs[i], yl[i], z) for i in range(len(xs))]
    ax.plot([p[0] for p in pu], [p[1] for p in pu], color=color, lw=1.2, alpha=0.6, zorder=2.5)
    ax.plot([p[0] for p in pl], [p[1] for p in pl], color=color, lw=1.2, alpha=0.6, zorder=2.5)

# ============================================================
# Draw "No Path" marker (red X)
# ============================================================
def draw_no_path(ax, x, y, z=0.5):
    size = 2.0
    p1 = project(x - size, y - size, z)
    p2 = project(x + size, y + size, z)
    p3 = project(x - size, y + size, z)
    p4 = project(x + size, y - size, z)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#FF4444', lw=4, alpha=0.9, zorder=5)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color='#FF4444', lw=4, alpha=0.9, zorder=5)
    lp = project(x, y, z + 0.8)
    ax.text(lp[0], lp[1], 'No Path', fontsize=14, color='#FF4444', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#FF4444', lw=1.5), zorder=6)

# ============================================================
# Set axis limits helper
# ============================================================
def set_axis_limits(ax):
    all_pts = []
    for xw in [0, road_len]:
        for yw in [y_min, y_max]:
            for zw in [0, 2.5]:
                all_pts.append(project(xw, yw, zw))
    all_pts = np.array(all_pts)
    margin = 2
    ax.set_xlim(all_pts[:,0].min() - margin, all_pts[:,0].max() + margin)
    ax.set_ylim(all_pts[:,1].min() - margin, all_pts[:,1].max() + margin)

# ============================================================
# Figure: Two panels - white background
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 12))
fig.set_facecolor('white')

for ax in [ax1, ax2]:
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')
    draw_road(ax)

mid_y = (lane_y[0] + lane_y[1]) / 2

# ============================================================
# LEFT PANEL: Before Overtaking - bands overlap, no path
# ============================================================
# V3 (Blue) and V0 (Red) close together ahead, V6 behind
# Bring V3 and V0 closer in Y so their (same-width) bands overlap
v3_y_left = mid_y - 1.6   # V3 below center
v0_y_left = mid_y + 1.6   # V0 above center
draw_car_iso(ax1, 55, v3_y_left, -0.02, C_V3, C_V3_DARK, 'V3  Reference', label_dy=-4.0)
draw_car_iso(ax1, 60, v0_y_left, 0.03, C_V0, C_V0_DARK, 'V0  Slow Lead', label_dy=4.0)
draw_car_iso(ax1, 25, mid_y, 0.0, C_V6, C_V6_DARK, 'V6  Blocked', label_dy=-4.0)

# Same band widths as right panel - they overlap because cars are close
xb1 = np.linspace(15, 78, 60)
v0y_1 = v0_y_left + 0.2 * np.sin(xb1 / 10)
v0w_1 = 1.2 + 0.15 * np.sin(xb1 / 8)   # Same width as right panel
draw_band_iso(ax1, xb1, v0y_1, v0w_1, 0.15, C_V0, alpha=0.28)

v3y_1 = v3_y_left + 0.2 * np.sin(xb1 / 9)
v3w_1 = 1.0 + 0.12 * np.cos(xb1 / 7)   # Same width as right panel
draw_band_iso(ax1, xb1, v3y_1, v3w_1, 0.15, C_V3, alpha=0.28)

# "No Path" marker where bands overlap
draw_no_path(ax1, 50, mid_y, 0.5)

ax1.text(0.5, 0.96, 'Before Overtaking', transform=ax1.transAxes,
         fontsize=20, color=C_TEXT, fontweight='bold', ha='center')
ax1.text(0.5, 0.93, 'Confidence bands overlap - no safe corridor available',
         transform=ax1.transAxes, fontsize=14, color='#CC4444', ha='center')

set_axis_limits(ax1)

# ============================================================
# RIGHT PANEL: During Overtaking - V6 passes between bands
# ============================================================
# V3 and V0 further apart in Y so their bands don't overlap - gap = safe corridor
v3_y_right = lane_y[0]
v0_y_right = lane_y[1]
draw_car_iso(ax2, 44, v3_y_right, -0.02, C_V3, C_V3_DARK, 'V3  Reference', label_dy=-4.0)
draw_car_iso(ax2, 56, v0_y_right, 0.03, C_V0, C_V0_DARK, 'V0  Slow Lead', label_dy=4.0)
draw_car_iso(ax2, 50, mid_y, 0.15, C_V6, C_V6_DARK, 'V6  Overtaking', label_dy=-4.0)

# Same band widths as left panel - gap exists because cars are far apart
xb2 = np.linspace(15, 78, 60)
v0y_2 = v0_y_right + 0.2 * np.sin(xb2 / 10)
v0w_2 = 1.2 + 0.15 * np.sin(xb2 / 8)
draw_band_iso(ax2, xb2, v0y_2, v0w_2, 0.15, C_V0, alpha=0.25)

v3y_2 = v3_y_right + 0.2 * np.sin(xb2 / 9)
v3w_2 = 1.0 + 0.12 * np.cos(xb2 / 7)
draw_band_iso(ax2, xb2, v3y_2, v3w_2, 0.15, C_V3, alpha=0.25)

# Safe corridor
safe_verts = []
for i in range(len(xb2) - 1):
    v0_lower = v0y_2[i] - v0w_2[i]
    v0_lower_next = v0y_2[i+1] - v0w_2[i+1]
    v3_upper = v3y_2[i] + v3w_2[i]
    v3_upper_next = v3y_2[i+1] + v3w_2[i+1]
    if v0_lower > v3_upper + 0.3:
        quad = [project(xb2[i], v3_upper, 0.15),
                project(xb2[i+1], v3_upper_next, 0.15),
                project(xb2[i+1], v0_lower_next, 0.15),
                project(xb2[i], v0_lower, 0.15)]
        safe_verts.append(quad)
if safe_verts:
    ax2.add_collection(PolyCollection(safe_verts, facecolor=C_PATH, alpha=0.15, edgecolor='none', zorder=2.3))

# Overtaking trajectory
xt = np.linspace(18, 75, 50)
yt = np.full_like(xt, mid_y)
traj_pts = [project(xt[i], yt[i], 0.3) for i in range(len(xt))]
ax2.plot([p[0] for p in traj_pts], [p[1] for p in traj_pts], '--', color=C_PATH, lw=3, alpha=0.9, zorder=3.5)

# Arrow
mid_t = len(xt) // 2
p_start = project(xt[mid_t], yt[mid_t], 0.5)
p_end = project(xt[mid_t + 3], yt[mid_t + 3], 0.5)
ax2.annotate('', xy=p_end, xytext=p_start,
             arrowprops=dict(arrowstyle='->', color=C_PATH, lw=2.5), zorder=3.5)

ax2.text(0.5, 0.96, 'During Overtaking', transform=ax2.transAxes,
         fontsize=20, color=C_TEXT, fontweight='bold', ha='center')
ax2.text(0.5, 0.93, 'V6 plans safe trajectory between V0 and V3 confidence bands',
         transform=ax2.transAxes, fontsize=14, color='#B8860B', ha='center')

set_axis_limits(ax2)

# ============================================================
# Legend (shared, on right panel)
# ============================================================
legend = [
    mpatches.Patch(facecolor=C_V0, alpha=0.4, label='V0 Confidence Band'),
    mpatches.Patch(facecolor=C_V3, alpha=0.4, label='V3 Confidence Band'),
    mpatches.Patch(facecolor=C_PATH, alpha=0.25, label='Safe Corridor'),
    plt.Line2D([0], [0], color=C_PATH, ls='--', lw=2.5, label='Planned Path'),
]
ax2.legend(handles=legend, loc='upper left', fontsize=12, facecolor='white',
          edgecolor='#888', labelcolor=C_TEXT, framealpha=0.9)

# Super title
fig.suptitle('BPINN-Based Autonomous Overtaking System', fontsize=24, color=C_TEXT,
             fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/overtaking_3d_diagram.png', dpi=200, facecolor='white',
           edgecolor='none', bbox_inches='tight')
print('Saved!')
plt.show()
