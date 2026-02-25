import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

def generate_star_points(n_points=3000, n_arms=8, inner_r=0.6, outer_r=1.5):
    """Generates an 8-pointed star point cloud using rejection sampling."""
    points = []
    while len(points) < n_points:
        # Sample points in a bounding box
        x, y = np.random.uniform(-outer_r, outer_r, 2)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Star boundary: r(theta) = base + amplitude * cos(n * theta)
        # Using a squared cosine creates sharper 'points' like the image
        r_limit = inner_r + (outer_r - inner_r) * (np.abs(np.cos(n_arms * theta / 2)))**4
        
        if r < r_limit:
            # Add some jitter/noise for the 'cloud' effect
            points.append([x + np.random.normal(0, 0.05), y + np.random.normal(0, 0.05)])
    return np.array(points)

def generate_circle_points(n_points=3000, radius=1.2):
    """Generates a dense circular point cloud."""
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def base_density_pde(radius=1.2, grid_res=80, sigma_smooth=1.5):
    """Smooth circular (base) density in PDE form: 2D Gaussian-like blob."""
    x = np.linspace(-2, 2, grid_res)
    y = np.linspace(-2, 2, grid_res)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    sigma = radius / 2
    density = np.exp(-0.5 * (R / sigma) ** 2)
    density = gaussian_filter(density, sigma=sigma_smooth)
    return X, Y, density


def star_density_pde(inner_r=0.6, outer_r=1.5, n_arms=8, grid_res=80, sigma_smooth=1.8):
    """Smooth star-shaped (target) density in PDE form: indicator inside star, then smoothed."""
    x = np.linspace(-2, 2, grid_res)
    y = np.linspace(-2, 2, grid_res)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    r_limit = inner_r + (outer_r - inner_r) * (np.abs(np.cos(n_arms * Theta / 2))) ** 4
    density = np.where(R < r_limit, 1.0, 0.0)
    density = gaussian_filter(density, sigma=sigma_smooth)
    return X, Y, density


# --- Configuration ---
n_points = 2500
n_flow_lines = 28   # fewer lines so each reads as an individual trajectory
rainbow = plt.cm.rainbow


def make_smooth_ribbon(ax, y_start, y_end, n_pts=200, n_lines=None, spread_start=0.35, spread_end=0.9, seed=42):
    """Non-crossing, non-parallel: each line has its own start/end (ordered so they never cross)."""
    n_lines = n_lines or n_flow_lines
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_pts)
    # Same order at base and target so lines never cross; different spreads => fan out (not parallel)
    start_x = np.linspace(-spread_start, spread_start, n_lines)
    end_x = np.linspace(-spread_end, spread_end, n_lines)
    bow_strength = 0.18
    y_base = y_start + t * (y_end - y_start)
    x_center = np.zeros(n_pts)
    y_center = np.zeros(n_pts)
    for i in range(n_lines):
        x_linear = (1 - t) * start_x[i] + t * end_x[i]
        x_bow = bow_strength * np.sin(np.pi * t)
        x_path = x_linear + x_bow
        # Tiny smooth y-variation per line (same bow for all so no crossing)
        y_path = y_base + 0.006 * np.sin(2 * np.pi * t + 0.1 * i)
        points = np.column_stack([x_path, y_path])
        segments = np.concatenate([points[:-1], points[1:]], axis=1).reshape(-1, 2, 2)
        color_param = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, array=color_param, cmap=rainbow, linewidths=0.7, alpha=0.78)
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        if i == n_lines // 2:
            x_center, y_center = x_path.copy(), y_path.copy()
    return x_center, y_center, t


def get_smooth_center_line(y_start, y_end, n_pts=200, n_lines=None, spread_start=0.35, spread_end=0.9, seed=42):
    """Return (x_center, y_center, t) for the smooth center line only (no drawing). Same path as bottom arrow in fig2."""
    n_lines = n_lines or n_flow_lines
    t = np.linspace(0, 1, n_pts)
    start_x = np.linspace(-spread_start, spread_start, n_lines)
    end_x = np.linspace(-spread_end, spread_end, n_lines)
    bow_strength = 0.18
    y_base = y_start + t * (y_end - y_start)
    i = n_lines // 2
    x_linear = (1 - t) * start_x[i] + t * end_x[i]
    x_bow = bow_strength * np.sin(np.pi * t)
    x_center = x_linear + x_bow
    y_center = y_base + 0.006 * np.sin(2 * np.pi * t + 0.1 * i)
    return x_center, y_center, t


def make_zigzag_ribbon(ax, y_start, y_end, n_lines=None, n_segments=8, spread_start=0.35, spread_end=0.9, zigzag_amp=0.25, seed=42):
    """Non-crossing zig-zag (piecewise linear) lines from y_start to y_end; rainbow along flow."""
    n_lines = n_lines or n_flow_lines
    rng = np.random.default_rng(seed)
    start_x = np.linspace(-spread_start, spread_start, n_lines)
    end_x = np.linspace(-spread_end, spread_end, n_lines)
    n_way = n_segments + 1
    t_way = np.linspace(0, 1, n_way)
    y_way = y_start + t_way * (y_end - y_start)
    x_center = np.zeros(n_way)
    y_center = y_way.copy()
    for i in range(n_lines):
        x_linear = (1 - t_way) * start_x[i] + t_way * end_x[i]
        # Zigzag: alternate +zigzag_amp and -zigzag_amp at inner waypoints
        zig = np.zeros(n_way)
        zig[1:-1] = zigzag_amp * np.array([1 if k % 2 == 0 else -1 for k in range(1, n_way - 1)])
        x_path = x_linear + zig
        points = np.column_stack([x_path, y_way])
        segments = np.concatenate([points[:-1], points[1:]], axis=1).reshape(-1, 2, 2)
        color_param = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, array=color_param, cmap=rainbow, linewidths=0.7, alpha=0.78)
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        if i == n_lines // 2:
            x_center = x_path.copy()
    return x_center, y_center, t_way


# 1. Generate clouds (for any legacy use) and PDE density grids
star_cloud = generate_star_points(n_points)
circle_cloud = generate_circle_points(n_points)
X_base, Y_base, density_base = base_density_pde()
X_star, Y_star, density_star = star_density_pde()

# 2. Set up the figure (shorter lines than original, but distributions fully visible)
fig, ax = plt.subplots(figsize=(5, 6.5))
ax.set_facecolor('white')

Y_STAR = 6
Y_MIDDLE = 3.0
Y_CIRCLE = 0
y_start, y_end = Y_CIRCLE + 1.0, Y_STAR - 1.0

# --- Plot Top Star (point cloud) ---
ax.scatter(star_cloud[:, 0], star_cloud[:, 1] + Y_STAR, s=1, color='skyblue', alpha=0.6)

# --- Plot Bottom Circle (point cloud), purple ---
ax.scatter(circle_cloud[:, 0], circle_cloud[:, 1] + Y_CIRCLE, s=1, color='#BB8FCE', alpha=0.6)

# --- Dense ribbon: continuous from base to target ---
x_center, y_center, t_curve = make_smooth_ribbon(ax, y_start, y_end)
purple_base = '#6A0DAD'
for x_pos in np.linspace(-0.3, 0.3, 5):
    ax.arrow(x_pos, Y_MIDDLE, 0, 0.28, head_width=0.06, color=purple_base, alpha=0.85)

# Outside arrow: same as bottom arrow in second figure (smooth center line, 0% -> 12%)
arrow_offset_base = -0.8
n_pts_1 = len(x_center)
ax.annotate('', xy=(x_center[int(0.12 * n_pts_1)] + arrow_offset_base, y_center[int(0.12 * n_pts_1)]),
            xytext=(x_center[0] + arrow_offset_base, y_center[0]),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.8, alpha=0.9, mutation_scale=18))

ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.25, Y_STAR + 1.35)
ax.axis('off')

plt.tight_layout()
plt.savefig('flow_matching_illustration.pdf')

# ========== Second figure: base and target only ==========
fig2, ax2 = plt.subplots(figsize=(5, 6.5))
ax2.set_facecolor('white')

Y_STAR_2 = 6
Y_CIRCLE_2 = 0
y_start_2, y_end_2 = Y_CIRCLE_2 + 1.0, Y_STAR_2 - 1.0

ax2.scatter(star_cloud[:, 0], star_cloud[:, 1] + Y_STAR_2, s=1, color='skyblue', alpha=0.6)
ax2.scatter(circle_cloud[:, 0], circle_cloud[:, 1] + Y_CIRCLE_2, s=1, color='#BB8FCE', alpha=0.6)

x_c2, y_c2, _ = make_smooth_ribbon(ax2, y_start_2, y_end_2, seed=42)
n_pts = len(x_c2)
arrow_offset_2 = -1.15
arrow_offset_base = -0.8
ax2.annotate('', xy=(x_c2[int(0.12 * n_pts)] + arrow_offset_base, y_c2[int(0.12 * n_pts)]),
             xytext=(x_c2[0] + arrow_offset_base, y_c2[0]),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.8, alpha=0.9, mutation_scale=18))
ax2.annotate('', xy=(x_c2[int(0.88 * n_pts)] + arrow_offset_2, y_c2[int(0.88 * n_pts)]),
             xytext=(x_c2[-1] + arrow_offset_2, y_c2[-1]),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.8, alpha=0.9, mutation_scale=18))

ax2.set_xlim(-1.65, 1.65)
ax2.set_ylim(-1.25, Y_STAR_2 + 1.35)
ax2.axis('off')
plt.tight_layout()
plt.savefig('continuous_flow_illustration.pdf')

# ========== Third figure: zig-zag (residual) flow lines ==========
fig3, ax3 = plt.subplots(figsize=(5, 6.5))
ax3.set_facecolor('white')
Y_STAR_3 = 6
Y_CIRCLE_3 = 0
y_start_3, y_end_3 = Y_CIRCLE_3 + 1.0, Y_STAR_3 - 1.0

ax3.scatter(star_cloud[:, 0], star_cloud[:, 1] + Y_STAR_3, s=1, color='skyblue', alpha=0.6)
ax3.scatter(circle_cloud[:, 0], circle_cloud[:, 1] + Y_CIRCLE_3, s=1, color='#BB8FCE', alpha=0.6)

x_c3, y_c3, t_way3 = make_zigzag_ribbon(
    ax3, y_start_3, y_end_3, n_segments=8, zigzag_amp=0.25, seed=42
)
n_w = len(x_c3)
# Bottom arrow: same as in second figure (smooth center line, same position and direction)
x_smooth, y_smooth, _ = get_smooth_center_line(y_start_3, y_end_3, seed=42)
n_s = len(x_smooth)
ax3.annotate('', xy=(x_smooth[int(0.12 * n_s)] + arrow_offset_base, y_smooth[int(0.12 * n_s)]),
             xytext=(x_smooth[0] + arrow_offset_base, y_smooth[0]),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.8, alpha=0.9, mutation_scale=18))
ax3.annotate('', xy=(x_c3[int(0.88 * n_w)] + arrow_offset_2, y_c3[int(0.88 * n_w)]),
             xytext=(x_c3[-1] + arrow_offset_2, y_c3[-1]),
             arrowprops=dict(arrowstyle='->', color='k', lw=1.8, alpha=0.9, mutation_scale=18))

ax3.set_xlim(-1.65, 1.65)
ax3.set_ylim(-1.25, Y_STAR_3 + 1.35)
ax3.axis('off')
plt.tight_layout()
plt.savefig('residual_flow_illustration.pdf')