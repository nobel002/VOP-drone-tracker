import numpy as np
import matplotlib

try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Aesthetics: Dark Background
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2, left=0.1)

# Lock the rotation for that "instrument" feel
ax.view_init(elev=20, azim=45)
# Note: In some versions, ax.mouse_init() might not exist;
# we simply don't provide a way to rotate via mouse by setting:
ax.disable_mouse_rotation()

# 2. Draw a Subtle Wireframe Sphere
r = 1
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 15)
x_sph = r * np.outer(np.cos(u), np.sin(v))
y_sph = r * np.outer(np.sin(u), np.sin(v))
z_sph = r * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_wireframe(x_sph, y_sph, z_sph, color='#444444', linewidth=0.5, alpha=0.3)

# 3. Scientific Elements
# Main Point - using a bright "radar green"
point, = ax.plot([], [], [], color='#00ff00', marker='o', markersize=8,
                 markeredgecolor='white', zorder=100)

# Projection lines
line_z, = ax.plot([], [], [], color='#ff4444', linestyle='--', linewidth=1, alpha=0.7)
line_xy, = ax.plot([], [], [], color='#ffff44', linestyle='--', linewidth=1, alpha=0.7)


def get_coords(deg_theta, deg_phi):
    theta = np.radians(deg_theta)
    phi = np.radians(deg_phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


# 4. Sliders - FIXED: Changed 'valcolor' to 'color'
ax_theta = plt.axes([0.25, 0.08, 0.5, 0.02], facecolor='#222222')
ax_phi = plt.axes([0.25, 0.04, 0.5, 0.02], facecolor='#222222')

# Using 'color' instead of 'valcolor' to avoid the AttributeError
s_theta = Slider(ax_theta, 'Lon θ ', 0, 360, valinit=45, color='#00ff00')
s_phi = Slider(ax_phi, 'Lat φ ', 0, 180, valinit=45, color='#00ff00')


def update(val):
    x, y, z = get_coords(s_theta.val, s_phi.val)

    point.set_data([x], [y])
    point.set_3d_properties([z])

    line_z.set_data([x, x], [y, y])
    line_z.set_3d_properties([0, z])

    line_xy.set_data([0, x], [0, y])
    line_xy.set_3d_properties([0, 0])

    fig.canvas.draw_idle()


# Run the update once to position the point initially
update(None)

s_theta.on_changed(update)
s_phi.on_changed(update)

# 5. Clean up Axes and add a Professional Title
ax.set_xlim([-1, 1]);
ax.set_ylim([-1, 1]);
ax.set_zlim([-1, 1])
ax.grid(False)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.set_title("DOA SPHERICAL LOCALIZATION", color='white', pad=20, fontsize=12, family='monospace')

plt.show()