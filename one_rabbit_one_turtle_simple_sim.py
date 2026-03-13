import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
import os

# ---------------------------------------------------------
# Folder Setup: Create and move into 'onerabbitoneturtle'
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'onerabbitoneturtle')
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
print(f"All outputs will be saved to: {output_dir}")

# ---------------------------------------------------------
# Simulation Environment Setup
# ---------------------------------------------------------
dt = 0.05
T_total = 18.0
N = int(T_total / dt)
time_log = np.linspace(0, T_total, N, endpoint=False)

x_min, x_max = 0.0, 10.0
y_min, y_max = 0.0, 8.0
river_center = 5.0
river_half_width = 2.0

x_g = np.array([2.0, 3.0, 0.0, 0.0])
x_a = np.array([9.0, 3.0, 0.0, 0.0])
goal_g = np.array([8.5, 4.0])
goal_a = np.array([2.0, 4.0])

v_max = 1.0  # Velocity limit (max and min are +-v_max)

traj_g = np.zeros((N, 2))
traj_a = np.zeros((N, 2))
mode_log = np.zeros(N)

# Loggers for barrier functions
hg_log = np.zeros(N)    # Individual CBF for Rabbit
H_ga_log = np.zeros(N)  # Pairwise CBF for Rabbit
H_ag_log = np.zeros(N)  # Pairwise CBF for Turtle

mode = 1 
k1, k2 = 2.0, 2.0 # CBF parameters

# ---------------------------------------------------------
# Main Simulation Loop (MPC style with CBF)
# ---------------------------------------------------------
print("Running CasADi Simulation...")
for k in range(N):
    traj_g[k, :] = x_g[0:2]
    traj_a[k, :] = x_a[0:2]
    mode_log[k] = mode
    
    # 1. Calculate and Record Barrier Functions
    px_g, vx_g = x_g[0], x_g[2]
    b_val = (px_g - river_center)**2 - river_half_width**2
    current_hg = 2 * (px_g - river_center) * vx_g + k1 * b_val
    hg_log[k] = current_hg
    
    # Calculate pairwise collaborative influence (Eq 30)
    if mode == 3 and current_hg < 0:
        h_ga = -current_hg
    else:
        h_ga = 0.0
        
    H_ga_log[k] = current_hg + h_ga
    H_ag_log[k] = 0.0 # From Eq 31: ha = hag = 0
    
    # 2. Mode Switching Logic
    dist_ga = np.linalg.norm(x_g[0:2] - x_a[0:2])
    
    if mode == 1:
        if x_g[0] > 2.5 and x_g[0] < 7.0 and dist_ga > 0.5:
            mode = 2 
    elif mode == 2:
        if dist_ga < 0.2:
            mode = 3 
    elif mode == 3:
        if x_g[0] > 7.5:
            mode = 1 

    # 3. Nominal Controllers
    Kp, Kd = 2.0, 1.5
    if mode == 1:
        u_g_nom = -Kp * (x_g[0:2] - goal_g) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - goal_a) - Kd * x_a[2:4]
    elif mode == 2:
        u_g_nom = -Kp * (x_g[0:2] - goal_g) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - x_g[0:2]) - Kd * (x_a[2:4] - x_g[2:4])
    elif mode == 3:
        target_cross = np.array([8.0, 4.0])
        u_g_nom = -Kp * (x_g[0:2] - target_cross) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - target_cross) - Kd * x_a[2:4]

    # 4. Opti Problem Setup
    opti = ca.Opti()
    u_g_var = opti.variable(2)
    u_a_var = opti.variable(2)

    cost = ca.sumsqr(u_g_var - u_g_nom) + ca.sumsqr(u_a_var - u_a_nom)
    opti.minimize(cost)

    # CBF Constraints
    b = (px_g - river_center)**2 - river_half_width**2 
    h_g = 2 * (px_g - river_center) * vx_g + k1 * b
    h_g_dot = 2 * vx_g**2 + 2 * (px_g - river_center) * u_g_var[0] + k1 * 2 * (px_g - river_center) * vx_g

    if mode in [1, 2]:
        opti.subject_to(h_g_dot + k2 * h_g >= 0)
    elif mode == 3:
        opti.subject_to(u_g_var == u_a_var)

    # Input constraints
    opti.subject_to(opti.bounded(-5.0, u_g_var, 5.0))
    opti.subject_to(opti.bounded(-5.0, u_a_var, 5.0))

    # Velocity constraints
    opti.subject_to(opti.bounded(-v_max, x_g[2:4] + u_g_var * dt, v_max))
    opti.subject_to(opti.bounded(-v_max, x_a[2:4] + u_a_var * dt, v_max))

    p_opts = {"print_time": False}
    s_opts = {"print_level": 0, "sb": "yes"}
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        u_g_opt = sol.value(u_g_var)
        u_a_opt = sol.value(u_a_var)
    except:
        u_g_opt = opti.debug.value(u_g_var)
        u_a_opt = opti.debug.value(u_a_var)

    # State update
    x_g[0:2] += x_g[2:4] * dt
    x_g[2:4] += u_g_opt * dt
    x_a[0:2] += x_a[2:4] * dt
    x_a[2:4] += u_a_opt * dt

    # Hard clip velocity
    x_g[2:4] = np.clip(x_g[2:4], -v_max, v_max)
    x_a[2:4] = np.clip(x_a[2:4], -v_max, v_max)

    if mode == 3:
        x_g[0:2] = x_a[0:2]
        x_g[2:4] = x_a[2:4]

print("Simulation finished. Generating outputs...")

# ---------------------------------------------------------
# Output 1: Plot CBF values over time and save as PDF
# ---------------------------------------------------------
plt.figure(figsize=(9, 5))


plt.plot(time_log, H_ag_log, 'g-', linewidth=2, label='Turtle Pairwise CBF ($H_{ag}$)')

plt.plot(time_log, H_ga_log, 'b-', linewidth=2, label='Rabbit Pairwise CBF ($H_{ga}$)')

plt.plot(time_log, hg_log, 'r--', linewidth=1.5, alpha=0.5, label='Rabbit Individual CBF ($h_g$)')

plt.axhline(0, color='black', linestyle=':', linewidth=1.5, label='Safety Boundary ($H_{ij} \geq 0$)')


plt.fill_between(time_log, hg_log, 0, where=(hg_log < 0), color='red', alpha=0.15, label='Safe Region Expansion')

plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Barrier Function Value', fontsize=12)
plt.title('Minimum Value of Pairwise Barrier Functions for Collaboration', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('barrier_function_plot.pdf')
plt.close()
print("Saved 'barrier_function_plot.pdf'")

# ---------------------------------------------------------
# Output 2: Generate PDF with snapshots every 1s (Sim Time)
# ---------------------------------------------------------
steps_per_sec = int(1.0 / dt)
pdf_pages = PdfPages('simulation_snapshots.pdf')

for step in range(0, N, steps_per_sec):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(3, 7, color='lightblue', alpha=0.5, label='River')
    
    ax.scatter(traj_g[step, 0], traj_g[step, 1], c='red', s=100, label='Rabbit Current')
    ax.scatter(traj_a[step, 0], traj_a[step, 1], c='green', s=100, label='Turtle Current')
    ax.scatter(goal_g[0], goal_g[1], c='red', marker='*', s=150)
    ax.scatter(goal_a[0], goal_a[1], c='green', marker='*', s=150)
    
    current_mode = int(mode_log[step])
    ax.set_title(f'Simulation Time: {step*dt:.1f}s | Current Mode: q{current_mode}', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    pdf_pages.savefig(fig)
    plt.close(fig)

pdf_pages.close()
print("Saved 'simulation_snapshots.pdf'")

# ---------------------------------------------------------
# Output 3: Create and save GIF Animation
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axvspan(3, 7, color='lightblue', alpha=0.5, label='River')
ax.scatter(goal_g[0], goal_g[1], c='red', marker='*', s=150, label='Rabbit Goal')
ax.scatter(goal_a[0], goal_a[1], c='green', marker='*', s=150, label='Turtle Goal')

point_g = ax.scatter([], [], c='red', s=100, label='Rabbit')
point_a = ax.scatter([], [], c='green', s=100, label='Turtle')
title = ax.set_title('Resiliency Through Pairwise Collaboration')
ax.legend(loc='upper right')
ax.grid(True)

def init():
    point_g.set_offsets(np.empty((0, 2)))
    point_a.set_offsets(np.empty((0, 2)))
    return point_g, point_a, title

def update(frame):
    point_g.set_offsets(traj_g[frame, 0:2])
    point_a.set_offsets(traj_a[frame, 0:2])
    title.set_text(f'Time: {frame*dt:.1f}s | Mode: q{int(mode_log[frame])}')
    return point_g, point_a, title

ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=False)
ani.save('simulation_animation.gif', writer=PillowWriter(fps=20))
plt.close()
print("Saved 'simulation_animation.gif'")
print("All tasks completed successfully!")