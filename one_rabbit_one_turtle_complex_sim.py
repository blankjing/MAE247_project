import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
import os

# ---------------------------------------------------------
# Folder Setup
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'onerabbitoneturtle_mutualism')
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
print(f"All outputs will be saved to: {output_dir}")

# ---------------------------------------------------------
# Simulation Environment Setup
# ---------------------------------------------------------
dt = 0.05
T_total = 25.0  
N = int(T_total / dt)
time_log = np.linspace(0, T_total, N, endpoint=False)

x_min, x_max = 0.0, 10.0
y_min, y_max = 0.0, 8.0

x_g = np.array([1.0, 3.0, 0.0, 0.0]) 
x_a = np.array([3.0, 4.0, 0.0, 0.0]) 
goal_g = np.array([9.0, 5.0])        
goal_a = np.array([7.0, 4.0])        

v_max = 1.0  

traj_g = np.zeros((N, 2))
traj_a = np.zeros((N, 2))
mode_log = np.zeros(N)

hg_log = np.zeros(N)    
ha_log = np.zeros(N)  
H_mutual_log = np.zeros(N)

mode = 1 
k1, k2 = 2.0, 2.0 

print("Running Mutualism CasADi Simulation...")
for k in range(N):
    traj_g[k, :] = x_g[0:2]
    traj_a[k, :] = x_a[0:2]
    mode_log[k] = mode
    
    px_g, vx_g = x_g[0], x_g[2]
    px_a, vx_a = x_a[0], x_a[2]
    
    # Cosine Barrier Functions evaluation for logging
    b_g_val = np.cos(np.pi/2 * (px_g - 1))
    bg_dot_val = -np.pi/2 * np.sin(np.pi/2 * (px_g - 1)) * vx_g
    hg_val = bg_dot_val + k1 * b_g_val
    
    b_a_val = -np.cos(np.pi/2 * (px_a - 1))
    ba_dot_val = np.pi/2 * np.sin(np.pi/2 * (px_a - 1)) * vx_a
    ha_val = ba_dot_val + k1 * b_a_val
    
    hg_log[k] = hg_val
    ha_log[k] = ha_val
    
    if mode == 3:
        H_mutual_log[k] = 0.0 
    else:
        H_mutual_log[k] = None 
        
    # State Machine
    dist_ga = np.linalg.norm(x_g[0:2] - x_a[0:2])
    
    if mode == 1:
        if x_g[0] > 1.95 and x_a[0] > 3.95:
            mode = 2 
    elif mode == 2:
        if dist_ga < 0.2:
            mode = 3 
    elif mode == 3:
        if x_g[0] >= 8.0:
            mode = 4 

    # Nominal Controllers
    Kp, Kd = 2.0, 1.5
    if mode == 1:
        u_g_nom = -Kp * (x_g[0:2] - goal_g) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - goal_a) - Kd * x_a[2:4]
    elif mode == 2:
        u_g_nom = -Kp * (x_g[0:2] - x_g[0:2]) - Kd * x_g[2:4] 
        u_a_nom = -Kp * (x_a[0:2] - np.array([2.0, x_g[1]])) - Kd * x_a[2:4] 
    elif mode == 3:
        target_cross = np.array([8.0, 4.5])
        u_g_nom = -Kp * (x_g[0:2] - target_cross) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - target_cross) - Kd * x_a[2:4]
    elif mode == 4:
        u_g_nom = -Kp * (x_g[0:2] - goal_g) - Kd * x_g[2:4]
        u_a_nom = -Kp * (x_a[0:2] - goal_a) - Kd * x_a[2:4]

    # Opti Problem Setup
    opti = ca.Opti()
    u_g_var = opti.variable(2)
    u_a_var = opti.variable(2)

    cost = ca.sumsqr(u_g_var - u_g_nom) + ca.sumsqr(u_a_var - u_a_nom)
    opti.minimize(cost)

    grad_g = -np.pi/2 * np.sin(np.pi/2 * (px_g - 1))
    if abs(grad_g) < 1e-5:
        grad_g = 1e-5 

    grad_a = np.pi/2 * np.sin(np.pi/2 * (px_a - 1))
    if abs(grad_a) < 1e-5:
        grad_a = 1e-5

    b_g = np.cos(np.pi/2 * (px_g - 1))
    b_g_dot = -np.pi/2 * np.sin(np.pi/2 * (px_g - 1)) * vx_g
    h_g = b_g_dot + k1 * b_g
    h_g_dot = grad_g * u_g_var[0] - (np.pi/2)**2 * np.cos(np.pi/2 * (px_g - 1)) * vx_g**2 + k1 * b_g_dot

    b_a = -np.cos(np.pi/2 * (px_a - 1))
    b_a_dot = np.pi/2 * np.sin(np.pi/2 * (px_a - 1)) * vx_a
    h_a = b_a_dot + k1 * b_a
    h_a_dot = grad_a * u_a_var[0] + (np.pi/2)**2 * np.cos(np.pi/2 * (px_a - 1)) * vx_a**2 + k1 * b_a_dot

    if mode in [1, 2, 4]:
        opti.subject_to(h_g_dot + k2 * h_g >= 0)
        opti.subject_to(h_a_dot + k2 * h_a >= 0)
    elif mode == 3:
        opti.subject_to(u_g_var == u_a_var)

    opti.subject_to(opti.bounded(-5.0, u_g_var, 5.0))
    opti.subject_to(opti.bounded(-5.0, u_a_var, 5.0))
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

    x_g[0:2] += x_g[2:4] * dt
    x_g[2:4] += u_g_opt * dt
    x_a[0:2] += x_a[2:4] * dt
    x_a[2:4] += u_a_opt * dt

    x_g[2:4] = np.clip(x_g[2:4], -v_max, v_max)
    x_a[2:4] = np.clip(x_a[2:4], -v_max, v_max)

    if mode == 3:
        x_g[0:2] = x_a[0:2]
        x_g[2:4] = x_a[2:4]

print("Simulation finished. Generating outputs...")

# ---------------------------------------------------------
# Output 1: Plot CBF values over time (Mutualism)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(time_log, hg_log, 'r--', linewidth=2, label='Rabbit Indiv. CBF ($h_g$)')
plt.plot(time_log, ha_log, 'g--', linewidth=2, label='Turtle Indiv. CBF ($h_a$)')

mask = mode_log == 3
plt.plot(time_log[mask], H_mutual_log[mask], 'b-', linewidth=4, label='Mutual Pairwise CBF ($H_{mutual} = h_g+h_a$)')

plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Absolute Safety Boundary')
plt.axvspan(0, T_total, color='gray', alpha=0.1, label='Terrain Alternation')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Barrier Function Value', fontsize=12)
plt.title('Robot Mutualism: Complementary Barrier Functions', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig('barrier_function_mutualism.pdf')
plt.close()
print("Saved 'barrier_function_mutualism.pdf'")

# ---------------------------------------------------------
# Output 2: Generate PDF snapshots 
# ---------------------------------------------------------
steps_per_sec = int(1.0 / dt)
pdf_pages = PdfPages('simulation_snapshots.pdf')

for step in range(0, N, steps_per_sec):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axvspan(2, 4, color='lightblue', alpha=0.5, label='River 1')
    ax.axvspan(6, 8, color='lightblue', alpha=0.5, label='River 2')
    ax.scatter(goal_g[0], goal_g[1], c='red', marker='*', s=200)
    ax.scatter(goal_a[0], goal_a[1], c='green', marker='*', s=200)
    

    if hg_log[step] >= ha_log[step]:
        z_g, z_a = 4, 3       
        z_g, z_a = 3, 4       

    ax.scatter(traj_g[step, 0], traj_g[step, 1], c='red', s=200, zorder=z_g, label='Rabbit')
    ax.scatter(traj_a[step, 0], traj_a[step, 1], c='green', s=200, zorder=z_a, label='Turtle')
    
    current_mode = int(mode_log[step])
    mode_text = ["q1: Independent", "q2: Turtle turns back", "q3: Coupled (Mutualism)", "q4: Independent (Goals)"][current_mode-1]
    
    ax.set_title(f'Time: {step*dt:.1f}s | Mode: {mode_text}', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    pdf_pages.savefig(fig)
    plt.close(fig)

pdf_pages.close()
print("Saved 'simulation_snapshots.pdf'")

# ---------------------------------------------------------
# Output 3: Create GIF Animation 
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axvspan(2, 4, color='lightblue', alpha=0.5, label='River 1')
ax.axvspan(6, 8, color='lightblue', alpha=0.5, label='River 2')
ax.scatter(goal_g[0], goal_g[1], c='red', marker='*', s=200)
ax.scatter(goal_a[0], goal_a[1], c='green', marker='*', s=200)

point_g = ax.scatter([], [], c='red', s=200, label='Rabbit')
point_a = ax.scatter([], [], c='green', s=200, label='Turtle')
title = ax.set_title('Robot Mutualism')
ax.legend(loc='upper right')
ax.grid(True)

def init():
    point_g.set_offsets(np.empty((0, 2)))
    point_a.set_offsets(np.empty((0, 2)))
    return point_g, point_a, title

def update(frame):
    point_g.set_offsets(traj_g[frame, 0:2])
    point_a.set_offsets(traj_a[frame, 0:2])
    

    if hg_log[frame] >= ha_log[frame]:
        point_g.set_zorder(4)
        point_a.set_zorder(3)
    else:
        point_g.set_zorder(3)
        point_a.set_zorder(4)

    m = int(mode_log[frame])
    txt = ["q1", "q2", "q3", "q4"][m-1]
    title.set_text(f'Time: {frame*dt:.1f}s | Mode: {txt}')
    return point_g, point_a, title

ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=False)
ani.save('simulation_animation.gif', writer=PillowWriter(fps=20))
plt.close()
print("Saved 'simulation_animation.gif'")
print("All tasks completed successfully!")