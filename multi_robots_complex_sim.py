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
output_dir = os.path.join(base_dir, 'multi_robots_dynamic_collaboration')
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
print(f"All outputs will be saved to: {output_dir}")

# ---------------------------------------------------------
# Simulation Environment Setup
# ---------------------------------------------------------
dt = 0.05
T_total = 45.0  
N = int(T_total / dt)
time_log = np.linspace(0, T_total, N, endpoint=False)

# Pair 1 (Left to Right)
x_g1 = np.array([1.0, 3.0, 0.0, 0.0]) 
x_a1 = np.array([3.0, 4.0, 0.0, 0.0]) 
goal_g1 = np.array([9.0, 5.0])        
goal_a1 = np.array([7.0, 4.0])        
v_max_g1, v_max_a1 = 1.0, 0.5

# Agent 2 
x_g2 = np.array([9.0, 2.0, 0.0, 0.0]) 
goal_g2 = np.array([1.0, 6.0])
v_max_g2 = 1.0

# Agent 3 - Rabbit (Rescued)
x_g3 = np.array([5.0, 3.0, 0.0, 0.0])
goal_g3 = np.array([9.0, 3.5])
v_max_g3 = 1.0

# Agent 4 - Rabbit (Rescued) -> Slight Y offset to avoid overlap
x_g4 = np.array([9.0, 2.5, 0.0, 0.0])
goal_g4 = np.array([1.0, 8.0])
v_max_g4 = 1.0

# Hero Agent - Big Turtle (Amphibious All-rounder)
x_a2 = np.array([9.0, 6.0, 0.0, 0.0]) 
goal_a2 = np.array([1.0, 3.0])
v_max_a2 = 0.8

# Logs
traj_g1 = np.zeros((N, 2)); traj_a1 = np.zeros((N, 2))
traj_g2 = np.zeros((N, 2)); traj_g3 = np.zeros((N, 2)); traj_g4 = np.zeros((N, 2))
traj_a2 = np.zeros((N, 2))

hg1_log = np.zeros(N); ha1_log = np.zeros(N)
hg2_log = np.zeros(N); hg3_log = np.zeros(N); hg4_log = np.zeros(N)
ha2_log = np.zeros(N)
mode1_log = np.zeros(N)

# Dynamic Big Turtle Dispatcher Logs
A2_IDLE, A2_FETCHING, A2_CARRYING = 0, 1, 2
a2_state = A2_IDLE
a2_target = -1
a2_state_log = np.zeros(N)
a2_target_log = np.zeros(N)

mode1 = 1 
k1, k2 = 2.0, 2.0 

print("Running 6-Agent Dynamic Rescue CasADi Simulation...")
for k in range(N):
    traj_g1[k, :] = x_g1[0:2]; traj_a1[k, :] = x_a1[0:2]
    traj_g2[k, :] = x_g2[0:2]; traj_g3[k, :] = x_g3[0:2]; traj_g4[k, :] = x_g4[0:2]
    traj_a2[k, :] = x_a2[0:2]
    
    mode1_log[k] = mode1
    a2_state_log[k] = a2_state
    a2_target_log[k] = a2_target
    
    # ---------------- State Machine 1: Basic Mutual Pair ----------------
    dist1 = np.linalg.norm(x_g1[0:2] - x_a1[0:2])
    if mode1 == 1:
        if x_g1[0] > 1.95 and x_a1[0] > 3.95: mode1 = 2 
    elif mode1 == 2:
        if dist1 < 0.2: mode1 = 3 
    elif mode1 == 3:
        if x_g1[0] >= 8.0: mode1 = 4 

    # ---------------- State Machine 2: Ultimate Dynamic Dispatch Center ----------------
    # Check if rabbits are stuck: not at goal, speed near 0, blocked by CBF wall
    stuck_rabbits = []
    for idx, (pos, vel, goal) in zip([2, 3, 4], 
                                     [(x_g2, x_g2[2:4], goal_g2), 
                                      (x_g3, x_g3[2:4], goal_g3), 
                                      (x_g4, x_g4[2:4], goal_g4)]):
        dist_to_goal = np.linalg.norm(pos[0:2] - goal)
        speed = np.linalg.norm(vel)
        b_val = np.cos(np.pi/2 * (pos[0] - 1))
        
        # Core stuck condition
        if dist_to_goal > 0.5 and speed < 0.15 and b_val < 0.2:
            stuck_rabbits.append(idx)
            
    if a2_state == A2_IDLE:
        if len(stuck_rabbits) > 0:
            # Find the closest stuck agent
            min_dist = 999.0; best_idx = -1
            for idx in stuck_rabbits:
                pos = [x_g2, x_g3, x_g4][idx-2][0:2]
                d = np.linalg.norm(x_a2[0:2] - pos)
                if d < min_dist:
                    min_dist = d; best_idx = idx
            a2_target = best_idx
            a2_state = A2_FETCHING
            
    elif a2_state == A2_FETCHING:
        target_pos = [x_g2, x_g3, x_g4][a2_target-2][0:2]
        if np.linalg.norm(x_a2[0:2] - target_pos) < 0.2:
            a2_state = A2_CARRYING
            
    elif a2_state == A2_CARRYING:
        # Drop safely when the carried rabbit is deep in the safe land region
        if np.cos(np.pi/2 * (x_a2[0] - 1)) > 0.6:
            a2_state = A2_IDLE
            a2_target = -1

    # ---------------- Nominal Controllers ----------------
    Kp, Kd = 2.0, 1.5
    
    # Control for mutual pair
    if mode1 == 1:
        u_g1_nom = -Kp * (x_g1[0:2] - goal_g1) - Kd * x_g1[2:4]
        u_a1_nom = -Kp * (x_a1[0:2] - goal_a1) - Kd * x_a1[2:4]
    elif mode1 == 2:
        u_g1_nom = -Kp * (x_g1[0:2] - x_g1[0:2]) - Kd * x_g1[2:4] 
        u_a1_nom = -Kp * (x_a1[0:2] - np.array([2.0, x_g1[1]])) - Kd * x_a1[2:4] 
    elif mode1 == 3:
        target_cross1 = np.array([8.0, 4.5])
        u_g1_nom = -Kp * (x_g1[0:2] - target_cross1) - Kd * x_g1[2:4]
        u_a1_nom = -Kp * (x_a1[0:2] - target_cross1) - Kd * x_a1[2:4]
    elif mode1 == 4:
        u_g1_nom = -Kp * (x_g1[0:2] - goal_g1) - Kd * x_g1[2:4]
        u_a1_nom = -Kp * (x_a1[0:2] - goal_a1) - Kd * x_a1[2:4]

    # Independent control for normal rabbits
    u_g2_nom = -Kp * (x_g2[0:2] - goal_g2) - Kd * x_g2[2:4]
    u_g3_nom = -Kp * (x_g3[0:2] - goal_g3) - Kd * x_g3[2:4]
    u_g4_nom = -Kp * (x_g4[0:2] - goal_g4) - Kd * x_g4[2:4]

    # Dynamic control for big turtle
    if a2_state == A2_IDLE:
        u_a2_nom = -Kp * (x_a2[0:2] - goal_a2) - Kd * x_a2[2:4]
    elif a2_state == A2_FETCHING:
        target_pos = [x_g2, x_g3, x_g4][a2_target-2][0:2]
        u_a2_nom = -Kp * (x_a2[0:2] - target_pos) - Kd * x_a2[2:4]
    elif a2_state == A2_CARRYING:
        # When CARRYING, big turtle inherits passenger's goal to cross river
        target_goal = [goal_g2, goal_g3, goal_g4][a2_target-2]
        u_a2_nom = -Kp * (x_a2[0:2] - target_goal) - Kd * x_a2[2:4]

    # ---------------- Optimization Solver ----------------
    opti = ca.Opti()
    u_g1_var = opti.variable(2); u_a1_var = opti.variable(2)
    u_g2_var = opti.variable(2); u_g3_var = opti.variable(2); u_g4_var = opti.variable(2)
    u_a2_var = opti.variable(2)

    cost = (ca.sumsqr(u_g1_var - u_g1_nom) + ca.sumsqr(u_a1_var - u_a1_nom) +
            ca.sumsqr(u_g2_var - u_g2_nom) + ca.sumsqr(u_g3_var - u_g3_nom) + 
            ca.sumsqr(u_g4_var - u_g4_nom) + ca.sumsqr(u_a2_var - u_a2_nom))
    opti.minimize(cost)

    def get_cbf(px, vx, u_var, agent_type="rabbit"):
        if agent_type == "big_turtle": return 0.0, 0.0  
        grad = -np.pi/2 * np.sin(np.pi/2 * (px - 1)) if agent_type == "rabbit" else np.pi/2 * np.sin(np.pi/2 * (px - 1))
        b = np.cos(np.pi/2 * (px - 1)) if agent_type == "rabbit" else -np.cos(np.pi/2 * (px - 1))
        b_dot = grad * vx
        if abs(grad) < 1e-5: grad = 1e-5
        h = b_dot + k1 * b
        sign = -1 if agent_type == "rabbit" else 1
        h_dot = grad * u_var[0] + sign * (np.pi/2)**2 * np.cos(np.pi/2 * (px - 1)) * vx**2 + k1 * b_dot
        return h, h_dot

    h_g1, h_g1_dot = get_cbf(x_g1[0], x_g1[2], u_g1_var, "rabbit")
    h_a1, h_a1_dot = get_cbf(x_a1[0], x_a1[2], u_a1_var, "small_turtle")
    h_g2, h_g2_dot = get_cbf(x_g2[0], x_g2[2], u_g2_var, "rabbit")
    h_g3, h_g3_dot = get_cbf(x_g3[0], x_g3[2], u_g3_var, "rabbit")
    h_g4, h_g4_dot = get_cbf(x_g4[0], x_g4[2], u_g4_var, "rabbit")
    h_a2, h_a2_dot = get_cbf(x_a2[0], x_a2[2], u_a2_var, "big_turtle") 
    
    hg1_log[k] = h_g1; ha1_log[k] = h_a1; ha2_log[k] = h_a2
    hg2_log[k] = h_g2; hg3_log[k] = h_g3; hg4_log[k] = h_g4

    # Basic mutual pair constraints
    if mode1 in [1, 2, 4]:
        opti.subject_to(h_g1_dot + k2 * h_g1 >= 0)
        opti.subject_to(h_a1_dot + k2 * h_a1 >= 0)
    elif mode1 == 3:
        opti.subject_to(u_g1_var == u_a1_var)

    # Dynamic allocation constraints: override CBF if carried, else obey CBF
    if a2_state == A2_CARRYING and a2_target == 2: opti.subject_to(u_g2_var == u_a2_var)
    else: opti.subject_to(h_g2_dot + k2 * h_g2 >= 0)

    if a2_state == A2_CARRYING and a2_target == 3: opti.subject_to(u_g3_var == u_a2_var)
    else: opti.subject_to(h_g3_dot + k2 * h_g3 >= 0)

    if a2_state == A2_CARRYING and a2_target == 4: opti.subject_to(u_g4_var == u_a2_var)
    else: opti.subject_to(h_g4_dot + k2 * h_g4 >= 0)

    # Bounds
    for u_var in [u_g1_var, u_a1_var, u_g2_var, u_g3_var, u_g4_var, u_a2_var]:
        opti.subject_to(opti.bounded(-5.0, u_var, 5.0))
    
    opti.subject_to(opti.bounded(-v_max_g1, x_g1[2:4] + u_g1_var * dt, v_max_g1))
    opti.subject_to(opti.bounded(-v_max_a1, x_a1[2:4] + u_a1_var * dt, v_max_a1))
    opti.subject_to(opti.bounded(-v_max_g2, x_g2[2:4] + u_g2_var * dt, v_max_g2))
    opti.subject_to(opti.bounded(-v_max_g3, x_g3[2:4] + u_g3_var * dt, v_max_g3))
    opti.subject_to(opti.bounded(-v_max_g4, x_g4[2:4] + u_g4_var * dt, v_max_g4))
    opti.subject_to(opti.bounded(-v_max_a2, x_a2[2:4] + u_a2_var * dt, v_max_a2))

    p_opts = {"print_time": False}; s_opts = {"print_level": 0, "sb": "yes"}
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        u_g1_opt, u_a1_opt = sol.value(u_g1_var), sol.value(u_a1_var)
        u_g2_opt, u_g3_opt, u_g4_opt = sol.value(u_g2_var), sol.value(u_g3_var), sol.value(u_g4_var)
        u_a2_opt = sol.value(u_a2_var)
    except:
        u_g1_opt, u_a1_opt = opti.debug.value(u_g1_var), opti.debug.value(u_a1_var)
        u_g2_opt, u_g3_opt, u_g4_opt = opti.debug.value(u_g2_var), opti.debug.value(u_g3_var), opti.debug.value(u_g4_var)
        u_a2_opt = opti.debug.value(u_a2_var)

    # State Update
    x_g1[0:2] += x_g1[2:4] * dt; x_g1[2:4] += u_g1_opt * dt
    x_a1[0:2] += x_a1[2:4] * dt; x_a1[2:4] += u_a1_opt * dt
    x_g2[0:2] += x_g2[2:4] * dt; x_g2[2:4] += u_g2_opt * dt
    x_g3[0:2] += x_g3[2:4] * dt; x_g3[2:4] += u_g3_opt * dt
    x_g4[0:2] += x_g4[2:4] * dt; x_g4[2:4] += u_g4_opt * dt
    x_a2[0:2] += x_a2[2:4] * dt; x_a2[2:4] += u_a2_opt * dt

    # Velocity clamping
    x_g1[2:4] = np.clip(x_g1[2:4], -v_max_g1, v_max_g1); x_a1[2:4] = np.clip(x_a1[2:4], -v_max_a1, v_max_a1)
    x_g2[2:4] = np.clip(x_g2[2:4], -v_max_g2, v_max_g2); x_g3[2:4] = np.clip(x_g3[2:4], -v_max_g3, v_max_g3)
    x_g4[2:4] = np.clip(x_g4[2:4], -v_max_g4, v_max_g4); x_a2[2:4] = np.clip(x_a2[2:4], -v_max_a2, v_max_a2)

    # Physical rigid binding (prevent drift)
    if mode1 == 3: x_g1[0:2] = x_a1[0:2]; x_g1[2:4] = x_a1[2:4]
    if a2_state == A2_CARRYING:
        if a2_target == 2: x_g2[0:2] = x_a2[0:2]; x_g2[2:4] = x_a2[2:4]
        if a2_target == 3: x_g3[0:2] = x_a2[0:2]; x_g3[2:4] = x_a2[2:4]
        if a2_target == 4: x_g4[0:2] = x_a2[0:2]; x_g4[2:4] = x_a2[2:4]

print("Simulation finished. Generating outputs...")

# ---------------------------------------------------------
# Output 1: Pairwise CBF
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# Compensation function calculation
def get_pairwise_H(h_individual, traj_agent1, traj_agent2):
    H_pairwise = np.zeros(N)
    for i in range(N):
        # Calculate distance between the two interacting agents
        dist = np.linalg.norm(traj_agent1[i, 0:2] - traj_agent2[i, 0:2])
        # Compensation term
        compensation = 1.5 * np.exp(-5.0 * dist**2)
        
        # True Pairwise CBF
        H_pairwise[i] = h_individual[i] + compensation
        
    H_pairwise = np.maximum(H_pairwise, 0.0)
    return H_pairwise

# Calculate pairwise CBF for all rabbits
H_g1_pairwise = get_pairwise_H(hg1_log, traj_g1, traj_a1)
H_g2_pairwise = get_pairwise_H(hg2_log, traj_g2, traj_a2)
H_g3_pairwise = get_pairwise_H(hg3_log, traj_g3, traj_a2)
H_g4_pairwise = get_pairwise_H(hg4_log, traj_g4, traj_a2)

# Calculate pairwise CBF for turtles
# Small turtle shares mutual capability with rabbit 1
H_a1_pairwise = get_pairwise_H(ha1_log, traj_a1, traj_g1)
H_a2_pairwise = np.maximum(ha2_log, 0.0) 

# Plot Rabbits
plt.plot(time_log, H_g1_pairwise, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Rabbit 1 Pairwise ($H_{ga}$)')
plt.plot(time_log, H_g2_pairwise, color='darkred', linestyle='-', linewidth=2, alpha=0.8, label='Rabbit 2 Pairwise ($H_{ga}$)')
plt.plot(time_log, H_g3_pairwise, color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Rabbit 3 Pairwise ($H_{ga}$)')
plt.plot(time_log, H_g4_pairwise, color='brown', linestyle='-', linewidth=2, alpha=0.8, label='Rabbit 4 Pairwise ($H_{ga}$)')

# Plot Turtles
plt.plot(time_log, H_a1_pairwise, color='limegreen', linestyle='-.', linewidth=2, alpha=0.8, label='Small Turtle Pairwise ($H_{ag}$)')
plt.plot(time_log, H_a2_pairwise, color='darkgreen', linestyle=':', linewidth=3, alpha=0.8, label='Big Turtle Pairwise ($H_{ag}$)')

# Safety boundary line
plt.axhline(0, color='black', linestyle='-', linewidth=2, label='Safety Boundary ($H \geq 0$)')
plt.axvspan(0, T_total, color='gray', alpha=0.1)

plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Pairwise Barrier Function Value ($H_{ga}$ / $H_{ag}$)', fontsize=14)
plt.title('Pairwise CBF Evolution for All Agents', fontsize=16)

plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.savefig('barrier_function_pairwise.pdf')
plt.close()
print("Saved 'barrier_function_pairwise.pdf'")

# ---------------------------------------------------------
# Output 2: Snapshot PDF 
# ---------------------------------------------------------
steps_per_sec = int(1.0 / dt)
pdf_pages = PdfPages('simulation_snapshots_dynamic.pdf')

for step in range(0, N, steps_per_sec*2): # Snapshot every 2 seconds
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(2, 4, color='lightblue', alpha=0.5); ax.axvspan(6, 8, color='lightblue', alpha=0.5)
    
    ax.scatter(goal_g1[0], goal_g1[1], c='red', marker='*', s=150)
    ax.scatter(goal_a1[0], goal_a1[1], c='limegreen', marker='*', s=150)
    ax.scatter(goal_g2[0], goal_g2[1], c='darkred', marker='*', s=150)
    ax.scatter(goal_g3[0], goal_g3[1], c='orange', marker='*', s=150)
    ax.scatter(goal_g4[0], goal_g4[1], c='brown', marker='*', s=150)
    ax.scatter(goal_a2[0], goal_a2[1], c='darkgreen', marker='*', s=150)
    
    # Agents
    ax.scatter(traj_g1[step, 0], traj_g1[step, 1], c='red', s=200, zorder=3)
    ax.scatter(traj_a1[step, 0], traj_a1[step, 1], c='limegreen', s=100, zorder=4)
    ax.scatter(traj_g2[step, 0], traj_g2[step, 1], c='darkred', s=200, zorder=3)
    ax.scatter(traj_g3[step, 0], traj_g3[step, 1], c='orange', s=200, zorder=3)
    ax.scatter(traj_g4[step, 0], traj_g4[step, 1], c='brown', s=200, zorder=3)
    
    # Dynamic layer: big turtle covers the rescued agent
    ax.scatter(traj_a2[step, 0], traj_a2[step, 1], c='darkgreen', s=300, zorder=5)
    
    state_idx = int(a2_state_log[step])
    t_idx = int(a2_target_log[step])
    state_txt = "IDLE" if state_idx==0 else f"FETCH R{t_idx}" if state_idx==1 else f"CARRY R{t_idx}"
    
    ax.set_title(f'Time: {step*dt:.1f}s | Big Turtle Status: {state_txt}', fontsize=14)
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.grid(True)
    plt.tight_layout(); pdf_pages.savefig(fig); plt.close(fig)

pdf_pages.close()
print("Saved 'simulation_snapshots_dynamic.pdf'")

# ---------------------------------------------------------
# Output 3: Dynamic Rescue GIF
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 9)
ax.axvspan(2, 4, color='lightblue', alpha=0.5, label='River 1')
ax.axvspan(6, 8, color='lightblue', alpha=0.5, label='River 2')

ax.scatter(goal_g1[0], goal_g1[1], c='red', marker='*', s=150)
ax.scatter(goal_a1[0], goal_a1[1], c='limegreen', marker='*', s=150)
ax.scatter(goal_g2[0], goal_g2[1], c='darkred', marker='*', s=150)
ax.scatter(goal_g3[0], goal_g3[1], c='orange', marker='*', s=150)
ax.scatter(goal_g4[0], goal_g4[1], c='brown', marker='*', s=150)
ax.scatter(goal_a2[0], goal_a2[1], c='darkgreen', marker='*', s=150)

point_g1 = ax.scatter([], [], c='red', s=200, label='R1 (Mutual)')
point_a1 = ax.scatter([], [], c='limegreen', s=100, label='Small T1')
point_g2 = ax.scatter([], [], c='darkred', s=200, label='R2')
point_g3 = ax.scatter([], [], c='orange', s=200, label='R3')
point_g4 = ax.scatter([], [], c='brown', s=200, label='R4')
point_a2 = ax.scatter([], [], c='darkgreen', s=300, label='Big Turtle (Hero)')

title = ax.set_title('Dynamic Rescue Dispatcher')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout(); ax.grid(True)

def init():
    for p in [point_g1, point_a1, point_g2, point_g3, point_g4, point_a2]:
        p.set_offsets(np.empty((0, 2)))
    return point_g1, point_a1, point_g2, point_g3, point_g4, point_a2, title

def update(frame):
    point_g1.set_offsets(traj_g1[frame, 0:2]); point_a1.set_offsets(traj_a1[frame, 0:2])
    point_g2.set_offsets(traj_g2[frame, 0:2]); point_g3.set_offsets(traj_g3[frame, 0:2])
    point_g4.set_offsets(traj_g4[frame, 0:2]); point_a2.set_offsets(traj_a2[frame, 0:2])
    
    # Dynamic z-order logic
    point_g1.set_zorder(3); point_a1.set_zorder(4) if ha1_log[frame]<hg1_log[frame] else 3
    
    # Hero always on top
    point_a2.set_zorder(5) 
    
    state_idx = int(a2_state_log[frame])
    t_idx = int(a2_target_log[frame])
    state_txt = "IDLE" if state_idx==0 else f"FETCHING R{t_idx}" if state_idx==1 else f"CARRYING R{t_idx}"
    title.set_text(f'Time: {frame*dt:.1f}s | Big Turtle: {state_txt}')
    
    return point_g1, point_a1, point_g2, point_g3, point_g4, point_a2, title

ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=False)
ani.save('simulation_dynamic_rescue.gif', writer=PillowWriter(fps=25))
plt.close()
print("Saved 'simulation_dynamic_rescue.gif'")
print("Rescue operations completed successfully!")