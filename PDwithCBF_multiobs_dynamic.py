import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class Car:
    def __init__(self, x, y, theta, v=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

    def update_dynamics(self, a, alpha, dt):
        self.v += a * dt
        self.omega += alpha * dt
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta to [-pi, pi]

def generate_reference_trajectory(start, goal, num_points=100):
    x_ref = np.linspace(start[0], goal[0], num_points)
    y_ref = np.linspace(start[1], goal[1], num_points)
    return x_ref, y_ref

def pd_controller(car_state, x_ref, y_ref, dt):
    kp_linear = 1.0
    kp_angular = 1.5

    dx = x_ref - car_state['x']
    dy = y_ref - car_state['y']
    distance = np.hypot(dx, dy)
    angle_to_ref = np.arctan2(dy, dx)

    angle_error = angle_to_ref - car_state['theta']
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # Normalize angle error to [-pi, pi]

    v_max = 2.0    # Maximum linear velocity
    omega_max = np.pi  # Maximum angular velocity

    v_desired = kp_linear * distance
    v_desired = np.clip(v_desired, -v_max, v_max)

    omega_desired = kp_angular * angle_error
    omega_desired = np.clip(omega_desired, -omega_max, omega_max)

    a_desired = (v_desired - car_state['v']) / dt
    alpha_desired = (omega_desired - car_state['omega']) / dt

    return a_desired, alpha_desired

def compute_h(car_state, obstacle, l):
    x = car_state['x']
    y = car_state['y']
    theta = car_state['theta']
    v = car_state['v']
    omega = car_state['omega']

    x_o = obstacle['x']
    y_o = obstacle['y']
    v_o = obstacle['v']
    theta_o = obstacle['theta']
    R_o = obstacle['r']

    rel_pos = np.array([
        x_o - x - l * np.cos(theta),
        y_o - y - l * np.sin(theta)
    ])
    rel_vel = np.array([
        v_o * np.cos(theta_o) - v * np.cos(theta) + l * np.sin(theta) * omega,
        v_o * np.sin(theta_o) - v * np.sin(theta) - l * np.cos(theta) * omega
    ])

    epsilon = 1e-6
    norm_rel_pos = np.linalg.norm(rel_pos) + epsilon
    norm_rel_vel = np.linalg.norm(rel_vel) + epsilon

    sqrt_term = np.sqrt(norm_rel_pos**2 - (R_o+0.5)**2 + epsilon)
    cos_phi = sqrt_term / norm_rel_pos

    h = np.dot(rel_pos, rel_vel) + norm_rel_pos * norm_rel_vel * cos_phi
    return h

def solve_collision_cone_cbf(a_d, alpha_d, car_state, obstacles, gamma, dt, l):
    x_val = car_state['x']
    y_val = car_state['y']
    theta_val = car_state['theta']
    v_val = car_state['v']
    omega_val = car_state['omega']

    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')
    v = ca.MX.sym('v')
    omega = ca.MX.sym('omega')

    p = ca.vertcat(x, y, theta, v, omega)
    param_values = [x_val, y_val, theta_val, v_val, omega_val]

    a_opt = ca.MX.sym('a_opt')
    alpha_opt = ca.MX.sym('alpha_opt')
    x_opt = ca.vertcat(a_opt, alpha_opt)

    constraints = []

    for idx, obs in enumerate(obstacles):
        x_o_val = obs['x']
        y_o_val = obs['y']
        v_o_val = obs['v']
        theta_o_val = obs['theta']
        R_o_val = obs['r']

        x_o = ca.MX.sym(f"x_o_{idx}")
        y_o = ca.MX.sym(f"y_o_{idx}")
        v_o = ca.MX.sym(f"v_o_{idx}")
        theta_o = ca.MX.sym(f"theta_o_{idx}")
        R_o = ca.MX.sym(f"R_o_{idx}")

        p = ca.vertcat(p, x_o, y_o, v_o, theta_o, R_o)
        param_values.extend([x_o_val, y_o_val, v_o_val, theta_o_val, R_o_val])

        rel_pos = ca.vertcat(x_o - x - l * ca.cos(theta),
                             y_o - y - l * ca.sin(theta))
        rel_vel = ca.vertcat(
            v_o * ca.cos(theta_o) - v * ca.cos(theta) + l * ca.sin(theta) * omega,
            v_o * ca.sin(theta_o) - v * ca.sin(theta) - l * ca.cos(theta) * omega
        )

        epsilon = 1e-6
        norm_rel_pos = ca.norm_2(rel_pos) + epsilon
        norm_rel_vel = ca.norm_2(rel_vel) + epsilon

        sqrt_term = ca.sqrt(norm_rel_pos**2 - (R_o+0.5)**2 + epsilon)
        cos_phi = sqrt_term / norm_rel_pos

        h = ca.dot(rel_pos, rel_vel) + norm_rel_pos * norm_rel_vel * cos_phi

        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = omega
        v_dot = a_opt
        omega_dot = alpha_opt

        state_vars = ca.vertcat(x, y, theta, v, omega)
        state_dots = ca.vertcat(x_dot, y_dot, theta_dot, v_dot, omega_dot)

        h_dot = ca.jtimes(h, state_vars, state_dots)

        constraints.append(h_dot + gamma * h)

    cost = (a_opt - a_d) ** 2 + (alpha_opt - alpha_d) ** 2

    nlp = {'x': x_opt, 'f': cost, 'g': ca.vertcat(*constraints), 'p': p}

    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.tol': 1e-4,
        'ipopt.max_iter': 1000,
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.acceptable_obj_change_tol': 1e-3,
    })

    try:
        solution = solver(x0=[a_d, alpha_d], lbg=0, ubg=ca.inf, p=param_values)
        a_opt_val = float(solution['x'][0])
        alpha_opt_val = float(solution['x'][1])
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        a_opt_val = a_d
        alpha_opt_val = alpha_d

    return a_opt_val, alpha_opt_val

def run_simulation():
    dt = 0.1
    num_steps = 500
    gamma = 7
    l = 0.075
    start = (-5.0, -5.0)
    goal = (5.0, 5.0)
    car = Car(x=start[0], y=start[1], theta=np.pi / 4)

    # Define dynamic obstacles.
    # Example: One moving to the right (theta=0.0 at v=0.5), another moving upwards (theta=pi/2 at v=0.3).
    obstacles = [
        {'x': -3.0, 'y': 0.0, 'v': 0.5, 'theta': 0.0, 'r': 0.5},
        {'x': 0.1, 'y': 1.0, 'v': 0.2, 'theta': 5*np.pi/4, 'r': 0.5},
        {'x': 1.0, 'y': 1.0, 'v': 0.0, 'theta': np.pi/2, 'r': 0.5}
    ]

    x_ref_traj, y_ref_traj = generate_reference_trajectory(start, goal)

    car_x_traj = []
    car_y_traj = []
    time_steps = []
    velocities = []
    accelerations = []
    h_values = []
    h_time_steps = []
    angular_velocities = []
    angular_accelerations = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    car_plot, = ax.plot([], [], 'bo', label='Car')
    car_traj_plot, = ax.plot([], [], 'b-', label='Car Trajectory')
    trajectory_plot, = ax.plot(x_ref_traj, y_ref_traj, 'g--', label='Reference Trajectory')

    # Plot obstacles as circles
    obstacle_patches = []
    for obs in obstacles:
        obstacle_patch = plt.Circle((obs['x'], obs['y']), obs['r'], color='r', alpha=0.5)
        ax.add_patch(obstacle_patch)
        obstacle_patches.append(obstacle_patch)

    ax.legend()
    plt.title("PD with CBF and Dynamic Obstacles")

    for step in range(num_steps):
        current_time = step * dt
        current_waypoint = min(step, len(x_ref_traj) - 1)
        x_ref = x_ref_traj[current_waypoint]
        y_ref = y_ref_traj[current_waypoint]

        # Update obstacle positions (dynamic obstacles)
        for obs in obstacles:
            obs['x'] += obs['v'] * np.cos(obs['theta']) * dt
            obs['y'] += obs['v'] * np.sin(obs['theta']) * dt

        car_state = {'x': car.x, 'y': car.y, 'theta': car.theta, 'v': car.v, 'omega': car.omega}
        a_pd, alpha_pd = pd_controller(car_state, x_ref, y_ref, dt)

        cbf_active = False
        for obs in obstacles:
            if np.hypot(car.x - obs['x'], car.y - obs['y']) < (obs['r'] + 2.0):
                cbf_active = True
                break

        if cbf_active:
            a_opt, alpha_opt = solve_collision_cone_cbf(a_pd, alpha_pd, car_state, obstacles, gamma, dt, l)
            for obs in obstacles:
                h_value = compute_h(car_state, obs, l)
                h_values.append(h_value)
                h_time_steps.append(current_time)
        else:
            a_opt, alpha_opt = a_pd, alpha_pd

        car.update_dynamics(a_opt, alpha_opt, dt)

        car_x_traj.append(car.x)
        car_y_traj.append(car.y)
        time_steps.append(current_time)
        velocities.append(car.v)
        accelerations.append(a_opt)
        angular_velocities.append(car.omega)
        angular_accelerations.append(alpha_opt)

        # Update visualization
        car_plot.set_data(car.x, car.y)
        car_traj_plot.set_data(car_x_traj, car_y_traj)
        trajectory_plot.set_data(x_ref_traj[:current_waypoint + 1], y_ref_traj[:current_waypoint + 1])

        # Update obstacle positions in the animation
        for patch, obs in zip(obstacle_patches, obstacles):
            patch.center = (obs['x'], obs['y'])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        # Check if goal is reached
        if np.hypot(car.x - goal[0], car.y - goal[1]) < 0.2:
            print("Goal reached at step:", step)
            break

    plt.ioff()

    plt.savefig('final_trajectory_dynamic.png')
    plt.show()

    plt.figure()
    plt.plot(time_steps, velocities, label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s)')
    plt.title('Velocity vs Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_vs_time_dynamic.png')
    plt.show()

    plt.figure()
    plt.plot(time_steps, accelerations, label='Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Acceleration (m/s²)')
    plt.title('Acceleration vs Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('acceleration_vs_time_dynamic.png')
    plt.show()

    if h_values:
        plt.figure()
        plt.plot(h_time_steps, h_values, label='Barrier Function h')
        plt.xlabel('Time (s)')
        plt.ylabel('h Value')
        plt.title('Barrier Function h vs Time Steps (CBF Active, Dynamic Obstacles)')
        plt.legend()
        plt.grid(True)
        plt.savefig('h_vs_time_dynamic.png')
        plt.show()
    else:
        print("CBF was not active during the simulation; no h values to plot.")

    plt.figure()
    plt.plot(time_steps, angular_velocities, label='Angular Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity vs Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('angular_velocity_vs_time_dynamic.png')
    plt.show()

    plt.figure()
    plt.plot(time_steps, angular_accelerations, label='Angular Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Acceleration (rad/s²)')
    plt.title('Angular Acceleration vs Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('angular_acceleration_vs_time_dynamic.png')
    plt.show()

if __name__ == "__main__":
    run_simulation()
