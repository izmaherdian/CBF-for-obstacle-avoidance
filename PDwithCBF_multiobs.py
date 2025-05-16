"""
人工势场寻路算法实现
改进人工势场，解决不可达问题，仍存在局部最小点问题
"""
from Original_APF import APF, Vector2d
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import random


def check_vec_angle(v1: Vector2d, v2: Vector2d):
    v1_v2 = v1.deltaX * v2.deltaX + v1.deltaY * v2.deltaY
    angle = math.acos(v1_v2 / (v1.length * v2.length)) * 180 / math.pi
    return angle


class APF_Improved(APF):
    def __init__(self, start: (), goal: (), obstacles: [], k_att: float, k_rep: float, rr: float,
                 step_size: float, max_iters: int, goal_threshold: float, is_plot=False):
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.step_size = step_size
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = is_plot
        self.delta_t = 0.01

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            # obstacle = Vector2d(0, 0)
            obs_to_rob = self.current_pos - obstacle
            rob_to_goal = self.goal - self.current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector2d(obs_to_rob.direction[0], obs_to_rob.direction[1]) * self.k_rep * (
                        1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector2d(rob_to_goal.direction[0], rob_to_goal.direction[1]) * self.k_rep * ((1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep +=(rep_1+rep_2)
        return rep


if __name__ == '__main__':
    # 相关参数设置
    k_att, k_rep = 1.0, 0.8
    rr = 3
    step_size, max_iters, goal_threashold = .2, 500, .2  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
    step_size_ = 2

    # 设置、绘制起点终点
    start, goal = (0, 0), (15, 15)
    is_plot = True
    if is_plot:
        fig = plt.figure(figsize=(7, 7))
        subplot = fig.add_subplot(111)
        subplot.set_xlabel('X-distance: m')
        subplot.set_ylabel('Y-distance: m')
        subplot.plot(start[0], start[1], '*r')
        subplot.plot(goal[0], goal[1], '*r')
    # 障碍物设置及绘制
    obs = [[1, 4], [2, 4], [3, 3], [6, 1], [6, 7], [10, 6], [11, 12], [14, 14]]
    print('obstacles: {0}'.format(obs))
    for i in range(0):
        obs.append([random.uniform(2, goal[1] - 1), random.uniform(2, goal[1] - 1)])

    if is_plot:
        for OB in obs:
            circle = Circle(xy=(OB[0], OB[1]), radius=rr, alpha=0.3)
            subplot.add_patch(circle)
            subplot.plot(OB[0], OB[1], 'xk')
    # t1 = time.time()
    # for i in range(1000):

    a_desired = (v_desired - car_state['v']) / dt
    alpha_desired = (omega_desired - car_state['omega']) / dt

    return a_desired, alpha_desired

def compute_h(car_state, obstacle, l):
    x = car_state['x']                                                     # Extract car state
    y = car_state['y']
    theta = car_state['theta']
    v = car_state['v']
    omega = car_state['omega']

    x_o = obstacle['x']                                                    # Extract obstacle state
    y_o = obstacle['y']
    v_o = obstacle['v']
    theta_o = obstacle['theta']
    R_o = obstacle['r']

    rel_pos = np.array([                                                   # Compute relative position and velocity
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
    x_val = car_state['x']                                                 # Extract car state numeric values
    y_val = car_state['y']
    theta_val = car_state['theta']
    v_val = car_state['v']
    omega_val = car_state['omega']

    x = ca.MX.sym('x')                                                     # Define symbolic variables for state (parameters)
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')
    v = ca.MX.sym('v')
    omega = ca.MX.sym('omega')

    p = ca.vertcat(x, y, theta, v, omega)                                  # Collect all parameters
    param_values = [x_val, y_val, theta_val, v_val, omega_val]

    a_opt = ca.MX.sym('a_opt')                                              # Define optimization variables
    alpha_opt = ca.MX.sym('alpha_opt')
    x_opt = ca.vertcat(a_opt, alpha_opt)

    constraints = []

    for idx, obs in enumerate(obstacles):
        x_o_val = obs['x']                                                # Extract obstacle state numeric values
        y_o_val = obs['y']
        v_o_val = obs['v']
        theta_o_val = obs['theta']
        R_o_val = obs['r']

        x_o = ca.MX.sym(f"x_o_{idx}")                                      # Define symbolic variables for obstacle parameters
        y_o = ca.MX.sym(f"y_o_{idx}")
        v_o = ca.MX.sym(f"v_o_{idx}")
        theta_o = ca.MX.sym(f"theta_o_{idx}")
        R_o = ca.MX.sym(f"R_o_{idx}")

        p = ca.vertcat(p, x_o, y_o, v_o, theta_o, R_o)                     # Add obstacle parameters to the parameter list
        param_values.extend([x_o_val, y_o_val, v_o_val, theta_o_val, R_o_val])

        rel_pos = ca.vertcat(x_o - x - l * ca.cos(theta),                  # Compute relative position and velocity symbolically
                             y_o - y - l * ca.sin(theta))
        rel_vel = ca.vertcat(
            v_o * ca.cos(theta_o) - v * ca.cos(theta) + l * ca.sin(theta) * omega,
            v_o * ca.sin(theta_o) - v * ca.sin(theta) - l * ca.cos(theta) * omega
        )

        epsilon = 1e-6                                                     # Avoid division by zero
        norm_rel_pos = ca.norm_2(rel_pos) + epsilon
        norm_rel_vel = ca.norm_2(rel_vel) + epsilon

        sqrt_term = ca.sqrt(norm_rel_pos**2 - (R_o+0.5)**2 + epsilon)      # Collision cone condition
        cos_phi = sqrt_term / norm_rel_pos

        h = ca.dot(rel_pos, rel_vel) + norm_rel_pos * norm_rel_vel * cos_phi  # Control Barrier Function h

        x_dot = v * ca.cos(theta)                                          # State derivatives
        y_dot = v * ca.sin(theta)
        theta_dot = omega
        v_dot = a_opt
        omega_dot = alpha_opt

        state_vars = ca.vertcat(x, y, theta, v, omega)                     # Stack state variables and their derivatives
        state_dots = ca.vertcat(x_dot, y_dot, theta_dot, v_dot, omega_dot)

        h_dot = ca.jtimes(h, state_vars, state_dots)                       # Compute h_dot

        constraints.append(h_dot + gamma * h)                              # Append the constraint for this obstacle

    cost = (a_opt - a_d) ** 2 + (alpha_opt - alpha_d) ** 2                 # Define cost

    nlp = {'x': x_opt, 'f': cost, 'g': ca.vertcat(*constraints), 'p': p}   # Build nlp

    solver = ca.nlpsol('solver', 'ipopt', nlp, {                           # Create solver
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.tol': 1e-4,
        'ipopt.max_iter': 1000,
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.acceptable_obj_change_tol': 1e-3,
    })

    try:                                                                   # Solve the optimization problem
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
    car = Car(x=start[0], y=start[1], theta=np.pi / 4)                    # Adjusted initial theta

    obstacles = [                                                          # Define multiple obstacles
        {'x': 0.0, 'y': 0.0, 'v': 0.0, 'theta': 0.0, 'r': 0.5},
        {'x': 3.5, 'y': 3.5, 'v': 0.0, 'theta': 0.0, 'r': 0.5},           # New obstacle
    ]

    x_ref_traj, y_ref_traj = generate_reference_trajectory(start, goal)

    car_x_traj = []                                                        # Initialize lists to store car positions and variables
    car_y_traj = []
    time_steps = []
    velocities = []
    accelerations = []
    h_values = []                                                          # List to store h values when CBF is active
    h_time_steps = []                                                      # List to store time steps when h is computed
    angular_velocities = []
    angular_accelerations = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    car_plot, = ax.plot([], [], 'bo', label='Car')                         # Current position
    car_traj_plot, = ax.plot([], [], 'b-', label='Car Trajectory')         # Trajectory
    trajectory_plot, = ax.plot(x_ref_traj, y_ref_traj, 'g--', label='Reference Trajectory')

    for obs in obstacles:                                                  # Plot obstacles
        obstacle_patch = plt.Circle((obs['x'], obs['y']), obs['r'], color='r', alpha=0.5)
        ax.add_patch(obstacle_patch)
    ax.legend()
    plt.title("PD with CBF and Multiple Obstacles")

    for step in range(num_steps):
        current_time = step * dt
        current_waypoint = min(step, len(x_ref_traj) - 1)
        x_ref = x_ref_traj[current_waypoint]
        y_ref = y_ref_traj[current_waypoint]

        car_state = {'x': car.x, 'y': car.y, 'theta': car.theta, 'v': car.v, 'omega': car.omega}
        a_pd, alpha_pd = pd_controller(car_state, x_ref, y_ref, dt)

        cbf_active = False                                                 # Determine if CBF should be active based on proximity to any obstacle
        for obs in obstacles:
            if np.hypot(car.x - obs['x'], car.y - obs['y']) < (obs['r'] + 2.0):
                cbf_active = True
                break

        if cbf_active:
            a_opt, alpha_opt = solve_collision_cone_cbf(a_pd, alpha_pd, car_state, obstacles, gamma, dt, l)
            for obs in obstacles:                                          # Compute h values for obstacles when CBF is active
                h_value = compute_h(car_state, obs, l)
                h_values.append(h_value)
                h_time_steps.append(current_time)
        else:
            a_opt, alpha_opt = a_pd, alpha_pd

        car.update_dynamics(a_opt, alpha_opt, dt)                          # Update car dynamics

        car_x_traj.append(car.x)                                           # Append current car position and variables to lists
        car_y_traj.append(car.y)
        time_steps.append(current_time)
        velocities.append(car.v)
        accelerations.append(a_opt)
        angular_velocities.append(car.omega)
        angular_accelerations.append(alpha_opt)

        car_plot.set_data([car.x], [car.y])                                    # Update visualization
        car_traj_plot.set_data([car_x_traj], [car_y_traj])
        trajectory_plot.set_data(x_ref_traj[:current_waypoint + 1], y_ref_traj[:current_waypoint + 1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if np.hypot(car.x - goal[0], car.y - goal[1]) < 0.2:               # Check if goal is reached
            print("Goal reached at step:", step)
            break

    plt.ioff()

    # plt.savefig('final_trajectory.png')                                    # Save the final trajectory figure

    plt.show()

    # plt.figure()                                                           # Plot Velocity vs Time Steps
    # plt.plot(time_steps, velocities, label='Velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Linear Velocity (m/s)')
    # plt.title('Velocity vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    # # plt.savefig('velocity_vs_time.png')
    # plt.show()

    # plt.figure()                                                           # Plot Acceleration vs Time Steps
    # plt.plot(time_steps, accelerations, label='Acceleration')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Linear Acceleration (m/s²)')
    # plt.title('Acceleration vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    # # plt.savefig('acceleration_vs_time.png')
    # plt.show()

    # if h_values:                                                           # Plot h vs Time Steps when CBF is active
    #     plt.figure()
    #     plt.plot(h_time_steps, h_values, label='Barrier Function h')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('h Value')
    #     plt.title('Barrier Function h vs Time Steps (CBF Active)')
    #     plt.legend()
    #     plt.grid(True)
    #     # plt.savefig('h_vs_time.png')
    #     plt.show()
    # else:
    #     print("CBF was not active during the simulation; no h values to plot.")

    # plt.figure()                                                           # Plot Angular Velocity vs Time Steps
    # plt.plot(time_steps, angular_velocities, label='Angular Velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angular Velocity (rad/s)')
    # plt.title('Angular Velocity vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    # # plt.savefig('angular_velocity_vs_time.png')
    # plt.show()

    # plt.figure()                                                           # Plot Angular Acceleration vs Time Steps
    # plt.plot(time_steps, angular_accelerations, label='Angular Acceleration')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angular Acceleration (rad/s²)')
    # plt.title('Angular Acceleration vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    # # plt.savefig('angular_acceleration_vs_time.png')
    # plt.show()

if __name__ == "__main__":
    run_simulation()
