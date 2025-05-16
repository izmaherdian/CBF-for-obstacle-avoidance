import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def create_sphere(x_center, y_center, z_center, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = x_center + radius * np.cos(u) * np.sin(v)
    y = y_center + radius * np.sin(u) * np.sin(v)
    z = z_center + radius * np.cos(v)
    return x, y, z

def run_simulation(car, x_ref_traj, y_ref_traj, obstacles, dt=0.1, steps=1000):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    car_plot, = ax.plot3D([], [], [], 'bo', label='Car')
    car_traj_plot, = ax.plot3D([], [], [], 'b-', label='Car Trajectory')
    trajectory_plot, = ax.plot3D(x_ref_traj, y_ref_traj, [0]*len(x_ref_traj), 'g--', label='Reference Trajectory')

    car_x_traj = []
    car_y_traj = []
    car_z_traj = []

    current_waypoint = 0
    tolerance = 0.3
    ref_idx = 0

    # Draw obstacle spheres
    for obs in obstacles:
        x_s, y_s, z_s = create_sphere(obs['x'], obs['y'], 0, obs['r'])
        ax.plot_surface(x_s, y_s, z_s, color='r', alpha=0.3)

    def update(frame):
        nonlocal current_waypoint, ref_idx

        if current_waypoint >= len(x_ref_traj):
            return car_plot, car_traj_plot, trajectory_plot

        x_ref = x_ref_traj[current_waypoint]
        y_ref = y_ref_traj[current_waypoint]

        distance = np.hypot(x_ref - car.x, y_ref - car.y)

        if distance < tolerance and current_waypoint < len(x_ref_traj) - 1:
            current_waypoint += 1
            x_ref = x_ref_traj[current_waypoint]
            y_ref = y_ref_traj[current_waypoint]

        car.update(x_ref, y_ref, dt)

        car_x_traj.append(car.x)
        car_y_traj.append(car.y)
        car_z_traj.append(0)

        car_plot.set_data([car.x], [car.y])
        car_plot.set_3d_properties([0])

        car_traj_plot.set_data(car_x_traj, car_y_traj)
        car_traj_plot.set_3d_properties(car_z_traj)

        trajectory_plot.set_data(x_ref_traj[:current_waypoint+1], y_ref_traj[:current_waypoint+1])
        trajectory_plot.set_3d_properties([0]*(current_waypoint+1))

        return car_plot, car_traj_plot, trajectory_plot

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    ax.legend()
    plt.tight_layout()
    plt.show()
