# CBF-for-obstacle-avoidance
This project implements a simulation framework for trajectory planning and collision avoidance of a car-like robot. It integrates Proportional-Derivative (PD) control with Control Barrier Functions (CBFs) to ensure safe navigation in the presence of static and dynamic obstacles. 
# Features
a. **Car Dynamics Simulation**: Simulates a car's state evolution based on control inputs, considering linear and angular velocities.

b. **Reference Trajectory Generation**: Creates a smooth trajectory between start and goal positions.

c. **PD Controller**: Provides basic trajectory tracking by calculating linear and angular accelerations.

d. **Control Barrier Function (CBF)**: Ensures collision-free motion by solving an optimization problem to maintain safety constraints.

e. **Obstacle Avoidance**: Dynamically adjusts control inputs to avoid collisions using collision cone principles.

f. **Visualizations: Plots**:
1. Trajectory of the car and reference path
2. Linear velocity and acceleration vs. time
3. Angular velocity and acceleration vs. time
4. Barrier function values when active
  
g. **Simulation Logs**: Includes detailed logging and visualization of car state and performance metrics.

# Technologies

a. **Python Libraries**:
1. numpy for mathematical computations.
2. matplotlib for trajectory visualization and performance plotting.
3. casadi for symbolic computation and solving optimization problems.

b. **Control Theory**: Implements PD control and CBF-based safety constraints.

# How to Run
a. **Install dependencies**: numpy, matplotlib, and casadi.

b. **Run the script**: python <script_name>.py.

c. **Generated plots and a trajectory simulation visualization will be saved as .png files.**

# Results
![final_trajectory_dynamic](https://github.com/user-attachments/assets/d7972b9f-7ccd-44cd-bff3-e075c75aa315)

# References
a. A Collision Cone Approach for Control Barrier Functions, Manan Tayal and Bhavya Giri Goswami and Karthik Rajgopal and Rajpal Singh and Tejas Rao and Jishnu Keshavan and Pushpak Jagtap and Shishir Kolathaya, 2024, 2403.07043, https://arxiv.org/abs/2403.07043


