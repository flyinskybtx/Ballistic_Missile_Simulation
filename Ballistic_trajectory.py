# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 01:58:45 2022

@author: juliu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# Fundamental Constants
GRAV_CONST = 6.67e-11  # Gravitational Constant
MASS_EARTH = 5.972e24  # kg
RADIUS_EARTH = 6371000  # m
g_const = 9.81  # m/s^2
OMEGA_EARTH = 7.2921159e-5  # rad/s

# Missile
mass_launch = 6200  # kg, launch mass
mass_empty_rocket = 1600  # kg, mass of war head + empty rocket
mass_warhead = 800  # kg, mass of war head
isp = 237  # s, SRB general number todo： ？
v_e = isp * g_const  # m/s
TTW_const = 1.5  # thrust to weight
thrust_force = mass_launch * g_const * TTW_const  # thrust, in N
drag_coefficient = 2e-1  # drag coefficient, dimensionless
A = np.pi * 0.5 ** 2  # cross section, m^2
ELEVATION_ANGLE = 75  # angle of elevation, in deg

# Atmospheric constants
surface_temperature = 273.15 + 15  # K, surface temperature
surface_pressure = 1013e2  # Pa, surface pressure
gas_const = 8.314  # J/K/mol, ideal gas constant
temperature_lapse_rate_const = 0.0065  # K/m, temperature lapse rate
molecular_weight_const = 0.029  # kg/mol, molecular weight of the air

# %%

# Simulation properties
max_t = 3.6e3  # Simulation time in second, set to 60 min for maximum

# Visualization properties
plot_size = 1.2e7  # Size of the plot
num_frames = int(2100)  # Output frames
Tracing = True  # Viewing the sail with tracing mode.
SAVE_VIDEO = False  # Whether you want to save the video


# %%
def initial_condition():
    def launch_parameter():
        return [0, RADIUS_EARTH, 0, 100, mass_launch]  # x, y, vx, vy, m
        # 可以看出，原点是地心
    
    return launch_parameter()


initial_states = initial_condition()


def rotate(vector, theta):  # in degree
    theta = theta / 180 * np.pi
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return np.matmul(rotation_matrix, vector)


def air_density(h):
    # Here we are modeling air density by extrapolating Troposphere condition
    if h < 4.4e4:  # 44 km
        rho = surface_pressure * molecular_weight_const / gas_const / surface_temperature * (
                1 - temperature_lapse_rate_const * h / surface_temperature) ** (
                      g_const * molecular_weight_const / gas_const / temperature_lapse_rate_const - 1)
    else:
        rho = 0
    return rho


"""%% Altitude control
This is the tricky part, lots of freedom included.
Here we use:
   1. When the missile is still in a dense atmosphere, we let it accelerate
      in a direction similar to rhat (slight deviation from going straight up)
   2. When above 10 km, we let the acceleration align with the velocity vector. """


def calc_pointing(t, x, y, vx, vy):
    p = np.array([x, y])
    v = np.array([vx, vy])
    p_pointing = p / np.linalg.norm(p)  # 原点（地心）到 p 的方向向量
    if vx == 0 and vy == 0:
        v_pointing = p_pointing
    else:
        v_pointing = v / np.linalg.norm(v)
    
    if np.linalg.norm(
            p) > RADIUS_EARTH + 1e4:  # above 10 km, align with velocity
        phat = v_pointing
    else:
        if np.dot(p, v) > 0:  # 向下飞
            phat = rotate(p_pointing, ELEVATION_ANGLE - 90)
        else:  # 向上飞
            phat = v_pointing
    return phat


# %% Here are the function for ivp solver
def impact_event(t, y):
    return np.sqrt(y[0] ** 2 + y[1] ** 2) - RADIUS_EARTH


impact_event.terminal = True
impact_event.direction = -1


def state_function(t, states):
    """
    
    :param t:
    :param states: np.array([x, y, vx, vy, 剩余总重 m])
    :return: d_states: np.array([x, y, vx, vy, dm])
    """
    separation = False
    p_vec = states[:2]
    radius = np.linalg.norm(p_vec)  # 到地心距离
    v_vec = states[2:4]
    v_norm = np.linalg.norm(v_vec)
    phat = calc_pointing(t, states[0], states[1], states[2], states[3])
    dxdt = v_vec[0]
    dydt = v_vec[1]
    
    # This is for thrust, only exist when there are still enough fuel
    if states[4] > mass_empty_rocket:
        acc_thrust = thrust_force / states[4] * phat
        dmdt = -thrust_force / v_e
    else:
        separation = True
        acc_thrust = 0
        dmdt = 0
    
    # Gravity
    acc_grav = -GRAV_CONST * MASS_EARTH / radius ** 3 * p_vec
    
    # Air drag
    if separation:  # descending
        acc_air = -0.5 * air_density(
            radius - RADIUS_EARTH) * v_norm * drag_coefficient * (
                          0.5 * A) / mass_warhead * v_vec
    else:  # ascending
        acc_air = -0.5 * air_density(
            radius - RADIUS_EARTH) * v_norm * drag_coefficient * A / \
                  states[4] * v_vec
    
    # Centrifugal force
    acc_centrifugal = OMEGA_EARTH ** 2 * p_vec  # 离心加速度
    
    # Coriolis force
    acc_coriolis = 2 * OMEGA_EARTH * np.array([v_vec[1], -v_vec[0]])
    
    # Total acceleration
    acc = acc_thrust + acc_grav + acc_air + acc_centrifugal + acc_coriolis
    dvxdt = acc[0]
    dvydt = acc[1]
    
    d_states = np.array([dxdt, dydt, dvxdt, dvydt, dmdt])
    return d_states


# %%
# Solving the orbit
sol = solve_ivp(fun=state_function,
                t_span=[0, max_t],
                y0=initial_states,
                t_eval=np.linspace(0, max_t, num_frames),
                events=impact_event,
                method='DOP853')

t = sol.t
Data = sol.y
r = np.sqrt(Data[0, :] ** 2 + Data[1, :] ** 2)
x = Data[0, :]
y = Data[1, :]
vx = Data[2, :]
vy = Data[3, :]
m = Data[4, :]

# %%
# Visualization Setup
COLOR = '#303030'
LineColor = 'silver'
fig = plt.figure(figsize=(8, 4.5), facecolor=COLOR)
gs = GridSpec(2, 4, figure=fig)

# Picture
ax = fig.add_subplot(gs[:, :2])
ax.set_facecolor('#202020')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['bottom'].set_color(COLOR)
ax.spines['top'].set_color(COLOR)
ax.spines['right'].set_color(COLOR)
ax.spines['left'].set_color(COLOR)

earth = plt.Circle((0, 0), RADIUS_EARTH, color='cyan')
ax.add_patch(earth)
ax.set_aspect('equal', 'box')

line, = ax.plot(x[0], y[0], color='silver', linestyle='-', linewidth=1)
dot, = ax.plot([], [], color='silver', marker='o', markersize=1,
               markeredgecolor='w', linestyle='')

ax.set_xlim([-plot_size, plot_size])
ax.set_ylim([-plot_size, plot_size])
# %%
# Velocity Plot
ax1 = fig.add_subplot(gs[0, 2:])
ax1.set_facecolor(COLOR)
velline, = ax1.plot(t[0], np.sqrt(vx[0] ** 2 + vy[0] ** 2), color='silver')
ax1.spines['bottom'].set_color(LineColor)
ax1.spines['top'].set_color(LineColor)
ax1.spines['right'].set_color(LineColor)
ax1.spines['left'].set_color(LineColor)
ax1.set_xlim([0, t[-1]])
ax1.set_ylim([0, np.max(np.sqrt(vx ** 2 + vy ** 2)) * 1.2])
ax1.tick_params(labelcolor=LineColor, labelsize='medium', width=3,
                colors=LineColor)
ax1.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(4, 5))
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Velocity (m/s)')
ax1.xaxis.label.set_color(LineColor)
ax1.yaxis.label.set_color(LineColor)

# height Plot
ax2 = fig.add_subplot(gs[1, 2:])
ax2.set_facecolor(COLOR)
r = np.sqrt(x ** 2 + y ** 2)
h = r - RADIUS_EARTH
heightline, = ax2.plot(t[0], h[0], color='silver')
ax2.spines['bottom'].set_color(LineColor)
ax2.spines['top'].set_color(LineColor)
ax2.spines['right'].set_color(LineColor)
ax2.spines['left'].set_color(LineColor)
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([0, np.max(h) * 1.2])
ax2.tick_params(labelcolor=LineColor, labelsize='medium', width=3,
                colors=LineColor)
ax2.ticklabel_format(style='sci', useMathText=True, scilimits=(4, 5))
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Height (m)')
ax2.xaxis.label.set_color(LineColor)
ax2.yaxis.label.set_color(LineColor)

# %% Suptitle

Range = RADIUS_EARTH * np.arccos(
    np.dot([x[0], y[0]], [x[-1], y[-1]]) / RADIUS_EARTH ** 2) / 1000  # in km
v_f = np.sqrt(vx[-2] ** 2 + vy[-2] ** 2) / 340  # in Mach
fig.suptitle(
    'Launch angle {:.0f} degree, Range: {:.0f} km, impact velocity M{:.1f}'.format(
        ELEVATION_ANGLE, Range, v_f), color='silver')

plt.tight_layout()
# %%
ms2AUyr = 86400 * 365 / 1.5e11


def update(i):
    dot.set_data(x[i], y[i])
    line.set_data(x[:i], y[:i])
    velline.set_data(t[:i], np.sqrt(vx[:i] ** 2 + vy[:i] ** 2))
    heightline.set_data(t[:i], h[:i])
    if Tracing:
        # ax.set_xlim([-1.5*r,1.5*r])
        # ax.set_ylim([-1.5*r,1.5*r])
        ax.set_xlim([x[i] - 1e5, x[i] + 1e5])
        ax.set_ylim([y[i] - 1e5, y[i] + 1e5])
        # ax.set_xlim([np.min(x)-1e5,np.max(x)+1e5])
        # ax.set_ylim([np.min(y)-1e5,np.max(y)+1e5])
    O1 = ax.add_patch(earth)
    if SAVE_VIDEO:
        print(i)
        fig.savefig('./images/{:04d}.jpg'.format(i), dpi=300)
    return [dot, line, velline, heightline, O1]


if SAVE_VIDEO:
    for i in range(len(t)):  # 考虑到提前结束，不能是num_frames，而是t长度
        update(i)
else:
    ani = FuncAnimation(fig=fig,
                        func=update,
                        frames=len(t),
                        interval=10000 / num_frames,
                        blit=True,
                        repeat=False)
    plt.show()
