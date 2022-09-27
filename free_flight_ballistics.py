import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

from consts import *
from tf_tools import get_H, get_r, build_M_s2c, get_phi, get_lamda


# 二维的时候让x=0, lambda=pi/2， 用yz平面

def build_impact_event():
    def impact_event(t, state):
        p_c = state[:3]
        return get_H(p_c)
    
    impact_event.terminal = True
    impact_event.direction = -1
    return impact_event


def free_flight_state_function(t, state):  # state 在地心坐标系中, 横向量
    p_c = state[:3]
    v_c = state[3:6]
    
    # 重力 !! 注意当地水平面坐标系OY为低些过飞行系质心朝上，OX指向正东，所以是 [0, -g, 0].T
    g_s = np.array([0, - g_CONST * (RADIUS_EARTH / get_r(p_c)) ** 2, 0]).T
    M_s2c = build_M_s2c(p_c)
    g_c = np.matmul(M_s2c, g_s)
    
    # 牵连惯性力 convected  inertial force
    phi = get_phi(p_c)
    lamda = get_lamda(p_c)
    f_cif = - get_r(p_c) * OMEGA_EARTH ** 2 * \
            np.cos(phi) * np.array([np.cos(lamda),
                                    np.sin(lamda),
                                    0]).T
    # 科里奥利力 Coriolis force
    f_cor = 2 * OMEGA_EARTH * np.array([v_c[1],
                                        -v_c[0],
                                        0]).T  # [vyc, -vxc, 0]
    
    # 推力
    fT_c = np.zeros((1, 3))
    # 空气气动力
    fA_c = np.zeros((1, 3))
    
    # acc_c = g_c + f_cif + f_cor + fT_c + fA_c
    acc_c = g_c
    
    d_states = np.concatenate([v_c, acc_c.flatten()], axis=0)
    return d_states


def get_free_flight_ball_traj(initial_state, t_max=3.6e3, t_step=10):
    solver = solve_ivp(fun=free_flight_state_function,
                       t_span=[0, t_max],
                       y0=initial_state,
                       t_eval=np.arange(0, t_max, t_step),
                       events=build_impact_event(),
                       method='RK45')
    return {
        "t": solver.t,
        "p_c": solver.y[:3, :],
        "v_c": solver.y[:3, :],
    }


def draw_ball_traj(traj, fig=None):
    t = traj["t"]
    p_c = traj["p_c"]
    v_c = traj["v_c"]
    
    fig = fig or plt.figure(figsize=(8, 4.5), facecolor='#303030')
    gs = GridSpec(1, 2, figure=fig)
    
    # 2D跟随视角  Y-Z 轴
    x_view, y_view = p_c[1, :], p_c[2, :]
    
    ax_2d = fig.add_subplot(gs[0, 0])
    ax_2d.set_facecolor('#202020')
    ax_2d.xaxis.set_visible(False)
    ax_2d.yaxis.set_visible(False)
    ax_2d.spines['bottom'].set_color('#303030')
    ax_2d.spines['top'].set_color('#303030')
    ax_2d.spines['right'].set_color('#303030')
    ax_2d.spines['left'].set_color('#303030')
    
    _earth = plt.Circle((0, 0), RADIUS_EARTH, color='cyan')
    # ax_2d.add_patch(earth)
    
    ax_2d.set_aspect('equal', 'box')
    
    line, = ax_2d.plot(x_view[0], y_view[0], color='silver', linestyle='-',
                       linewidth=1)
    dot, = ax_2d.plot([], [], color='silver', marker='o', markersize=1,
                      markeredgecolor='w', linestyle='')
    
    # ax_2d.set_xlim([-1.2e7, 1.2e7])
    # ax_2d.set_ylim([-1.2e7, 1.2e7])
    
    plt.tight_layout()
    
    def update(i):
        dot.set_data(x_view[i], y_view[i])
        line.set_data(x_view[:i], y_view[:i])
        ax_2d.set_xlim([x_view[i] - 3e5, x_view[i] + 3e5])
        ax_2d.set_ylim([y_view[i] - 3e5, y_view[i] + 3e5])
        earth = ax_2d.add_patch(_earth)
        return [dot, line, earth]
    
    ani = FuncAnimation(fig=fig,
                        func=update,
                        frames=len(t),
                        interval=t[1] - t[0],
                        blit=True,
                        repeat=False)
    plt.show()
    
    # 3D视角  Y-Z 轴


if __name__ == '__main__':
    # initial_state = np.array(
    #     [0, RADIUS_EARTH,  0, # p_c
    #      0, 5 * SOUND_SPEED, 0, ])  # v_c
    
    initial_states = []
    ball_trajs = []
    for speed_factor in tqdm.tqdm(np.arange(8.5, 9, 0.5)):
        initial_state = np.array(
            [0, RADIUS_EARTH + 400 * 1000, 0,  # p_c
             0, SOUND_SPEED / np.sqrt(2),
             speed_factor * SOUND_SPEED / np.sqrt(2)])  # v_c
        initial_states.append(initial_state)
    
    for initial_state in initial_states:
        ball_traj = get_free_flight_ball_traj(initial_state)
        ball_trajs.append(ball_traj)
    
    for traj in ball_trajs:
        draw_ball_traj(traj)
