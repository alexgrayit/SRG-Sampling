import numpy as np
import lcm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import plotting
import time
import control
import LTI_SRG as srg

from lcm_messages.simulation_messages import SRGState
from lcm_messages.simulation_messages import startSim

import threading

k_d = 1.0
k_p = 1.0

A_v = -k_d
B_v = 1
C_v = 1
D_v = 0
SS_v = control.ss(A_v, B_v, C_v, D_v)

A_x = np.array([[0, 1], [-k_p, -k_d]])
B_x = np.array([[0], [1]])
C_x = np.array([[1, 0]])
D_x = np.array([[0]])
SS_x = control.ss(A_x, B_x, C_x, D_x)



MAX_POINTS = 100000
x_xs = deque(maxlen=MAX_POINTS)
x_ys = deque(maxlen=MAX_POINTS)
v_xs = deque(maxlen=MAX_POINTS)
v_ys = deque(maxlen=MAX_POINTS)
data_lock = threading.Lock()

global fig_x, ax_x, line_x, srg_x_xmin, srg_x_xmax, srg_x_ymin, srg_x_ymax
global fig_v, ax_v, line_v, srg_v_xmin, srg_v_xmax, srg_v_ymin, srg_v_ymax

plt.ion()

fig_x, ax_x = plt.subplots()
(line_x,) = ax_x.plot([], [], 'o', markersize=3)

ax_x.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
# control.nyquist_plot(SS_x)
srg.srg_lti(control.tf(SS_x), shade=True, show_lower=True, ax=ax_x)


fig_v, ax_v = plt.subplots()
(line_v,) = ax_v.plot([], [], 'o', markersize=3)

ax_v.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
# control.nyquist_plot(SS_v)
srg.srg_lti(control.tf(SS_v), shade=True, show_lower=True, ax=ax_v)


srg_x_xmin, srg_x_xmax = ax_x.get_xlim()
srg_x_ymin, srg_x_ymax = ax_x.get_ylim()

srg_v_xmin, srg_v_xmax = ax_v.get_xlim()
srg_v_ymin, srg_v_ymax = ax_v.get_ylim()


def srg_handler(channel, data):
    msg = SRGState.decode(data)
    # print(f"Received state: {msg}")
    L2_x = msg.L2_x
    L2_v = msg.L2_v
    x_norm = msg.x_norm
    v_norm = msg.v_norm
    u_norm = msg.u_norm

    # print(f"L2_x: {L2_x}, L2_v: {L2_v}, x_norm: {x_norm}, v_norm: {v_norm}, u_norm: {u_norm}")

    theta_x = np.arccos(L2_x / (x_norm * u_norm))
    theta_v = np.arccos(L2_v / (v_norm * u_norm))

    gain_x = x_norm / u_norm
    gain_v = v_norm / u_norm

    x_xs.append(gain_x * np.cos(theta_x))
    x_ys.append(gain_x * np.sin(theta_x))

    x_xs.append(gain_x * np.cos(-theta_x))
    x_ys.append(gain_x * np.sin(-theta_x))

    v_xs.append(gain_v * np.cos(theta_v))
    v_ys.append(gain_v * np.sin(theta_v))

    v_xs.append(gain_v * np.cos(-theta_v))
    v_ys.append(gain_v * np.sin(-theta_v))


def update_plot_x(frame):
    global fig_x, ax_x, line_x, srg_x_xmin, srg_x_xmax, srg_x_ymin, srg_x_ymax

    if len(x_xs) == 0:
        return line_x,
    with data_lock:
        # ax_x.clear()

        x = np.array(x_xs)
        y = np.array(x_ys)
    
    # print(f"Updating plot with {len(x)} points")
    # print(f"Points = {x}, {y}")
    line_x.set_data(x, y)
    if len(x) > 0:
        # x_bounds = (min(x) - 0.01, max(x) + 0.01)
        # y_bounds = (min(y) - 0.01, max(y) + 0.01)
        # print(f"Setting x limits to {x_bounds} and y limits to {y_bounds}")
        ax_x.set_xlim(min(min(x), srg_x_xmin) - 0.01, max(max(x), srg_x_xmax) + 0.01)
        ax_x.set_ylim(min(min(y), srg_x_ymin) - 0.01, max(max(y), srg_x_ymax) + 0.01)

    fig_x.canvas.draw()
    fig_x.canvas.flush_events()
    ax_x.set_aspect('equal')
    return line_x,
        

def update_plot_v(frame):
    global fig_v, ax_v, line_v, srg_v_xmin, srg_v_xmax, srg_v_ymin, srg_v_ymax

    if len(v_xs) == 0:
        return line_v,
    with data_lock:
        # ax_v.clear()

        x = np.array(v_xs)
        y = np.array(v_ys)
    
    line_v.set_data(x, y)
    
    if len(x) > 0:
        ax_v.set_xlim(min(min(x), srg_v_xmin) - 0.01, max(max(x), srg_v_xmax) + 0.01)
        ax_v.set_ylim(min(min(y), srg_v_ymin) - 0.01, max(max(y), srg_v_ymax) + 0.01)

    fig_v.canvas.draw_idle()
    fig_v.canvas.flush_events()
    ax_v.set_aspect('equal')
    return line_v,


def lcm_thread():
    lc = lcm.LCM()
    subscription = lc.subscribe("SRGState", srg_handler)

    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    lc = lcm.LCM()
    subscription = lc.subscribe("SRGState", srg_handler)

    startMsg = startSim()
    startMsg.sim_name = "drag_sim"
    lc.publish("startSim", startMsg.encode())

    t = threading.Thread(target=lcm_thread, daemon=True)
    t.start()

    while True:
        update_plot_x(None)
        update_plot_v(None)
        # time.sleep(0.0001)
