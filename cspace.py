import numpy as np
import matplotlib.pyplot as plt
import math
from arm import forward_kinematics, is_collision
from rrt import rrt
from matplotlib.animation import FuncAnimation

def animate_path(path, obstacles):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # left plot — real world
    ax1.set_xlim(-160, 160)
    ax1.set_ylim(-160, 160)
    ax1.set_aspect('equal')
    ax1.set_title('real world')
    ax1.grid(True, alpha=0.2)
    
    # draw obstacle
    obs_center, obs_radius = obstacles[0]
    circle = plt.Circle(obs_center, obs_radius, color='#D85A30', alpha=0.4)
    ax1.add_patch(circle)
    
    # arm lines
    link1, = ax1.plot([], [], 'g-', linewidth=4, solid_capstyle='round')
    link2, = ax1.plot([], [], 'b-', linewidth=3, solid_capstyle='round')
    joint,  = ax1.plot([], [], 'ko', markersize=6)
    
    # right plot — c-space
    ax2.imshow(grid, origin='lower', 
               extent=[-math.pi, math.pi, -math.pi, math.pi], 
               cmap='RdYlGn_r')
    t1s = [p[0] for p in path]
    t2s = [p[1] for p in path]

    # split path into segments at wraparound points
    segments_t1 = [[]]
    segments_t2 = [[]]
    for i in range(len(t1s)):
        segments_t1[-1].append(t1s[i])
        segments_t2[-1].append(t2s[i])
        if i < len(t1s) - 1:
            dt1 = abs(t1s[i+1] - t1s[i])
            dt2 = abs(t2s[i+1] - t2s[i])
            if dt1 > math.pi or dt2 > math.pi:
                segments_t1.append([])
                segments_t2.append([])

    for seg_t1, seg_t2 in zip(segments_t1, segments_t2):
        ax2.plot(seg_t1, seg_t2, 'b-', linewidth=1.5, alpha=0.5)

    dot, = ax2.plot([], [], 'wo', markersize=8)

    def update(frame):
        t1, t2 = path[frame]
        elbow, tip = forward_kinematics(t1, t2)
        origin = np.array([0.0, 0.0])
        
        # clear and redraw left plot
        ax1.cla()
        ax1.set_xlim(-160, 160)
        ax1.set_ylim(-160, 160)
        ax1.set_aspect('equal')
        ax1.set_title('real world')
        ax1.grid(True, alpha=0.2)
        
        # redraw obstacle
        circle = plt.Circle(obs_center, obs_radius, color='#D85A30', alpha=0.4)
        ax1.add_patch(circle)
        
        # redraw arm
        ax1.plot([origin[0], elbow[0]], [origin[1], elbow[1]], 'g-', linewidth=4, solid_capstyle='round')
        ax1.plot([elbow[0], tip[0]], [elbow[1], tip[1]], 'b-', linewidth=3, solid_capstyle='round')
        ax1.plot([origin[0], elbow[0]], [origin[1], elbow[1]], 'ko', markersize=6)
        
        # update dot on c-space
        dot.set_data([t1], [t2])
        fig.canvas.draw_idle()
        return dot,
    
    frame = [0]
    paused = [False]

    def on_key(event):
        if event.key == ' ':
            if paused[0]:
                ani.resume()
            else:
                ani.pause()
            paused[0] = not paused[0]
        elif event.key == 'right' and paused[0]:
            frame[0] = min(frame[0] + 1, len(path) - 1)
            update(frame[0])
            fig.canvas.draw()
        elif event.key == 'left' and paused[0]:
            frame[0] = max(frame[0] - 1, 0)
            update(frame[0])
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = FuncAnimation(fig, update, frames=len(path), 
                        interval=50, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()

OBSTACLE = [(np.array([60.0, -40.0]), 35.0)]
N = 200

t1_vals = np.linspace(-math.pi, math.pi, N)
t2_vals = np.linspace(-math.pi, math.pi, N)

grid = np.zeros((N, N))

for i, t1 in enumerate(t1_vals):
    for j, t2 in enumerate(t2_vals):
        if is_collision(t1, t2, OBSTACLE):
            grid[j, i] = 1.0

start = np.array([-2.0, -2.0])
goal = np.array([2.0, 2.0])

path = rrt(start, goal, OBSTACLE, max_iter=10000, step_size=0.1)

if path is None:
    print("no path found")
else:
    animate_path(path, OBSTACLE)