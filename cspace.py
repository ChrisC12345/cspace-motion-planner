# visualize the c-space of a 2-link arm and animate a path through it

import numpy as np
import matplotlib.pyplot as plt
import math
from arm import forward_kinematics, is_collision
from obstacle import Obstacle, ObstacleType
from rrt import rrt, smooth_path
from matplotlib.animation import FuncAnimation

def draw_cspace(obstacles):
    N = 200
    t1_vals = np.linspace(-math.pi, math.pi, N)
    t2_vals = np.linspace(-math.pi, math.pi, N)
    grid = np.zeros((N, N))
    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            if is_collision(t1, t2, obstacles):
                grid[j, i] = 1.0
    return grid

def interpolate_path(path, resolution=0.05):
    dense_path = []
    for i in range(len(path) - 1):
        a = np.array(path[i])
        b = np.array(path[i+1])
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi  # shortest arc
        length = np.linalg.norm(diff)
        steps = max(2, int(length / resolution))
        for t in range(steps):
            config = a + (t / steps) * diff
            # wrap back into [-pi, pi]
            config = ((config + math.pi) % (2 * math.pi)) - math.pi
            dense_path.append(tuple(config))
    dense_path.append(path[-1])
    return dense_path

def animate_path(path, obstacles, title='path'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # left plot — real world
    ax1.set_xlim(-160, 160)
    ax1.set_ylim(-160, 160)
    ax1.set_aspect('equal')
    ax1.set_title('Real World')
    ax1.grid(True, alpha=0.2)
    
    # draw obstacles once
    for obstacle in obstacles:
        if obstacle.getType() == ObstacleType.CIRCLE:
            center, radius = obstacle.getParams()
            circle = plt.Circle(center, radius, color='#D85A30', alpha=0.4)
            ax1.add_patch(circle)
        elif obstacle.getType() == ObstacleType.POLYGON:
            vertices = obstacle.getParams()
            polygon = plt.Polygon(vertices, color='#D85A30', alpha=0.4)
            ax1.add_patch(polygon)
    
    # arm lines — initialized empty, updated each frame
    link1, = ax1.plot([], [], 'g-', linewidth=4, solid_capstyle='round')
    link2, = ax1.plot([], [], 'b-', linewidth=3, solid_capstyle='round')
    joint, = ax1.plot([], [], 'ko', markersize=6)

    start = forward_kinematics(path[0][0], path[0][1])
    goal = forward_kinematics(path[-1][0], path[-1][1])
    ax1.plot([0, start[0][0]], [0, start[0][1]], color='red', linewidth=2, solid_capstyle='round')
    ax1.plot([start[0][0], start[1][0]], [start[0][1], start[1][1]], color='red', linewidth=2, solid_capstyle='round')
    ax1.plot([0,goal[0][0]], [0, goal[0][1]], color='yellow', linewidth=2, solid_capstyle='round')
    ax1.plot([goal[0][0], goal[1][0]], [goal[0][1], goal[1][1]], color='yellow', linewidth=2, solid_capstyle='round')

    # right plot — c-space
    ax2.imshow(grid, origin='lower', 
               extent=[-math.pi, math.pi, -math.pi, math.pi], 
               cmap='RdYlGn_r')
    ax2.set_title('Configuration Space')
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

    # pre-compute all frames
    arm_frames = [forward_kinematics(t1, t2) for t1, t2 in path]
    

    def update(frame):
        t1, t2 = path[frame]
        elbow, tip = arm_frames[frame]
        origin = np.array([0.0, 0.0])
        
        link1.set_data([origin[0], elbow[0]], [origin[1], elbow[1]])
        link2.set_data([elbow[0], tip[0]], [elbow[1], tip[1]])
        joint.set_data([origin[0], elbow[0]], [origin[1], elbow[1]])
        dot.set_data([t1], [t2])
        #fig.canvas.draw_idle()
        return link1, link2, joint, dot
    
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

    # inside animate_path — remove fig.canvas.draw_idle() from update(), then:
    ani = FuncAnimation(fig, update, frames=len(path),
                        interval=20, blit=True, repeat=True,
                        cache_frame_data=False)
    plt.tight_layout()

def is_reachable(grid, start, goal, N=200):
    # convert configs to grid indices
    def to_idx(config):
        i = int((config[0] + math.pi) / (2 * math.pi) * N)
        j = int((config[1] + math.pi) / (2 * math.pi) * N)
        return np.clip(i, 0, N-1), np.clip(j, 0, N-1)
    
    si, sj = to_idx(start)
    gi, gj = to_idx(goal)
    
    # BFS flood fill through free cells
    from collections import deque
    visited = np.zeros((N, N), dtype=bool)
    queue = deque([(si, sj)])
    visited[si, sj] = True
    
    while queue:
        i, j = queue.popleft()
        if i == gi and j == gj:
            return True
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = (i+di) % N, (j+dj) % N  # toroidal wraparound
            if not visited[ni, nj] and grid[nj, ni] == 0:
                visited[ni, nj] = True
                queue.append((ni, nj))
    return False

OBSTACLE = [
    # Obstacle(ObstacleType.CIRCLE, (np.array([100.0, 0.0]), 10.0)),
    # Obstacle(ObstacleType.CIRCLE, (np.array([-100.0, 0.0]), 10.0))
    Obstacle(ObstacleType.POLYGON, [np.array([50.0, 50.0]), np.array([70.0, 50.0]), np.array([70.0, 70.0]), np.array([50.0, 70.0])]),
]

grid = draw_cspace(OBSTACLE)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, origin='lower',
          extent=[-math.pi, math.pi, -math.pi, math.pi],
          cmap='RdYlGn_r')
ax.set_xlabel('θ₁')
ax.set_ylabel('θ₂')
ax.set_title('click start (green) then goal (blue), press Enter to plan')

clicks = []
markers = []

def on_click(event):
    if event.inaxes != ax or len(clicks) >= 2:
        return
    clicks.append(np.array([event.xdata, event.ydata]))
    color = 'go' if len(clicks) == 1 else 'b*'
    marker, = ax.plot(event.xdata, event.ydata, color, markersize=12)
    markers.append(marker)
    fig.canvas.draw()

def on_key(event):
    if event.key == 'enter' and len(clicks) == 2:
        plt.close()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

if len(clicks) < 2:
    print("need to click start and goal")
else:
    start, goal = clicks[0], clicks[1]
    print(f"start: {start}, goal: {goal}")
    print("start in collision:", is_collision(start[0], start[1], OBSTACLE))
    print("goal in collision:", is_collision(goal[0], goal[1], OBSTACLE))

    if not is_reachable(grid, start, goal):
        print("no path exists — goal is not reachable from start")
        
    else:
        path = rrt(start, goal, OBSTACLE, max_iter=10000, step_size=0.2)

        if path is None:
            print("no path found")
        else:
            print("path length before smoothing:", len(path))
            smoothed = smooth_path(path.copy(), OBSTACLE, samples=4)
            print("path length after smoothing:", len(smoothed))
            animate_path(interpolate_path(path), OBSTACLE, title='raw RRT')
            animate_path(interpolate_path(smoothed), OBSTACLE, title='smoothed + interpolated')

plt.show()
