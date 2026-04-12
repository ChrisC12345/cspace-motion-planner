# RRT path planning for 2-link arm in C-space

import numpy as np
import math
from arm import is_collision_batch

_TWO_PI = 2 * math.pi


def _torus_dist_sq(a, nodes):
    """Vectorized squared torus distance from point a to every row in nodes array."""
    diff = np.abs(nodes - a)
    diff = np.minimum(diff, _TWO_PI - diff)
    return diff[:, 0] ** 2 + diff[:, 1] ** 2


def _line_free(a, b, obstacles):
    """Return True if the straight arc a→b in C-space is collision-free."""
    diff = (b - a + math.pi) % _TWO_PI - math.pi
    n = max(4, int(np.linalg.norm(diff) / 0.05))
    ts = np.linspace(0, 1, n, endpoint=False)
    configs = ((a + ts[:, None] * diff) + math.pi) % _TWO_PI - math.pi
    return not np.any(is_collision_batch(configs[:, 0], configs[:, 1], obstacles))


def rrt(start, goal, obstacles, max_iter=5000, step_size=0.05):
    start = np.asarray(start, float)
    goal  = np.asarray(goal,  float)

    # pre-allocate storage — avoids repeated list reallocation
    nodes  = np.empty((max_iter + 2, 2))
    nodes[0] = start
    n_nodes  = 1
    parent   = [-1]   # parent[i] = index of parent, -1 for root

    GOAL_BIAS = 0.1

    for _ in range(max_iter):
        # sample: bias toward goal 10 % of the time
        point = goal if np.random.random() < GOAL_BIAS \
                     else np.random.uniform(-math.pi, math.pi, 2)

        # nearest neighbour — one vectorised numpy call instead of a Python loop
        nearest_idx = int(np.argmin(_torus_dist_sq(point, nodes[:n_nodes])))
        nearest = nodes[nearest_idx]

        # steer toward sample by step_size
        direction = (point - nearest + math.pi) % _TWO_PI - math.pi
        norm = math.hypot(direction[0], direction[1])
        if norm < 1e-10:
            continue
        new_node = ((nearest + direction * (step_size / norm)) + math.pi) % _TWO_PI - math.pi

        if _line_free(nearest, new_node, obstacles):
            nodes[n_nodes] = new_node
            parent.append(nearest_idx)
            n_nodes += 1

            # check if we reached the goal
            if _torus_dist_sq(goal, nodes[n_nodes-1:n_nodes])[0] < step_size ** 2:
                path = []
                idx = n_nodes - 1
                while idx >= 0:
                    path.append(tuple(nodes[idx]))
                    idx = parent[idx]
                path.reverse()
                return path

    return None


def smooth_path(path, obstacles, samples=10):
    path = [np.asarray(p, float) for p in path]
    i = 0
    while i < len(path) - 2:
        if _line_free(path[i], path[i + 2], obstacles):
            path.pop(i + 1)
        else:
            i += 1
    return [tuple(p) for p in path]
