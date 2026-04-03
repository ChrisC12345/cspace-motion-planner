# RRT path planning for 2-link arm in C-space

import numpy as np
import random
import math
from arm import is_collision

def torus_distance(a, b):
    diff = np.abs(a - b)
    # for each dimension, the real distance is the shorter way around
    diff = np.minimum(diff, 2 * math.pi - diff)
    return np.linalg.norm(diff)

def rrt(start, goal, obstacles, max_iter=5000, step_size=0.05):
    # start and goal are (t1, t2) as np.arrays
    # returns a list of (t1, t2) waypoints from start to goal
    # or None if no path found within max_iter
    nodes = [start]
    parent = {tuple(start):None}
    for i in range(max_iter):
        # step 1: random point in C-space
        # t1 and t2 both range from -pi to pi
        point = np.array((random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi)))
        
        # step 2: find nearest node in tree
        minDist = float('inf')
        nearestNode = start
        for node in nodes:
            dist = torus_distance(point, node)
            if dist < minDist:
                minDist = dist
                nearestNode = node

        # step 3: move from nearest node towards point by step_size
        direction = point - nearestNode
        # take shortest arc in each dimension
        direction = (direction + math.pi) % (2 * math.pi) - math.pi
        translation = direction / np.linalg.norm(direction) * step_size
        newNode = nearestNode + translation
        newNode = ((newNode + math.pi) % (2 * math.pi)) - math.pi
        # step 4: check entire segment, not just endpoint
        if line_collision_free(tuple(nearestNode), tuple(newNode), obstacles, samples=4):
            nodes.append(newNode)
            parent[tuple(newNode)] = tuple(nearestNode)
            if torus_distance(newNode, goal) < step_size:
                path = []
                node = tuple(newNode)
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
    closest = min(nodes, key=lambda n: torus_distance(n, goal))
    print("closest node to goal:", torus_distance(closest, goal))     
    return None

def line_collision_free(a, b, obstacles, samples=10):
    a = np.array(a)
    b = np.array(b)
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    length = np.linalg.norm(diff)
    steps = max(10, int(length / 0.01))  # same resolution as interpolation
    for i in range(steps):
        t = i / steps
        config = a + t * diff
        config = ((config + math.pi) % (2 * math.pi)) - math.pi
        if is_collision(config[0], config[1], obstacles):
            return False
    return True

def smooth_path(path, obstacles, samples=10):
    # path is a list of (t1, t2) tuples
    # returns a shorter, smoother path
    i = 0
    while i < len(path) - 2:
        if line_collision_free(path[i], path[i+2], obstacles, samples):
            path.pop(i+1)  # remove middle point
            # don't advance i — try skipping again from same position
        else:
            i += 1  # can't skip, move forward
    return path