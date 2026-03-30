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
        # hint: np.linalg.norm to measure distance between two configs
        minDist = float('inf')
        nearestNode = start
        for node in nodes:
            dist = torus_distance(point, node)
            if dist < minDist:
                minDist = dist
                nearestNode = node

        # step 3: move from nearest node towards point by step_size
        direction = point - nearestNode
        translation = direction / np.linalg.norm(direction) * step_size
        newNode = nearestNode + translation
        newNode = ((newNode + math.pi) % (2 * math.pi)) - math.pi
        # step 4: if new node is in collision, discard it and continue
        if not is_collision(newNode[0], newNode[1], obstacles):
            nodes.append(newNode)
            parent[tuple(newNode)] = tuple(nearestNode)

        # step 5: if new node is close enough to goal, construct path and return it
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

