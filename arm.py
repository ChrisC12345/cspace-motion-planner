# simple 2-link arm kinematics and collision checking
import math
import numpy as np
from enum import Enum
from obstacle import ObstacleType

L1 = 80
L2 = 60


def forward_kinematics(t1, t2):
    """returns (elbow, tip) as (x,y) tuples"""
    elbow = np.array([L1*math.cos(t1), L1*math.sin(t1)])
    angle = t1+t2
    elbowToTip = np.array([L2*math.cos(angle), L2*math.sin(angle)])
    return elbow, elbow + elbowToTip

def is_collision_circle(p1, p2, center, radius):
    """returns True if segment collides with circle"""
    d = p2 - p1
    t = np.dot(center-p1, d) / np.dot(d,d)
    if t < 0 :
        return np.linalg.norm(center-p1) < radius
    elif t > 1 :
        return np.linalg.norm(center-p2) < radius
    else:
        closest = p1 + t*d
        return np.linalg.norm(center-closest) < radius
    


def is_collision_segment(p1, p2, p3, p4):
    """returns True if the segments p1-p2 and p3-p4 intersect including endpoints"""
    # checks if p1-p3-p2-p4 is a convex quadrilateral
    e1 = p3 - p1
    e2 = p2 - p3
    e3 = p4 - p2
    e4 = p1 - p4
    c1, c2, c3, c4 = np.cross(e1,e2), np.cross(e2,e3), np.cross(e3,e4), np.cross(e4,e1)

    # sign of cross product determines cw/ccw orientation so all must have same sign to be convex
    if c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0:
        return True
    elif c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0:
        return True
    elif c1 != 0 and c2 != 0 and c3 != 0 and c4 != 0:
        return False

    # deal with pesky edge case of collinearity, won't happen 99% of the time but we want to be robust
    def on_segment(p1, p2, p3):
        """given three collinear points, returns True if p3 is on the line segment p1-p2 including endpoints"""
        return min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1])
    
    if c1 == 0 and on_segment(p1,p2,p3): return True
    if c2 == 0 and on_segment(p3,p4,p2): return True
    if c3 == 0 and on_segment(p1,p2,p4): return True
    if c4 == 0 and on_segment(p3,p4,p1): return True

    return False

def is_collision_polygon(p1, p2, polygon):
    """returns True if segment p1-p2 collides with polygon, treating endpoints as not part of the segment"""
    for i in range(len(polygon)):
        p3 = polygon[i]
        p4 = polygon[(i+1) % len(polygon)]
        if is_collision_segment(p1, p2, p3, p4):
            return True
    return False

def is_collision(t1, t2, obstacles):
    # checks both links against all obstacles
    # obstacle is a tuple (obstacle type, parameters)
    origin = np.array((0,0))
    elbow, tip = forward_kinematics(t1,t2)
    for obstacle in obstacles :
        if obstacle.getType() == ObstacleType.POLYGON:
            vertices = obstacle.getParams()
            if is_collision_polygon(origin, elbow, vertices) or is_collision_polygon(elbow, tip, vertices):
                return True
        elif obstacle.getType() == ObstacleType.CIRCLE:
            center, radius = obstacle.getParams()
            if is_collision_circle(origin, elbow, center, radius) or is_collision_circle(elbow, tip, center, radius):
                return True
    return False
