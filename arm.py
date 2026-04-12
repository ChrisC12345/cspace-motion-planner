# simple 2-link arm kinematics and collision checking
import math
import numpy as np
from enum import Enum
from obstacle import ObstacleType

L1 = 80
L2 = 60

def set_arm_lengths(l1, l2):
    global L1, L2
    L1 = l1
    L2 = l2


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

def _seg_circle(p1x, p1y, p2x, p2y, cx, cy, r2):
    """Vectorized segment-circle collision; p1/p2 are arrays, cx/cy/r2 are scalars."""
    dx, dy = p2x - p1x, p2y - p1y
    ls = np.where(dx*dx + dy*dy == 0, 1e-30, dx*dx + dy*dy)
    t = np.clip(((cx - p1x)*dx + (cy - p1y)*dy) / ls, 0.0, 1.0)
    qx, qy = p1x + t*dx - cx, p1y + t*dy - cy
    return qx*qx + qy*qy < r2


def _seg_seg(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Vectorized segment-segment intersection; p1/p2 are arrays, p3/p4 are scalars."""
    e1x, e1y = p3x-p1x, p3y-p1y
    e2x, e2y = p2x-p3x, p2y-p3y
    e3x, e3y = p4x-p2x, p4y-p2y
    e4x, e4y = p1x-p4x, p1y-p4y
    c1 = e1x*e2y - e1y*e2x
    c2 = e2x*e3y - e2y*e3x
    c3 = e3x*e4y - e3y*e4x
    c4 = e4x*e1y - e4y*e1x
    r = ((c1>0)&(c2>0)&(c3>0)&(c4>0)) | ((c1<0)&(c2<0)&(c3<0)&(c4<0))
    r |= (c1==0) & (np.minimum(p1x,p2x)<=p3x) & (p3x<=np.maximum(p1x,p2x)) & \
                   (np.minimum(p1y,p2y)<=p3y) & (p3y<=np.maximum(p1y,p2y))
    r |= (c2==0) & (min(p3x,p4x)<=p2x) & (p2x<=max(p3x,p4x)) & \
                   (min(p3y,p4y)<=p2y) & (p2y<=max(p3y,p4y))
    r |= (c3==0) & (np.minimum(p1x,p2x)<=p4x) & (p4x<=np.maximum(p1x,p2x)) & \
                   (np.minimum(p1y,p2y)<=p4y) & (p4y<=np.maximum(p1y,p2y))
    r |= (c4==0) & (min(p3x,p4x)<=p1x) & (p1x<=max(p3x,p4x)) & \
                   (min(p3y,p4y)<=p1y) & (p1y<=max(p3y,p4y))
    return r


def is_collision_batch(t1, t2, obstacles):
    """Vectorized is_collision for arrays of configs. Returns bool array of same shape as t1/t2."""
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    ex = L1 * np.cos(t1);  ey = L1 * np.sin(t1)
    angle = t1 + t2
    tx = ex + L2 * np.cos(angle);  ty = ey + L2 * np.sin(angle)
    ox = np.zeros_like(ex);  oy = np.zeros_like(ey)
    result = np.zeros_like(t1, dtype=bool)
    for obs in obstacles:
        if obs.getType() == ObstacleType.CIRCLE:
            center, radius = obs.getParams()
            cx, cy, r2 = float(center[0]), float(center[1]), float(radius)**2
            result |= _seg_circle(ox, oy, ex, ey, cx, cy, r2)
            result |= _seg_circle(ex, ey, tx, ty, cx, cy, r2)
        elif obs.getType() == ObstacleType.POLYGON:
            verts = obs.getParams()
            for k in range(len(verts)):
                p3x, p3y = float(verts[k][0]), float(verts[k][1])
                p4x, p4y = float(verts[(k+1)%len(verts)][0]), float(verts[(k+1)%len(verts)][1])
                result |= _seg_seg(ox, oy, ex, ey, p3x, p3y, p4x, p4y)
                result |= _seg_seg(ex, ey, tx, ty, p3x, p3y, p4x, p4y)
    return result


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
