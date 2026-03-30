import math
import numpy as np

L1 = 80
L2 = 60

def forward_kinematics(t1, t2):
    # returns (elbow, tip) as (x,y) tuples
    elbow = np.array([L1*math.cos(t1), L1*math.sin(t1)])
    angle = t1+t2
    elbowToTip = np.array([L2*math.cos(angle), L2*math.sin(angle)])
    return elbow, elbow + elbowToTip

def segment_circle_distance(p1, p2, center, radius):
    # returns True if segment collides with circle
    d = p2 - p1
    t = np.dot(center-p1, d) / np.dot(d,d)
    if t < 0 :
        return np.linalg.norm(center-p1) < radius
    elif t > 1 :
        return np.linalg.norm(center-p2) < radius
    else:
        closest = p1 + t*d
        return np.linalg.norm(center-closest) < radius


def is_collision(t1, t2, obstacles):
    # checks both links against all obstacles
    origin = np.array((0,0))
    elbow, tip = forward_kinematics(t1,t2)
    for obstacle in obstacles :
        center,radius = obstacle
        if segment_circle_distance(origin, elbow, center, radius) or segment_circle_distance(elbow,tip,center,radius):
            return True
    return False

if __name__ == "__main__":
    obs = [(np.array([60.0, -40.0]), 35.0)]
    
    # arm pointing away from obstacle
    print(is_collision(math.pi * 0.75, 0, obs))
    
    # arm pointing directly at obstacle
    t1 = math.atan2(-40, 60)
    print(is_collision(t1, 0, obs))
    
    # sanity check — print where the tip ends up
    elbow, tip = forward_kinematics(t1, 0)
    print("tip:", tip)