# Configuration Space Motion Planner

This project creates and graphs a 2d configuration space for a double-jointed arm with circular obstacles. 
The configuration space is a 2d coordinate plane with x and y axis representing the angles of the two segments of the arm. 
Each point in the space represents a unique position of the arm.
This project can generate paths between two points in the configuration space that avoids circular obstacles. 

## arm.py
This file contains calculates elbow and tip coordinates of the arm using basic trigonometry. 
It also has methods to check whether the arm intersects circular obstacles using vector projections.


## rrt.py
This file uses the rapidly exploring random tree algorithm to generate a path between a start position and end position on the configurations space. 
Essentially, this algorithm creates a tree of nodes and every step, 
it chooses a random direction and moves some small distance in that direction from the nearest node in the tree. 
If the step does not pass through any obstacles, then it adds it to the tree. 
It keeps doing this until getting within a step distance to the ending point.
It contains a method to smooth the path using the rrt-star algorithm which checks if three consecutive nodes can be smoothed be removing the middle node.

## cspace.py
This file graphs the configuration space using matplotlib by checking if each pixel is inside an obstacle. 
Users can select two starting points in the configuration space and generate a rrt path between them.
The project will animate the arm as the path moves forward in time.
