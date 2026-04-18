from enum import Enum

class ObstacleType(Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"

class Obstacle:
    def __init__(self, type, params):
        self.type = type
        self.params = params

    def getType(self):
        return self.type

    def getParams(self):
        return self.params
    
class CircleObstacle(Obstacle):
    def __init__(self, center, radius):
        super().__init__(ObstacleType.CIRCLE, (center, radius))

class PolygonObstacle(Obstacle):
    """vertices should be a list of numpy arrays (x,y) in cw or ccw order, at least 1 vertex"""
    def __init__(self, vertices):
        super().__init__(ObstacleType.POLYGON, vertices)