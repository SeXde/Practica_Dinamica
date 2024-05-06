class Particle:
    def __init__(self, x: int, y: int, width: float, height: float, weight: float, vx=0, vy=0):
        # Bounding box parameters
        self.x = x
        self.y = y
        self.w = width
        self.h = height

        # Velocity
        self.vx = 0
        self.vy = 0

        # Weight
        self.weight = weight

    def get_bounding_box(self) -> (int, int, int, int):
        return self.x, self.y, self.w, self.h
