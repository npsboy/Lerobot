import math

class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
        
    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def length(self):
        return math.hypot(self.x, self.y)

    def normalized(self):
        mag = self.length()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def get_angle(self):
        return math.atan2(self.y, self.x)

    def sin(self):
        mag = self.length()
        if mag == 0:
            return 0
        return self.y / mag

    def cos(self):
        mag = self.length()
        if mag == 0:
            return 1
        return self.x / mag

    def __lt__(self, other):
        return self.length() < other.length()

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"
