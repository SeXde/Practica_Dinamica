class Particle:
    """
    A class to represent a particle with bounding box, velocity, and weight.

    Attributes:
    -----------
    x : int
        The x-coordinate of the particle.
    y : int
        The y-coordinate of the particle.
    width : float
        The width of the particle's bounding box.
    height : float
        The height of the particle's bounding box.
    weight : float
        The weight of the particle.
    vx : float, optional
        The velocity of the particle in the x-direction (default is 0).
    vy : float, optional
        The velocity of the particle in the y-direction (default is 0).
    """

    def __init__(self, x: int, y: int, width: float, height: float, weight: float, vx=0, vy=0):
        """
        Initializes the Particle with the given parameters.

        Parameters:
        -----------
        x : int
            The x-coordinate of the particle.
        y : int
            The y-coordinate of the particle.
        width : float
            The width of the particle's bounding box.
        height : float
            The height of the particle's bounding box.
        weight : float
            The weight of the particle.
        vx : float, optional
            The velocity of the particle in the x-direction (default is 0).
        vy : float, optional
            The velocity of the particle in the y-direction (default is 0).
        """

        # Bounding box parameters
        self.x = x
        self.y = y
        self.w = width
        self.h = height

        # Velocity
        self.vx = vx
        self.vy = vy

        # Weight
        self.weight = weight

    def get_bounding_box(self) -> (int, int, int, int):
        """
        Returns the bounding box of the particle.

        Returns:
        --------
        tuple
            A tuple containing the x-coordinate, y-coordinate, width, and height of the bounding box.
        """
        return self.x, self.y, self.w, self.h
