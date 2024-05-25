import cv2
import random
import numpy as np

from .particle import Particle


class ParticleFilter:
    """
    A class to represent a Particle Filter for tracking objects in an image sequence.

    Attributes:
    -----------
    num_particles : int
        Number of particles in the filter.
    particle_shape : (int, int)
        Shape (height, width) of each particle.
    image_shape : (int, int)
        Shape (height, width) of the image.
    particles : list of Particle
        List of particles in the filter.
    """

    def __init__(self, num_particles: int, particle_shape: (int, int), image_shape: (int, int)):
        """
        Initializes the ParticleFilter with the given parameters.

        Parameters:
        -----------
        num_particles : int
            Number of particles in the filter.
        particle_shape : (int, int)
            Shape (height, width) of each particle.
        image_shape : (int, int)
            Shape (height, width) of the image.
        """
        self.image_shape = image_shape
        self.num_particles = num_particles
        self.particle_shape = particle_shape
        self.particles = self.__generate_particles(num_particles, particle_shape, image_shape)

    def draw_particles(self, frame: np.array):
        """
        Draws the particles on the given frame.

        Parameters:
        -----------
        frame : np.array
            The frame on which to draw the particles.
        """
        for p in self.particles:
            cv2.rectangle(frame, (p.x, p.y), (p.x + p.w, p.y + p.h), (255, 0, 0), 1)

    def track(self, bs_foreground: np.array) -> (int, int, int, int):
        """
        Tracks the object using the background-subtracted foreground image.

        Parameters:
        -----------
        bs_foreground : np.array
            Background-subtracted foreground image.

        Returns:
        --------
        tuple or None
            Bounding box of the best particle or None if no particles are found.
        """
        accumulated_weights = self.__update_weights(bs_foreground)
        if accumulated_weights[-1] == 0:
            self.particles = self.__generate_particles(self.num_particles, self.particle_shape, self.image_shape)
            return None
        best_particle = self.__best_particle()
        self.__roulette_algorithm(accumulated_weights)
        return best_particle.get_bounding_box()

    def __update_weights(self, bs_foreground: np.array):
        """
        Updates the weights of the particles based on the foreground image.

        Parameters:
        -----------
        bs_foreground : np.array
            Background-subtracted foreground image.

        Returns:
        --------
        list of float
            Accumulated weights of the particles.
        """
        accumulated_weights = []
        accumulator = 0
        num_white_pixels = self.__num_white_pixels(bs_foreground)
        for p in self.particles:
            if num_white_pixels > 0:
                particle_crop = bs_foreground[p.y:p.y + p.h, p.x:p.x + p.w]
                p.weight = cv2.countNonZero(particle_crop) / num_white_pixels
            else:
                p.weight = 0
            accumulator += p.weight
            accumulated_weights.append(accumulator)
        return accumulated_weights

    def __num_white_pixels(self, bs_foreground: np.array) -> int:
        """
        Counts the number of white pixels in the foreground image.

        Parameters:
        -----------
        bs_foreground : np.array
            Background-subtracted foreground image.

        Returns:
        --------
        int
            Number of white pixels in the foreground image.
        """
        num = 0
        for p in self.particles:
            particle_crop = bs_foreground[p.y:p.y + p.h, p.x:p.x + p.w]
            num += cv2.countNonZero(particle_crop)
        return num

    def __best_particle(self) -> Particle:
        """
        Finds the particle with the highest weight.

        Returns:
        --------
        Particle
            The particle with the highest weight.
        """
        max_weight = 0
        best_particle = self.particles[0]
        for p in self.particles:
            if p.weight > max_weight:
                max_weight = p.weight
                best_particle = p
        return best_particle

    def __diffusion(self, particle: Particle) -> Particle:
        """
        Applies diffusion to the given particle to create a new particle.

        Parameters:
        -----------
        particle : Particle
            The particle to diffuse.

        Returns:
        --------
        Particle
            The new diffused particle.
        """
        dif_x = int(particle.x + 10 * np.random.normal(0, 1) + particle.vx)
        dif_y = int(particle.y + 10 * np.random.normal(0, 1) + particle.vy)
        if dif_x < 0:
            dif_x = 0
        elif dif_x + particle.w > self.image_shape[1]:
            dif_x = self.image_shape[1] - particle.w
        if dif_y < 0:
            dif_y = 0
        elif dif_y + particle.h > self.image_shape[0]:
            dif_y = self.image_shape[0] - particle.h

        dif_vx = particle.vx + np.random.normal(0, 1)
        dif_vy = particle.vy + np.random.normal(0, 1)

        return Particle(dif_x, dif_y, particle.w, particle.h, particle.weight, dif_vx, dif_vy)

    def __roulette_algorithm(self, accumulated_weights: [float]):
        """
        Applies the roulette wheel selection algorithm to select new particles.

        Parameters:
        -----------
        accumulated_weights : list of float
            Accumulated weights of the particles.
        """
        new_particles = []
        for i in range(len(self.particles)):
            rand_num = np.random.uniform(0, 1)
            index = np.argmax(np.array(accumulated_weights) > rand_num)
            particle = self.particles[index]
            new_particles.append(self.__diffusion(particle))
        self.particles = new_particles

    @staticmethod
    def __generate_particles(num_particles: int, particle_shape: (int, int), image_shape: (int, int)):
        """
        Generates initial particles randomly distributed across the image.

        Parameters:
        -----------
        num_particles : int
            Number of particles to generate.
        particle_shape : (int, int)
            Shape (height, width) of each particle.
        image_shape : (int, int)
            Shape (height, width) of the image.

        Returns:
        --------
        list of Particle
            List of generated particles.
        """
        particles = []
        for i in range(num_particles):
            x = random.randint(0, image_shape[1] - particle_shape[1] - 1)
            y = random.randint(0, image_shape[0] - particle_shape[0] - 1)
            weight = 1 / num_particles
            particles.append(Particle(x, y, particle_shape[1], particle_shape[0], weight))
        return particles
