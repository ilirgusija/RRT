import math
import random
from itertools import count

import matplotlib.pyplot as plt

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.x == other.x and self.y == other.y
        return False
    def __hash__(self):
        return hash((self.x, self.y))

class RRT:
    def __init__(self, start, goal, obstacle, workspace, animation=True, eta=2.5,
                 goal_sample_rate=0.01):
        """
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle:Coordinates of rectangle obstacle [left,right,bottom,top]
        workspace:Min/max coordinates of our square arena [min,max]
        """
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.min_rand = workspace[0]
        self.max_rand = workspace[1]
        self.eta = eta
        self.goal_sample_rate = goal_sample_rate
        self.animation=animation
        self.obstacle = []
        if isinstance(obstacle, tuple):
            self.obstacle.append(self.convert_to_rectangle(obstacle[0],obstacle[1],self.min_rand,self.max_rand)) 
        elif isinstance(obstacle, list):
            for o in obstacle:
                ref, length, orientation, translation = o
                self.obstacle.append(self.generate_rectangle_from_reference(ref, length, orientation, translation))
        self.vertices = []
        self.iterations = 0
        self.num_vertices = 0

    def planning(self):
        self.vertices = [self.start]
        for self.iterations in count(): 
            if self.goal in self.vertices:
                break
            x_rand = self.sample_random_vertex()
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)
            x_new = self.steer(v_nearest, x_rand)

            if self.is_edge_valid(v_nearest, x_new):
                self.vertices.append(x_new)

            if self.iterations % 3 and self.animation is True == 0:
                self.update_graph(x_rand)

        return self.final_path(len(self.vertices) - 1)

    def steer(self, v_nearest, x_rand):
        x_new = Vertex(v_nearest.x, v_nearest.y)
        d, angle = self.calc_distance_and_angle(x_new, x_rand)

        x_new.path_x = [x_new.x]
        x_new.path_y = [x_new.y]

        if self.eta < d:
            x_new.x += self.eta * math.cos(angle)
            x_new.y += self.eta * math.sin(angle)
        else:
            x_new.x += d * math.cos(angle)
            x_new.y += d * math.sin(angle)

        x_new.path_x.append(x_new.x)
        x_new.path_y.append(x_new.y)

        x_new.parent = v_nearest 

        return x_new

    def is_vertex_valid(self, vertex):
        if vertex is None:
            return False
        for o in self.obstacle:
            left, right, bottom, top = o
            for x, y in zip(vertex.path_x, vertex.path_y):
                if (left  <= x <= right and bottom <= y <= top):
                    return False  
        return True  

    def is_edge_valid(self, v_nearest, x_rand):
        path_resolution=0.1
        x_new = Vertex(v_nearest.x, v_nearest.y)
        d, angle = self.calc_distance_and_angle(x_new, x_rand)
        if not self.is_vertex_valid(x_rand):
            return False
        x_new.path_x = [x_new.x]
        x_new.path_y = [x_new.y]

        if self.eta > d:
            n_steps = math.floor(d / path_resolution)
        else:
            n_steps = math.floor(self.eta / path_resolution)

        for _ in range(n_steps):
            x_new.x += path_resolution * math.cos(angle)
            x_new.y += path_resolution * math.sin(angle)
            if not self.is_vertex_valid(x_new):
                return False
            x_new.path_x.append(x_new.x)
            x_new.path_y.append(x_new.y)

        d, _ = self.calc_distance_and_angle(x_new, x_rand)
        if d <= path_resolution:
            x_new.path_x.append(x_rand.x)
            x_new.path_y.append(x_rand.y)
            x_new.x = x_rand.x
            x_new.y = x_rand.y

        return True

    def update_graph(self, sampled_vec=None):
        plt.clf()
        # Plot the sampled vector as a black plus sign
        if sampled_vec is not None:
            plt.plot(sampled_vec.x, sampled_vec.y, "Pk")

        # Plot edges as yellow lines
        for vertex in self.vertices:
            if vertex.parent:
                plt.plot(vertex.path_x, vertex.path_y, "-y")

        for o in self.obstacle:
            # Plot the blue rectangle obstacle
            self.plot_rectangle(o)

        # Plot the green start "S" and red goal "G"
        plt.plot(self.start.x, self.start.y, c="g", marker=r"$\mathbb{S}$")
        plt.plot(self.goal.x, self.goal.y, c="r", marker=r"$\mathbb{G}$")

        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def sample_random_vertex(self):
        if random.random() <= self.goal_sample_rate:
            sampled_vec = Vertex(self.goal.x, self.goal.y)
        else: 
            while True:
                sampled_vec = Vertex(random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand))
                if self.is_vertex_valid(sampled_vec) is True:
                    break
        return sampled_vec

    def get_nearest_vertex(self, x_rand, vertices):
        dlist = [self.L2_norm(vertex, x_rand) for vertex in vertices]
        minind = dlist.index(min(dlist))
        return vertices[minind]

    def final_path(self, g_idx):
        path = [[self.goal.x, self.goal.y]]
        vertex = self.vertices[g_idx]
        while vertex.parent is not None:
            path.append([vertex.x, vertex.y])
            vertex = vertex.parent
        path.append([vertex.x, vertex.y])
        self.num_vertices = len(self.vertices)
        return path

    @staticmethod
    def L2_norm(left, right):
        return (left.x - right.x)**2 + (left.y - right.y)**2

    @staticmethod
    def convert_to_rectangle(l, width, min_rand, max_rand):
        map_width = max_rand - min_rand  # Assuming square map

        # Calculate the top, bottom, left, and right boundaries of the rectangle
        top = max_rand - l
        bottom = l
        left = (map_width - width) / 2
        right = left + width

        # Check for valid rectangle within map bounds
        if bottom < min_rand or top > max_rand or width > map_width:
            raise ValueError("Invalid rectangle dimensions: exceeds map bounds.")

        return left, right, bottom, top

    @staticmethod
    def plot_rectangle(rectangle, color="-b"):
        left, right, bottom, top = rectangle

        # Rectangle corners
        x_coords = [left, right, right, left, left]
        y_coords = [bottom, bottom, top, top, bottom]

        # Plot the rectangle
        plt.plot(x_coords, y_coords, color)

    @staticmethod
    def calc_distance_and_angle(parent, child):
        dx = child.x - parent.x
        dy = child.y - parent.y
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        return length, angle

    @staticmethod
    def generate_rectangle_from_reference(reference, length, orientation="horizontal", translation=(10, 0)):
        x_ref, y_ref = reference
        dx, dy = translation
        x_translated = x_ref + dx
        y_translated = y_ref + dy

        if orientation == "horizontal":
            # Fixed height of 10, horizontal length is variable
            left = x_translated-length/2
            right = x_translated + length/2
            bottom = y_translated - 2.5
            top = y_translated + 2.5
        elif orientation == "vertical":
            # Fixed width of 10, vertical length is variable
            left = x_translated - 2.5
            right = x_translated + 2.5
            bottom = y_translated-length/2
            top = y_translated + length/2
        else:
            raise ValueError("Invalid orientation. Choose 'horizontal' or 'vertical'.")

        return left, right, bottom, top
