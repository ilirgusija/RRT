import math
import random
from RRT import RRT, Vertex
from itertools import count

import matplotlib.pyplot as plt
import numpy as np

class RRT_Connect(RRT):
    def __init__(self, start, goal, obstacle, workspace, eta=2.5,
                 goal_sample_rate=0.01):
        super().__init__(start,goal,obstacle,workspace,eta,goal_sample_rate)
        self.vertices_b = []

    def planning(self):
        self.vertices = [self.start]
        self.vertices_b = [self.goal]
        for counter in count():  
            print(f"Counter: {counter}")
            # if not len(set(self.vertices).intersection(set(self.vertices_b))) == 0:
            intersection = list(set(self.vertices).intersection(set(self.vertices_b)))
            if not len(intersection) == 0:
                print(f"Found intersection on iteration ${counter}")
                print(f"Intersection length: {len(intersection)}")
                print(f"Intersection elements:")
                for i in intersection:
                    print(f"x: {i.x}, y: {i.y}")
                break
            x_rand = self.sample_random_vertex()
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)
            x_new = self.steer(v_nearest, x_rand)

            if self.is_edge_valid(v_nearest, x_new):
                self.vertices.append(x_new)
                self.connect(x_new)

            if counter % 3 == 0:
                self.update_graph(x_rand)

            self.vertices, self.vertices_b = self.vertices_b, self.vertices

        return self.final_paths(len(self.vertices) - 1, len(self.vertices_b) - 1)

    def connect(self, x_connect):
        v_nearest = self.get_nearest_vertex(x_connect, self.vertices_b)
        while True:
            x_step = self.steer(v_nearest, x_connect)
            if self.is_edge_valid(v_nearest, x_step):
                self.vertices_b.append(x_step)
                v_nearest = x_step

            if not (x_step != x_connect and self.is_edge_valid(v_nearest, x_step)):
                break
        return

    def final_paths(self, intersection_1, intersection_2):
        intersection = list(set(self.vertices).intersection(set(self.vertices_b)))
        if len(intersection) == 0:
            print("Error in breaking out of planning, the two sets do not share a vertex")

        path_a = [[intersection[0].x, intersection[0].y]]
        path_b = [[intersection[0].x, intersection[0].y]]

        vertex = self.vertices[intersection_1]
        while vertex.parent is not None:
            path_a.append([vertex.x, vertex.y])
            vertex = vertex.parent

        vertex_b = self.vertices_b[intersection_2]
        while vertex_b.parent is not None:
            path_b.append([vertex_b.x, vertex_b.y])
            vertex_b = vertex_b.parent

        path_a.append([vertex.x, vertex.y])
        path_b.append([vertex_b.x, vertex_b.y])
        path = path_a[1:][::-1] + path_b

        return path

    def sample_random_vertex(self):
        while True:
            sampled_vec = Vertex(random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
            if self.is_vertex_valid(sampled_vec):
                break
        return sampled_vec

def main():
    obstacles = [(25,10), (4,10)]

    # Set Initial parameters
    rrt_connect = RRT_Connect(
        start=[25.0, 50.0],
        goal=[75.0, 50.0],
        workspace=[0, 100],
        obstacle=obstacles[0],
        )
    path = rrt_connect.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        rrt_connect.update_graph()
        plt.plot([x for (x, _) in path], [y for (_, y) in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()

if __name__ == '__main__':
    main()
