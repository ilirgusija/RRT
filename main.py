import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RRT import RRT
from RRT_Connect import RRT_Connect

import math

def compute_path_length(path):
    if len(path) < 2:
        return 0  

    total_length = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += distance

    return total_length

def select_planner(alg, start, goal, obstacle, animation):
    rrt=None
    if alg == "RRT":
        rrt = RRT(
            start=start,
            goal=goal,
            workspace=[0, 100],
            obstacle=obstacle,
            animation=animation
        )
    elif alg == "RRT_Connect":
        rrt = RRT_Connect(
            start=start,
            goal=goal,
            workspace=[0, 100],
            obstacle=obstacle,
            animation=animation
        )
    return rrt

def histograms(alg, obs, iterations, num_verts):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(iterations, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Iterations Histogram")
    plt.xlabel("Iterations")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(num_verts, bins=10, color='green', alpha=0.7, edgecolor='black')
    plt.title("Number of Vertices Histogram")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"figures/hist_{alg}_{obs}.png", format="png", dpi=300)

def save_medians_to_csv(iterations, num_verts, paths, alg, obs, csv_file="results.csv"):
    median_iterations = np.median(iterations)
    median_num_verts = np.median(num_verts)
    median_paths = np.median(paths)

    column_name = f"{alg}_{obs}"

    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(index=["iterations", "num_verts", "paths"])

    df[column_name] = [median_iterations, median_num_verts, median_paths]

    df.to_csv(csv_file)

def main(obs, alg, animation):
    start=[25.0, 50.0]
    goal=[75.0, 50.0]
    # Define obstacles
    bug_trap_start = [[start, 30, "vertical", (12.5,0)], 
                 [start, 20, "horizontal", (0, -12.5)],
                 [start, 20, "horizontal", (0,12.5)],
                 [start, 14, "vertical", (-12.5,8)],
                 [start, 14, "vertical", (-12.5,-8)] ]
    bug_trap_goal = [[goal, 30, "vertical", (-12.5,0)], 
                 [goal, 20, "horizontal", (0, -12.5)],
                 [goal, 20, "horizontal", (0,12.5)],
                 [goal, 14, "vertical", (12.5,8)],
                 [goal, 14, "vertical", (12.5,-8)] ]
    obstacles = [(25, 10), (4, 10), bug_trap_start, bug_trap_goal,
                 bug_trap_start + bug_trap_goal]
    obstacle = obstacles[obs-1]

    # Run the selected algorithm
    path = []
    planners=[]
    num_iters=1
    for i in range(0,num_iters):
        print(f"Iteration: {i}")
        if i==num_iters-1:
            planners.append(select_planner(alg, start, goal, obstacle, True)) 
        else:
            planners.append(select_planner(alg, start, goal, obstacle, animation)) 

        path.append(planners[i].planning())

        if planners[i].animation is True:
            planners[i].update_graph()
            plt.plot([x for (x, _) in path[i]], [y for (_, y) in path[i]], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.savefig(f"figures/final_path_{alg}_{obs}.png", format="png", dpi=300)

    if num_iters >= 100:
        iterations = [planner.iterations for planner in planners]
        num_verts = [planner.num_vertices for planner in planners]
        histograms(alg, obs, iterations, num_verts)
        save_medians_to_csv(iterations, num_verts, [compute_path_length(p) for p in path], alg, obs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run specific RRT algorithms with selected obstacles.")
    parser.add_argument(
        "--animation",
        type=str,
        choices=["t","f"],
        default="t",
        help="Select True or False for animation"
    )
    parser.add_argument(
        "--obstacle", 
        type=int, 
        choices=[1, 2, 3, 4, 5], 
        default=0, 
        help="Select the obstacle index (1-5)."
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        choices=["RRT", "RRT_Connect"], 
        default="RRT", 
        help="Select the RRT algorithm to run."
    )
    args = parser.parse_args()
    animation = True if args.animation=="t" else False
    main(args.obstacle, args.algorithm, animation)
