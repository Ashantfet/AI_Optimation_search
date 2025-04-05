import gymnasium as gym
import numpy as np
import imageio
import cv2  # OpenCV for arrows
import matplotlib.pyplot as plt
import time  # Import time module for execution time measurement

class IDAStarSolver:
    def __init__(self, env):
        self.env = env
        self.start_state = 0
        self.goal_state = env.observation_space.n - 1
        self.best_path = None
        self.best_cost = float("inf")
        self.execution_time = None  # Store execution time

    def heuristic(self, state):
        """ Manhattan distance heuristic """
        size = int(np.sqrt(self.env.observation_space.n))
        x1, y1 = divmod(state, size)
        x2, y2 = divmod(self.goal_state, size)
        return abs(x1 - x2) + abs(y1 - y2)

    def dfs(self, state, path, g, threshold):
        """ Depth-First Search (DFS) with threshold pruning """
        f = g + self.heuristic(state)  # f = g + h

        if f > threshold:
            return f  # Return new minimum threshold

        if state == self.goal_state:
            self.best_path = path
            self.best_cost = g
            return None  # Found solution

        min_threshold = float("inf")

        for action in range(self.env.action_space.n):  
            for prob, next_state, _, _ in self.env.unwrapped.P[state][action]:
                if prob > 0 and next_state not in path:
                    result = self.dfs(next_state, path + [next_state], g + 1, threshold)
                    if result is None:
                        return None  # Solution found
                    min_threshold = min(min_threshold, result)

        return min_threshold  

    def solve(self):
        """ Iterative Deepening A* """
        start_time = time.perf_counter()  # Start timing
        threshold = self.heuristic(self.start_state)

        while True:
            result = self.dfs(self.start_state, [self.start_state], 0, threshold)
            if result is None:
                self.execution_time = time.perf_counter() - start_time  # Record execution time
                return self.best_path, self.best_cost, self.execution_time  # Found solution
            if result == float("inf"):
                self.execution_time = time.perf_counter() - start_time  # Record execution time if no solution
                return None, float("inf"), self.execution_time  # No solution
            threshold = result  

    def generate_gif(self, path, filename="ida_frozen_lake.gif"):
        """Generates a GIF using Gym's actual step mechanics to show movement."""
        frames = []
        obs, _ = self.env.reset()  # Reset the environment

        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]

            # Find the correct action to move from current_state to next_state
            for action in range(self.env.action_space.n):  # 4 possible actions
                for prob, state, reward, done in self.env.unwrapped.P[current_state][action]:
                    if state == next_state and prob > 0:
                        obs, _, _, _, _ = self.env.step(action)  # Take the step properly
                        break

            frame = self.env.render()  # Capture the rendered frame
            frames.append(frame)

        imageio.mimsave(filename, frames, duration=0.5)
        print(f"✅ GIF saved as {filename}")

        # Show the final frame
        plt.imshow(frames[-1])
        plt.axis("off")
        #plt.show()


    def render_frozen_lake(self, agent_pos, prev_pos=None):
        """ Generates an image of Frozen Lake with movement arrows """
        size = int(np.sqrt(self.env.observation_space.n))  
        lake_map = self.env.unwrapped.desc  
        img = np.ones((size * 100, size * 100, 3), dtype=np.uint8) * 255  

        color_map = {
            b"S": (0, 255, 0),  
            b"F": (200, 200, 200),  
            b"H": (0, 0, 0),  
            b"G": (255, 215, 0)  
        }

        # Draw the map
        for i in range(size):
            for j in range(size):
                tile = lake_map[i][j]
                img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color_map[tile]

        # Draw agent
        agent_x, agent_y = divmod(agent_pos, size)
        cv2.circle(img, (agent_y * 100 + 50, agent_x * 100 + 50), 30, (255, 0, 0), -1)  

        # Draw movement arrow if there was a previous position
        if prev_pos is not None:
            prev_x, prev_y = divmod(prev_pos, size)
            start = (prev_y * 100 + 50, prev_x * 100 + 50)  
            end = (agent_y * 100 + 50, agent_x * 100 + 50)  
            img = cv2.arrowedLine(img, start, end, (0, 0, 255), 5)  

        return img

# Initialize Frozen Lake environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Solve using IDA*
ida_solver = IDAStarSolver(env)
solution_path, solution_cost, execution_time = ida_solver.solve()

# Check if a solution was found
if solution_path:
    print("Best Path:", solution_path)
    print("Best Cost:", solution_cost)
    print(f"Execution Time: {execution_time:.6f} seconds")

    # Generate GIF with arrows showing movement
    ida_solver.generate_gif(solution_path)
else:
    print("No solution found.")
