import gymnasium as gym
import numpy as np
import heapq
import imageio
import cv2  # OpenCV for drawing arrows
import matplotlib.pyplot as plt
import time  # Import time module for execution time measurement

class BranchAndBoundSolver:
    def __init__(self, env):
        self.env = env
        self.start_state = 0
        self.goal_state = env.observation_space.n - 1
        self.best_cost = float("inf")
        self.best_path = None
        self.execution_time = None  # Store execution time

    def solve(self):
        """Solves Frozen Lake using Branch and Bound."""
        start_time = time.perf_counter()
  # Start timing
        queue = [(0, [self.start_state])]  # Min-heap: (cost, path)
        heapq.heapify(queue)

        while queue:
            cost, path = heapq.heappop(queue)
            current_state = path[-1]

            if current_state == self.goal_state:
                self.best_cost = cost
                self.best_path = path
                self.execution_time = time.perf_counter() - start_time
  # Record execution time
                return self.best_path, self.best_cost, self.execution_time  # Stop immediately

            for action in range(self.env.action_space.n):  # Actions: Left, Down, Right, Up
                for prob, next_state, reward, done in self.env.unwrapped.P[current_state][action]:
                    if prob > 0 and next_state not in path:  # Avoid cycles
                        new_cost = cost + 1
                        if new_cost < self.best_cost:  # Bounding condition
                            heapq.heappush(queue, (new_cost, path + [next_state]))

        self.execution_time = time.perf_counter() - start_time
  # Record execution time if no solution found
        return None, float("inf"), self.execution_time  # Return no solution

    def generate_gif(self, path, filename="bnb_frozen_lake.gif"):
        """Generates a GIF with the original Frozen Lake animations."""
        frames = []
        obs, _ = self.env.reset()  # Reset environment

        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]

            # Find the action that moves from current_state to next_state
            for action in range(self.env.action_space.n):  # 4 possible actions
                for prob, state, reward, done in self.env.unwrapped.P[current_state][action]:
                    if state == next_state and prob > 0:
                        obs, _, _, _, _ = self.env.step(action)  # Take the step properly
                        break

            frame = self.env.render()  # Render the current frame
            frames.append(frame)

        imageio.mimsave(filename, frames, duration=0.5)
        print(f"✅ GIF saved as {filename}")

        # Show the final frame
        plt.imshow(frames[-1])
        plt.axis("off")
        # plt.show()
    def render_frozen_lake(self, agent_pos, prev_pos=None):
        """Generates an image of Frozen Lake with movement arrows."""
        size = int(np.sqrt(self.env.observation_space.n))  # Grid size (e.g., 4x4)
        lake_map = self.env.unwrapped.desc  # Get map layout
        img = np.ones((size * 100, size * 100, 3), dtype=np.uint8) * 255  # White background

        color_map = {
            b"S": (0, 255, 0),  # Green for Start
            b"F": (200, 200, 200),  # Light gray for Frozen tile
            b"H": (0, 0, 0),  # Black for Hole
            b"G": (255, 215, 0)  # Yellow for Goal
        }

        # Draw map
        for i in range(size):
            for j in range(size):
                tile = lake_map[i][j]
                img[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = color_map[tile]

        # Draw agent
        agent_x, agent_y = divmod(agent_pos, size)
        cv2.circle(img, (agent_y * 100 + 50, agent_x * 100 + 50), 30, (255, 0, 0), -1)  # Red circle for agent

        # Draw movement arrow if there was a previous position
        if prev_pos is not None:
            prev_x, prev_y = divmod(prev_pos, size)
            start = (prev_y * 100 + 50, prev_x * 100 + 50)  # Start of arrow
            end = (agent_y * 100 + 50, agent_x * 100 + 50)  # End of arrow
            img = cv2.arrowedLine(img, start, end, (0, 0, 255), 5)  # Red arrow

        return img

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Solve using BnB
bnb_solver = BranchAndBoundSolver(env)
solution_path, solution_cost, execution_time = bnb_solver.solve()

# Check if a solution was found before proceeding
if solution_path:
    print("Best Path:", solution_path)
    print("Best Cost:", solution_cost)
    print(f"Execution Time: {execution_time:.6f} seconds")

    # Generate GIF with arrows showing movement
    bnb_solver.generate_gif(solution_path)
else:
    print("No solution found.")
