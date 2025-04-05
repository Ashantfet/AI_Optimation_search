import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

class HillClimbingTSP:
    def __init__(self, distance_matrix, num_restarts=20 ,max_iterations=1000):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_restarts = num_restarts
        self.max_iterations = max_iterations
        self.city_coords = np.random.rand(self.num_cities, 2)  # Fixed positions

    def total_distance(self, tour):
        return sum(self.distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + \
               self.distance_matrix[tour[-1], tour[0]]

    def swap_two_cities(self, tour):
        new_tour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def hill_climbing(self):
        best_tour, best_distance, best_progress = None, float('inf'), []
        all_frames = []
        start_time = time.time()
        MAX_RUNTIME = 600 # 10 minutes timeout   


        print("🚀 Starting Hill Climbing...")

        for restart in range(self.num_restarts):
            current_tour = np.random.permutation(self.num_cities)
            current_distance = self.total_distance(current_tour)
            progress = [current_distance]

            print(f"🔄 Restart {restart+1} | Initial Distance: {current_distance:.2f}")

            for iteration in range(self.max_iterations):
                if time.time() - start_time > MAX_RUNTIME:
                    print("⏳ Timeout reached. Ending early.")
                    break

                new_tour = self.swap_two_cities(current_tour)
                new_distance = self.total_distance(new_tour)

                if new_distance < current_distance:
                    current_tour, current_distance = new_tour, new_distance
                    progress.append(new_distance)

                if iteration % 10 == 0:
                    frame = self.plot_tour(current_tour, iteration, current_distance)
                    all_frames.append(frame)

            if current_distance < best_distance:
                best_tour, best_distance, best_progress = current_tour, current_distance, progress

        total_time = time.time() - start_time
        self.save_gif(all_frames, "hill_climbing_tsp.gif")
        convergence_point = len(best_progress)
        reward = -best_distance

        return best_tour, best_distance, total_time, convergence_point, reward, best_progress

    def plot_tour(self, tour, iteration, dist):
        fig, ax = plt.subplots(figsize=(5, 5))
        ordered = self.city_coords[tour]

        ax.plot(ordered[:, 0], ordered[:, 1], 'o-', color='steelblue', markersize=6)
        ax.plot([ordered[-1, 0], ordered[0, 0]], [ordered[-1, 1], ordered[0, 1]], 'r--', lw=1.5)
        ax.set_title(f"Iteration {iteration}\nDistance: {dist:.2f}", fontsize=10)
        ax.axis('off')
        fig.tight_layout()

        fname = "temp_hc_frame.png"
        plt.savefig(fname)
        plt.close(fig)
        return imageio.imread(fname)

    def save_gif(self, frames, filename):
        if not frames:
            print("⚠️ No frames to save.")
            return
        try:
            imageio.mimsave(filename, frames, duration=0.2)
            print(f"✅ GIF saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")

# --- Usage ---
if __name__ == "__main__":
    num_cities = 126
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)

    hc = HillClimbingTSP(distance_matrix)
    tour, dist, exec_time, convergence, reward, progress = hc.hill_climbing()

    print(f"🔹 Best Distance: {dist:.2f}")
    print(f"⏳ Time: {exec_time:.2f} sec")
    print(f"📈 Convergence Point: {convergence}")
    print(f"🏆 Reward: {reward}")
    print(f"💰 Progress: {progress}")