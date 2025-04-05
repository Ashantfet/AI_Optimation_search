import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

class SimulatedAnnealingTSP:
    def __init__(self, distance_matrix, initial_temp=1000, cooling_rate=0.995, min_temp=1):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.city_coords = np.random.rand(self.num_cities, 2)  # Fixed city positions

    def total_distance(self, tour):
        return sum(self.distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + \
               self.distance_matrix[tour[-1], tour[0]]

    def swap_two_cities(self, tour):
        new_tour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def simulated_annealing(self):
        current_tour = np.random.permutation(self.num_cities)
        current_distance = self.total_distance(current_tour)

        best_tour, best_distance = current_tour.copy(), current_distance
        temperature = self.initial_temp
        all_frames = []
        progress = [current_distance]

        start_time = time.time()
        MAX_RUNTIME = 600  # 10 minutes timeout
        print(f"🔄 Initial Distance: {current_distance:.2f}")

        print("🚀 Starting Simulated Annealing...")

        iterations = 0
        best_iteration = 0

        while temperature > self.min_temp:
            if time.time() - start_time > MAX_RUNTIME:
                print("⏳ Timeout reached.")
                break

            new_tour = self.swap_two_cities(current_tour)
            new_distance = self.total_distance(new_tour)
            delta = new_distance - current_distance

            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current_tour, current_distance = new_tour, new_distance
                progress.append(current_distance)

            if current_distance < best_distance:
                best_tour, best_distance = current_tour.copy(), current_distance
                best_iteration = iterations

            if iterations % 10 == 0:
                frame = self.plot_tour(current_tour, current_distance, temperature, iterations)
                all_frames.append(frame)

            temperature *= self.cooling_rate
            iterations += 1

        total_time = time.time() - start_time
        self.save_gif(all_frames, "simulated_annealing_tsp.gif")
        reward = -best_distance

        print(f"🏁 Done | Best Distance: {best_distance:.2f}, Time: {total_time:.2f}s, Iterations: {iterations}")

        return best_tour, best_distance, total_time, best_iteration, reward, progress

    def plot_tour(self, tour, energy, temperature, iteration):
        fig, ax = plt.subplots(figsize=(5, 5))
        ordered = self.city_coords[tour]

        ax.plot(ordered[:, 0], ordered[:, 1], 'o-', color='darkorange', markersize=6)
        ax.plot([ordered[-1, 0], ordered[0, 0]], [ordered[-1, 1], ordered[0, 1]], 'r--', lw=1.5)
        ax.set_title(f"SA Iter {iteration}\nDistance: {energy:.2f}, T: {temperature:.2f}", fontsize=10)
        ax.axis('off')
        fig.tight_layout()

        fname = "temp_sa_frame.png"
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

# --- Usage Example ---
if __name__ == "__main__":
    num_cities = 126
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)

    sa_solver = SimulatedAnnealingTSP(distance_matrix)
    tour, dist, exec_time, convergence, reward, progress = sa_solver.simulated_annealing()

    print(f"🔹 Best Distance: {dist:.2f}")
    print(f"⏳ Time Taken: {exec_time:.2f} seconds")
    print(f"📉 Convergence Point (iteration): {convergence}")
    print(f"🏆 Reward (Distance): {reward}")
    print(f"📈 Progress: {progress}")