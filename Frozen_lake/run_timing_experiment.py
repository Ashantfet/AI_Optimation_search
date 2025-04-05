import subprocess
import time
import csv

# File paths
bnb_script = "bnb_frozen_lake.py"
ida_script = "ida_frozen_lake.py"
csv_filename = "execution_times.csv"
log_filename = "execution_log.txt"

num_runs = 5
results = []

# Open log file
with open(log_filename, "w") as log_file:

    def run_script(script_name):
        """ Runs a script, extracts execution time, and logs output """
        start_time = time.time()
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        elapsed_time = time.time() - start_time

        output = result.stdout.strip()
        log_file.write(f"\n=== Output of {script_name} ===\n{output}\n=============================\n")

        try:
            time_taken = float(output.split("Execution Time: ")[1].split(" seconds")[0])
            return time_taken
        except (IndexError, ValueError) as e:
            log_file.write(f"⚠️ Parsing error in {script_name}: {e}\n")
            return float("inf")

    # Run experiments
    for i in range(num_runs):
        log_file.write(f"\nRunning test {i+1}/{num_runs}...\n")
        print(f"Running test {i+1}/{num_runs}...")

        bnb_time = run_script(bnb_script)
        ida_time = run_script(ida_script)

        results.append([i+1, bnb_time, ida_time])

# Save results to CSV
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "BnB Execution Time (s)", "IDA* Execution Time (s)"])
    writer.writerows(results)

print(f"✅ Execution times saved in {csv_filename}")
print(f"📜 Full log saved in {log_filename}")
