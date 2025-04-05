import os
import subprocess
import time
import csv
import re

scripts = ["hill_climbing.py", "simulated_annealing.py"]

csv_filename = "execution_results.csv"
log_filename = "execution_log.txt"
num_runs = 1

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

results = []

# Metrics to extract
metric_keys = ["Best Distance", "Time", "Convergence Point", "Reward"]

def extract_metrics(output):
    """Extracts relevant metrics from script output"""
    metrics = {key: "N/A" for key in metric_keys}
    try:
        for line in output.splitlines():
            for key in metric_keys:
                if key in line:
                    match = re.search(rf"{key}[:\s]+(-?\d+\.?\d*)", line)
                    if match:
                        metrics[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
    except Exception as e:
        print(f"⚠️ Failed to extract metrics: {e}")
    return metrics

with open(log_filename, "w", encoding="utf-8") as log_file:

    def run_script(script_name):
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", script_name],
                capture_output=True,
                text=True,
                timeout=1200,
                encoding="utf-8",
                errors="replace",
                env=env
            )
            elapsed_time = time.time() - start_time
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            log_file.write(f"\n=== Output of {script_name} ===\n{output}\n")
            if error_output:
                log_file.write(f"\n⚠️ Errors:\n{error_output}\n")
            log_file.write("=============================\n")

            metrics = extract_metrics(output)
            metrics["Execution Time"] = elapsed_time
            return metrics

        except subprocess.TimeoutExpired:
            log_file.write(f"⚠️ {script_name} timed out!\n")
            return {key: "Timeout" for key in metric_keys + ["Execution Time"]}
        except Exception as e:
            log_file.write(f"❌ Error running {script_name}: {e}\n")
            return {key: "Error" for key in metric_keys + ["Execution Time"]}

    for i in range(num_runs):
        print(f"Running test {i+1}/{num_runs}...")
        log_file.write(f"\n🔄 Running test {i+1}/{num_runs}...\n")

        row = [i + 1]
        for script in scripts:
            metrics = run_script(script)
            row.extend([metrics[key] for key in metric_keys] + [metrics["Execution Time"]])
        results.append(row)

# Create CSV header
header = ["Iteration"]
for script in scripts:
    header += [f"{script} - {key}" for key in metric_keys + ["Execution Time"]]

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results)

print(f"✅ Execution results saved in {csv_filename}")
print(f"📜 Full log saved in {log_filename}")
print("🔍 Check the log file for detailed output of each script.")
