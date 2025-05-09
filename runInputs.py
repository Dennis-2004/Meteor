import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_RUNS = 500

def run_input_script(id):
    try:
        subprocess.run(["python3", "input.py", str(id)], check=True)
        return "input.py finished"
    except subprocess.CalledProcessError as e:
        return f"input.py failed: {e}"

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=NUM_RUNS) as executor:
        futures = [executor.submit(run_input_script, i%128) for i in range(NUM_RUNS)]

        for future in as_completed(futures):
            print(future.result())
