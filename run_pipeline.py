import argparse
import subprocess
import sys
from pathlib import Path

# Orchestrates preprocess → features → train → predict end-to-end

def run_step(step_name: str, command: str):
    """Run a pipeline step."""
    print(f"\n{'='*50}")
    print(f"Running {step_name}...")
    print(f"{'='*50}")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"{step_name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error in {step_name}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the datathon forecasting pipeline")
    parser.add_argument("--portfolio", default="all", help="Portfolio to process (default: all)")
    parser.add_argument("--output", default="outputs/forecast_v01.csv", help="Output forecast file")
    parser.add_argument("--target_month", default="2023-08", help="Target month for forecast (YYYY-MM)")

    args = parser.parse_args()

    run_step("Preprocessing", "python3 src/preprocess.py")

    # Step 2: Feature Engineering
    run_step("Feature Engineering", "python3 src/features.py")

    # Step 3: Training
    run_step("Model Training", f"python3 src/train.py")

    # Step 4: Prediction
    run_step("Prediction", f"python3 src/predict.py --target_month {args.target_month} --output {args.output}")

    # Step 5: Evaluation (optional)
    run_step("Evaluation", "python3 src/evaluate.py")

    print(f"\n{'='*50}")
    print("Pipeline completed successfully!")
    print(f"Forecast saved to: {args.output}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()