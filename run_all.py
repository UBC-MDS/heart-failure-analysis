import os

# Command 1: Download and convert the dataset
command_1 = """
python scripts/download_and_convert.py \
  --url "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip" \
  --write_to "data/raw"
"""

# Command 2: Process and analyze the dataset
command_2 = """
python scripts/process_and_analyze.py \
  --file_path "./data/raw/heart_failure_clinical_records_dataset_converted.csv" \
  --output_dir "./data/processed"
"""

# Command 3: Perform correlation analysis
command_3 = """
python scripts/correlation_analysis.py \
  --train_file "./data/processed/heart_failure_train.csv" \
  --test_file "./data/processed/heart_failure_test.csv" \
  --output_file "./results/figures/heatmap.png"
"""

# Command 4: Run the modeling script
command_4 = """
python scripts/modelling.py \
  --training-data "./data/processed/heart_failure_train.csv" \
  --pipeline-to "results/pipeline" \
  --plot-to "results/figures" \
  --seed 123
"""

# Command 5: Evaluate the model
command_5 = """
python scripts/model_evaluation.py \
  --scaled-test-data "data/processed/heart_failure_test.csv" \
  --pipeline-from "results/pipeline/pipeline.pickle" \
  --results-to "results/figures"
"""

# Execute all commands
commands = [command_1, command_2, command_3, command_4, command_5]

for i, cmd in enumerate(commands, 1):
    print(f"Running Command {i}...")
    result = os.system(cmd)
    if result != 0:
        print(f"Error occurred while running Command {i}. Exiting...")
        exit(1)
    print(f"Command {i} executed successfully.\n")
