import os

# Command 1: Download and convert the dataset
command_1 = """
python src/download_and_convert.py \
  --url "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
"""

# Command 2: Process and analyze the dataset
command_2 = """
python src/process_and_analyze.py \
  --file_path "../data/heart_failure_clinical_records_dataset_converted.csv"
"""

# Command 3: Perform correlation analysis
command_3 = """
python src/correlation_analysis.py \
  --train_file "./data/processed/heart_failure_train.csv" \
  --test_file "./data/processed/heart_failure_test.csv" \
  --output_file "./results/figures/heatmap.png"
"""

# Command 4: Run the modeling script
command_4 = """
python src/modelling.py \
  --training-data "./data/processed/heart_failure_train.csv" \
  --pipeline-to "results/pipeline" \
  --plot-to "results/figures" \
  --seed 123
"""

# Command 5: Evaluate the model
command_5 = """
python src/model_evaluation.py \
  --scaled-test-data "data/processed/heart_failure_test.csv" \
  --pipeline-from "results/pipeline/heart_failure_model.pickle" \
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
