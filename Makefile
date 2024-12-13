.PHONY: all clean

all: data/raw/heart_failure_clinical_records.data \
	data/processed/heart_failure_train.csv \
	results/figures/correlation_heatmap.png \
	results/models/pipeline.pickle results/figures/training_plots \
	results/tables/confusion_matrix.csv \
	results/tables/test_scores.csv \
	reports/heart-failure-analysis.html \
	reports/heart-failure-analysis.pdf \

# Download and convert data
data/raw/heart_failure_clinical_records.data: scripts/download_and_convert.py
	python scripts/download_and_convert.py \
		--url="https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip" \
		--write_to=data/raw

# Process and analyze data
data/processed/heart_failure_train.csv data/processed/heart_failure_test.csv : scripts/process_and_analyze.py data/raw/heart_failure_clinical_records.data
	python scripts/process_and_analyze.py \
		--file_path="data/raw/heart_failure_clinical_records_dataset_converted.csv" \
		--output_dir=data/processed

# Perform correlation analysis
results/figures/correlation_heatmap.png : scripts/correlation_analysis.py data/processed/heart_failure_train.csv data/processed/heart_failure_test.csv
	python scripts/correlation_analysis.py \
		--train_file=data/processed/heart_failure_train.csv \
		--test_file=data/processed/heart_failure_test.csv \
		--output_file="./results/figures/heatmap.png"

# Train and evaluate the model
results/models/pipeline.pickle results/figures/training_plots: data/processed/heart_failure_train.csv
	python scripts/modelling.py \
		--training-data "./data/processed/heart_failure_train.csv" \
		--pipeline-to "results/models" \
		--plot-to "results/figures" \
		--seed 123

results/tables/confusion_matrix.csv results/tables/test_scores.csv: scripts/model_evaluation.py data/processed/heart_failure_test.csv results/models/pipeline.pickle
	python scripts/model_evaluation.py \
		--scaled-test-data "data/processed/heart_failure_test.csv" \
		--pipeline-from "results/models/pipeline.pickle" \
		--results-to "results/tables"

# Build HTML and PDF reports
# Rule to generate HTML
reports/heart-failure-analysis.html:
	quarto render reports/heart-failure-analysis.qmd --to html --embed-resources --standalone

# Rule to generate PDF
reports/heart-failure-analysis.pdf:
	quarto render reports/heart-failure-analysis.qmd --to pdf


# Clean up analysis
clean:
	rm -rf data/raw/* \
		data/processed/* \
		results/figures/* \
		results/img/* \
		results/models/* \
		results/pipeline/* \
		
	rm -f \
		results/tables/test_scores.csv \
		results/tables/confusion_matrix.csv \
		reports/heart-failure-analysis.html \
		reports/heart-failure-analysis.pdf
		

