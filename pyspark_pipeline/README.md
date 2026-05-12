# PySpark Traffic Pipeline

This directory contains a local-first Spark workflow for the traffic project.

## What it does

- loads the raw traffic CSV
- cleans and deduplicates rows
- engineers lag, rolling, and cyclical time features
- exports a feature table and cleaned CSV
- optionally trains Spark ML regression and congestion classification models

## Usage

Install dependencies first, including `pyspark`, then run:

```bash
python -m pyspark_pipeline.traffic_pipeline
```

Optional flags:

- `--no-models` to only generate features
- `--no-clean-csv` to skip writing the cleaned CSV
- `--input`, `--clean-output`, `--feature-output`, `--summary-output`, `--model-dir` to override paths

The default output summary is written to `report/spark_pipeline_summary.json`.
