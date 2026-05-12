"""PySpark pipeline for traffic data preparation and Spark ML training.

This module provides a reproducible end-to-end Spark workflow for the project:

1. Load the raw traffic CSV.
2. Clean and deduplicate the data.
3. Engineer time-series features with Spark SQL windows.
4. Export a feature table and cleaned CSV.
5. Optionally train Spark ML regression and congestion-classification models.

The implementation is intentionally lightweight and local-first so it can run on
the project dataset without requiring a cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_INPUT = ROOT / "data" / "raw_traffic.csv"
DEFAULT_CLEAN_OUTPUT = ROOT / "data" / "clean_traffic.csv"
DEFAULT_FEATURE_OUTPUT = ROOT / "data" / "spark_traffic_features.parquet"
DEFAULT_SUMMARY_OUTPUT = ROOT / "report" / "spark_pipeline_summary.json"
DEFAULT_MODEL_DIR = ROOT / "models" / "spark"


def build_spark(app_name: str = "AI Smart City Traffic Pipeline") -> SparkSession:
    """Create a local Spark session for repeatable offline execution."""

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_traffic_data(spark: SparkSession, input_path: Path) -> DataFrame:
    """Load the raw traffic CSV with an explicit schema."""

    schema = T.StructType(
        [
            T.StructField("DateTime", T.StringType(), True),
            T.StructField("Junction", T.IntegerType(), True),
            T.StructField("Vehicles", T.IntegerType(), True),
            T.StructField("ID", T.StringType(), True),
        ]
    )

    df = spark.read.csv(str(input_path), header=True, schema=schema)
    return df.withColumn("DateTime", F.to_timestamp("DateTime"))


def _cyclical_features(df: DataFrame, column: str, period: int, prefix: str) -> DataFrame:
    return (
        df.withColumn(f"{prefix}_sin", F.sin(2 * F.lit(3.141592653589793) * F.col(column) / F.lit(period)))
        .withColumn(f"{prefix}_cos", F.cos(2 * F.lit(3.141592653589793) * F.col(column) / F.lit(period)))
    )


def engineer_features(df: DataFrame) -> DataFrame:
    """Clean the data and add lag, rolling, and calendar features."""

    ordered = df.dropna(subset=["DateTime", "Junction", "Vehicles"]).dropDuplicates(["DateTime", "Junction"])
    ordered = ordered.filter(F.col("Vehicles") >= 0)

    ordered = (
        ordered.withColumn("Hour", F.hour("DateTime"))
        .withColumn("Day", F.dayofmonth("DateTime"))
        .withColumn("Month", F.month("DateTime"))
        .withColumn("Weekday", F.dayofweek("DateTime") - F.lit(1))
        .withColumn("IsWeekend", F.when(F.col("Weekday") >= 5, F.lit(1)).otherwise(F.lit(0)))
    )

    ordered = _cyclical_features(ordered, "Hour", 24, "Hour")
    ordered = _cyclical_features(ordered, "Month", 12, "Month")

    junction_window = Window.partitionBy("Junction").orderBy("DateTime")
    rolling_window = junction_window.rowsBetween(-3, -1)
    long_window = junction_window.rowsBetween(-24, -1)

    ordered = (
        ordered.withColumn("Lag_1", F.lag("Vehicles", 1).over(junction_window))
        .withColumn("Lag_2", F.lag("Vehicles", 2).over(junction_window))
        .withColumn("Lag_3", F.lag("Vehicles", 3).over(junction_window))
        .withColumn("Lag_24", F.lag("Vehicles", 24).over(junction_window))
        .withColumn("Rolling_Mean_3", F.avg("Vehicles").over(rolling_window))
        .withColumn("Rolling_Std_3", F.stddev("Vehicles").over(rolling_window))
        .withColumn("Rolling_Mean_24", F.avg("Vehicles").over(long_window))
        .withColumn("Rolling_Max_24", F.max("Vehicles").over(long_window))
    )

    ordered = ordered.withColumn(
        "congestion_label",
        F.when(F.col("Vehicles") < 80, F.lit(0))
        .when(F.col("Vehicles") < 250, F.lit(1))
        .otherwise(F.lit(2)),
    )

    numeric_columns = [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_24",
        "Rolling_Mean_3",
        "Rolling_Std_3",
        "Rolling_Mean_24",
        "Rolling_Max_24",
    ]

    return ordered.dropna(subset=numeric_columns)


def split_chronologically(df: DataFrame, test_ratio: float = 0.2) -> tuple[DataFrame, DataFrame]:
    """Split rows using chronological order across all junctions."""

    ordered = df.orderBy("DateTime", "Junction", "ID")
    indexed = ordered.withColumn("row_num", F.row_number().over(Window.orderBy("DateTime", "Junction", "ID")))
    total_rows = indexed.count()
    split_index = int(total_rows * (1 - test_ratio))
    train = indexed.filter(F.col("row_num") <= split_index).drop("row_num")
    test = indexed.filter(F.col("row_num") > split_index).drop("row_num")
    return train, test


def _regression_features() -> List[str]:
    return [
        "Junction",
        "Hour",
        "Day",
        "Month",
        "Weekday",
        "IsWeekend",
        "Hour_sin",
        "Hour_cos",
        "Month_sin",
        "Month_cos",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_24",
        "Rolling_Mean_3",
        "Rolling_Std_3",
        "Rolling_Mean_24",
        "Rolling_Max_24",
    ]


def _classification_features() -> List[str]:
    return [
        "Junction",
        "Hour",
        "Day",
        "Month",
        "Weekday",
        "IsWeekend",
        "Hour_sin",
        "Hour_cos",
        "Month_sin",
        "Month_cos",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_24",
        "Rolling_Mean_3",
        "Rolling_Std_3",
        "Rolling_Mean_24",
        "Rolling_Max_24",
    ]


def train_regression_model(train_df: DataFrame, test_df: DataFrame, model_dir: Path) -> Dict[str, float]:
    """Train a Spark RandomForest regressor and return evaluation metrics."""

    feature_cols = _regression_features()
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{name}_imputed" for name in feature_cols])

    imputed_cols = [f"{name}_imputed" for name in feature_cols]
    pipeline = Pipeline(
        stages=[
            imputer,
            VectorAssembler(inputCols=imputed_cols, outputCol="features"),
            RandomForestRegressor(labelCol="Vehicles", featuresCol="features", numTrees=100, maxDepth=8, seed=42),
        ]
    )

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    rmse = RegressionEvaluator(labelCol="Vehicles", predictionCol="prediction", metricName="rmse").evaluate(predictions)
    mae = RegressionEvaluator(labelCol="Vehicles", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="Vehicles", predictionCol="prediction", metricName="r2").evaluate(predictions)
    mse = RegressionEvaluator(labelCol="Vehicles", predictionCol="prediction", metricName="mse").evaluate(predictions)

    model_path = model_dir / "regression"
    model_saved = False
    model_save_error = "Spark model persistence skipped because HADOOP_HOME is not configured on Windows."
    if _native_spark_writes_available():
        try:
            model.write().overwrite().save(str(model_path))
            model_saved = True
            model_save_error = ""
        except Exception as exc:
            model_save_error = str(exc)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mse": float(mse),
        "model_path": str(model_path),
        "model_saved": model_saved,
        "model_save_error": model_save_error,
    }


def train_classification_model(train_df: DataFrame, test_df: DataFrame, model_dir: Path) -> Dict[str, float]:
    """Train a Spark RandomForest classifier for congestion labels."""

    feature_cols = _classification_features()
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{name}_imputed" for name in feature_cols])
    imputed_cols = [f"{name}_imputed" for name in feature_cols]

    pipeline = Pipeline(
        stages=[
            imputer,
            VectorAssembler(inputCols=imputed_cols, outputCol="features"),
            RandomForestClassifier(labelCol="congestion_label", featuresCol="features", numTrees=120, maxDepth=8, seed=42),
        ]
    )

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    accuracy = MulticlassClassificationEvaluator(labelCol="congestion_label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(labelCol="congestion_label", predictionCol="prediction", metricName="f1").evaluate(predictions)
    weighted_precision = MulticlassClassificationEvaluator(labelCol="congestion_label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
    weighted_recall = MulticlassClassificationEvaluator(labelCol="congestion_label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)

    model_path = model_dir / "classification"
    model_saved = False
    model_save_error = "Spark model persistence skipped because HADOOP_HOME is not configured on Windows."
    if _native_spark_writes_available():
        try:
            model.write().overwrite().save(str(model_path))
            model_saved = True
            model_save_error = ""
        except Exception as exc:
            model_save_error = str(exc)

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "model_path": str(model_path),
        "model_saved": model_saved,
        "model_save_error": model_save_error,
    }


def _to_pandas_csv(df: DataFrame, output_path: Path) -> None:
    pdf = df.toPandas()
    pdf.to_csv(output_path, index=False)


def _native_spark_writes_available() -> bool:
    """Return True when the local environment is likely able to write Spark/Hadoop files."""

    if os.name != "nt":
        return True
    return bool(os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir"))


def run_pipeline(
    input_path: Path,
    clean_output: Path,
    feature_output: Path,
    summary_output: Path,
    model_dir: Path,
    train_models: bool = True,
    write_clean_csv: bool = True,
) -> Dict[str, object]:
    """Execute the Spark pipeline end-to-end and persist artifacts."""

    spark = build_spark()
    try:
        raw = load_traffic_data(spark, input_path)
        features = engineer_features(raw)

        feature_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_feature_output = feature_output
        feature_output_format = "spark_parquet" if _native_spark_writes_available() else "pandas_csv_fallback"
        feature_output_warning = ""
        if _native_spark_writes_available():
            try:
                features.write.mode("overwrite").parquet(str(feature_output))
            except Exception as exc:
                feature_output_format = "pandas_csv_fallback"
                feature_output_warning = str(exc)
                if resolved_feature_output.exists() and resolved_feature_output.is_dir():
                    shutil.rmtree(resolved_feature_output, ignore_errors=True)
                resolved_feature_output = resolved_feature_output.with_suffix(".csv")
                _to_pandas_csv(features, resolved_feature_output)
        else:
            resolved_feature_output = resolved_feature_output.with_suffix(".csv")
            _to_pandas_csv(features, resolved_feature_output)
            feature_output_warning = "Spark filesystem writes skipped because HADOOP_HOME is not configured on Windows."

        if write_clean_csv:
            clean_output.parent.mkdir(parents=True, exist_ok=True)
            ordered_clean = (
                features.select("DateTime", "Junction", "Vehicles", "ID", "Hour", "Month", "Day", "Weekday")
                .orderBy("DateTime", "Junction", "ID")
            )
            _to_pandas_csv(ordered_clean, clean_output)

        train_df, test_df = split_chronologically(features)

        summary: Dict[str, object] = {
            "input_path": str(input_path),
            "row_count": features.count(),
            "junction_count": features.select("Junction").distinct().count(),
            "date_min": features.select(F.min("DateTime")).first()[0].isoformat(),
            "date_max": features.select(F.max("DateTime")).first()[0].isoformat(),
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "feature_output": str(resolved_feature_output),
            "feature_output_format": feature_output_format,
            "feature_output_warning": feature_output_warning,
            "native_spark_writes_available": _native_spark_writes_available(),
        }

        if train_models:
            model_dir.mkdir(parents=True, exist_ok=True)
            summary["regression"] = train_regression_model(train_df, test_df, model_dir)
            summary["classification"] = train_classification_model(train_df, test_df, model_dir)

        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        spark.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI Smart City PySpark traffic pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_INPUT, help="Path to the raw traffic CSV")
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT, help="Path for the cleaned CSV")
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT, help="Output parquet path for features")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT, help="Path for the JSON summary")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory for Spark ML models")
    parser.add_argument("--no-models", action="store_true", help="Skip model training and only build features")
    parser.add_argument("--no-clean-csv", action="store_true", help="Skip writing the cleaned CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_pipeline(
        input_path=args.input,
        clean_output=args.clean_output,
        feature_output=args.feature_output,
        summary_output=args.summary_output,
        model_dir=args.model_dir,
        train_models=not args.no_models,
        write_clean_csv=not args.no_clean_csv,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())