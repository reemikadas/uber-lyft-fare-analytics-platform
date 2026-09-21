# Databricks notebook source
# MAGIC %md
# MAGIC # Ride Fare Prediction
# MAGIC
# MAGIC **Project:** Uber/Lyft Databricks ELT Pipeline  
# MAGIC **Source:** `rideshare_elt.gold.rides_weather_enriched`  
# MAGIC **Target:** `price`  
# MAGIC **Task:** Supervised regression
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Build and evaluate regression models that predict the quoted ride price using service, source, destination, distance, time, surge, and weather features.
# MAGIC
# MAGIC ## Modeling principles
# MAGIC
# MAGIC - Maintain one row per valid fare quote.
# MAGIC - Split the data before fitting preprocessing steps.
# MAGIC - Prevent target leakage.
# MAGIC - Exclude `price_per_mile` because it is calculated using the target price.
# MAGIC - Compare every model with a simple baseline.
# MAGIC - Evaluate performance using MAE, RMSE, and R².

# COMMAND ----------

from pyspark.sql import functions as F

gold_table = "rideshare_elt.gold.rides_weather_enriched"

if not spark.catalog.tableExists(gold_table):
    raise ValueError(f"Required Gold table was not found: {gold_table}")

model_source_df = spark.table(gold_table)

print(f"Source table: {gold_table}")
print(f"Rows: {model_source_df.count():,}")
print(f"Columns: {len(model_source_df.columns)}")

display(model_source_df.limit(10))

# COMMAND ----------

# validate the modeling grain and target
expected_rows = 637_976

source_validation = (
    model_source_df
    .agg(
        F.count("*").alias("total_rows"),
        F.countDistinct("id").alias("unique_quote_ids"),
        F.sum(F.col("price").isNull().cast("int")).alias("null_target_rows"),
        F.sum((F.col("price") < 0).cast("int")).alias("negative_target_rows"),
        F.min("price").alias("minimum_price"),
        F.max("price").alias("maximum_price")
    )
    .first()
)

if source_validation["total_rows"] != expected_rows:
    raise ValueError("Unexpected Gold row count.")

if source_validation["unique_quote_ids"] != expected_rows:
    raise ValueError("Duplicate or missing fare-quote IDs detected.")

if source_validation["null_target_rows"] != 0:
    raise ValueError("The target price contains null values.")

if source_validation["negative_target_rows"] != 0:
    raise ValueError("The target price contains negative values.")

print("Modeling source validation passed.")
print(f"Rows: {source_validation["total_rows"]:,}")
print(f"Unique quote IDs: {source_validation["unique_quote_ids"]:,}")
print(f"Null Target rows: {source_validation["null_target_rows"]:,}")
print(f"Negative Target rows: {source_validation["negative_target_rows"]:,}")
print(f"Price range: ${source_validation["minimum_price"]:,.2f} - ${source_validation["maximum_price"]:,.2f}")

# COMMAND ----------

# Remove leakage and pipeline metadata
modeling_df = (
    model_source_df
    .drop(
        "price_per_mile", # Target Leakage: calculated using price
        "_gold_created_at" # Pipeline metadata
    )
)

print(f"Source Columns: {len(model_source_df.columns)}")
print(f"Modeling Columns: {len(modeling_df.columns)}")

# COMMAND ----------

# Classify the columns
target_column = "price"

numeric_features = [
    "distance",
    "surge_multiplier",
    "query_hour_local",
    "source_temperature",
    "source_clouds",
    "source_pressure",
    "source_rain_amount",
    "source_humidity",
    "source_wind",
    "destination_temperature",
    "destination_clouds",
    "destination_pressure",
    "destination_rain_amount",
    "destination_humidity",
    "destination_wind",
    "average_endpoint_temperature",
    "endpoint_temperature_difference"
]

categorical_features = [
    "name",
    "source",
    "destination",
    "query_day_name_local",
    "is_weekend",
    "source_is_raining",
    "destination_is_raining",
    "rain_at_either_location"
]

excluded_columns = [
    "id",
    "product_id",
    "cab_type",
    "route",                    # Derived from source and destination
    "query_datetime_utc",
    "query_datetime_local",
    "query_date_local",         # Used only for chronological splitting
    "surge_applied"            # Redundant with surge_multiplier
]

print(f"Target Column: {target_column}")
print(f"Numeric Features: {len(numeric_features)}")
print(f"Categorical Features: {len(categorical_features)}")
print(f"Excluded Columns: {len(excluded_columns)}")

# COMMAND ----------

selected_columns = (
    [target_column]
    + numeric_features
    + categorical_features
    + excluded_columns
)

null_expressions = [
    F.sum(F.when(F.col(column_name).isNull(), 1).otherwise(0)).alias(column_name)
    for column_name in selected_columns
]

feature_null_summary_df = modeling_df.agg(*null_expressions)

print("Missing values in selected modeling columns:")
display(feature_null_summary_df)

# COMMAND ----------

# Check categorical cardinality
cardinality_expressions = [
    F.countDistinct(column_name).alias(column_name)
    for column_name in categorical_features
]

categorical_cardinality_df = modeling_df.agg(*cardinality_expressions)

print("Distinct values in categorical features:")
display(categorical_cardinality_df)

# COMMAND ----------

# Analyze the target distribution
price_summary_df = (
    modeling_df
    .agg(
        F.count("*").alias("fare_quotes"),
        F.round(F.avg("price"), 2).alias("average_price"),
        F.round(F.expr("percentile_approx(price, 0.25)"), 2).alias("price_p25"),
        F.round(F.expr("percentile_approx(price, 0.50)"), 2).alias("price_p50"),
        F.round(F.expr("percentile_approx(price, 0.75)"), 2).alias("price_p75"),
        F.round(F.stddev("price"), 2).alias("price_stddev"),
        F.min("price").alias("minimum_price"),
        F.max("price").alias("maximum_price")
    )
)

display(price_summary_df)

# COMMAND ----------

# Fare distribution by provider and product
price_by_product_df = (
    modeling_df
    .groupBy("cab_type", "name")
    .agg(
        F.count("*").alias("fare_quotes"),
        F.round(F.avg("price"), 2).alias("average_price"),
        F.round(F.expr("percentile_approx(price, 0.50)"), 2).alias("median_price"),
        F.min("price").alias("minimum_price"),
        F.max("price").alias("maximum_price")
    )
    .orderBy("cab_type", "median_price")
)

display(price_by_product_df)

# COMMAND ----------

# Calculate correlation with price
correlation_results = []

for feature_name in numeric_features:
    correlation_value = modeling_df.stat.corr(
        target_column,
        feature_name
    )

    correlation_results.append(
        (
            feature_name,
            float(correlation_value)
        )
    )

price_correlation_df = (
    spark.createDataFrame(
        correlation_results,
        [
            "feature",
            "correlation_with_price"
        ]
    )
    .withColumn(
        "absolute_correlation",
        F.abs("correlation_with_price")
    )
    .withColumn(
        "correlation_with_price",
        F.round("correlation_with_price", 4)
    )
    .withColumn(
        "absolute_correlation",
        F.round("absolute_correlation", 4)
    )
    .orderBy(F.desc("absolute_correlation"))
)

display(price_correlation_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

correlation_columns = (
    [target_column]
    + numeric_features
)

correlation_assembler = VectorAssembler(
    inputCols=correlation_columns,
    outputCol="correlation_features",
    handleInvalid="error"
)

correlation_vector_df = (
    correlation_assembler
    .transform(modeling_df.select(*correlation_columns))
    .select("correlation_features")
)

correlation_matrix = (
    Correlation
    .corr(
        correlation_vector_df,
        "correlation_features",
        "pearson"
    )
    .first()[0]
    .toArray()
)

correlation_matrix_pd = pd.DataFrame(
    correlation_matrix,
    index=correlation_columns,
    columns=correlation_columns
)

display(correlation_matrix_pd.round(3))

# COMMAND ----------

plt.figure(figsize=(16,12))

sns.heatmap(
    correlation_matrix_pd,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Correlation Matrix: Quoted Price and Numeric Features",
    fontsize=14
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Feature Selection
# MAGIC
# MAGIC The correlation matrix identified substantial duplication between source and
# MAGIC destination weather measurements. The final feature set retains one
# MAGIC representative measurement or an engineered endpoint feature.

# COMMAND ----------

# Final feature set after correlation analysis

numeric_features = [
    "distance",
    "surge_multiplier",
    "query_hour_local",
    "source_clouds",
    "source_pressure",
    "source_rain_amount",
    "source_humidity",
    "source_wind",
    "average_endpoint_temperature",
    "endpoint_temperature_difference"
]

categorical_features = [
    "name",
    "source",
    "destination",
    "query_day_name_local",
    "is_weekend",
    "rain_at_either_location"
]

excluded_columns = [
    "id",
    "product_id",
    "cab_type",
    "route",
    "query_datetime_utc",
    "query_datetime_local",
    "query_date_local",
    "surge_applied",

    # Removed after correlation analysis
    "source_temperature",
    "destination_temperature",
    "destination_clouds",
    "destination_pressure",
    "destination_rain_amount",
    "destination_humidity",
    "destination_wind",
    "source_is_raining",
    "destination_is_raining"
]

selected_columns = (
    [target_column]
    + numeric_features
    + categorical_features
    + excluded_columns
)

print(f"Target columns: {len([target_column])}")
print(f"Final numeric features: {len(numeric_features)}")
print(f"Final categorical features: {len(categorical_features)}")
print(f"Excluded columns: {len(excluded_columns)}")
print(f"Total classified columns: {len(selected_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chronological Train, Validation, and Test Split
# MAGIC
# MAGIC Records are split chronologically to prevent future observations from
# MAGIC influencing models trained on earlier data. The test set remains untouched
# MAGIC until final model evaluation.

# COMMAND ----------

observed_dates = [
    row["query_date_local"]
    for row in (
        modeling_df
        .select("query_date_local")
        .distinct()
        .orderBy("query_date_local")
        .collect()
    )
]

number_of_dates = len(observed_dates)

train_date_count = int(number_of_dates * 2 / 3) # (18 * 2/3) = 12
remaining_date_count = number_of_dates - train_date_count # 18 - 12 = 6
validation_date_count = remaining_date_count // 2 # 6 // 2 = 3

train_end_date = observed_dates[train_date_count - 1]
validation_end_date = observed_dates[train_date_count + validation_date_count - 1]

train_df = modeling_df.filter(
    F.col("query_date_local") <= F.lit(train_end_date)
)

validation_df = modeling_df.filter(
    (F.col("query_date_local") > F.lit(train_end_date))
    &
    (F.col("query_date_local") <= F.lit(validation_end_date))
)

test_df = modeling_df.filter(
    F.col("query_date_local") > F.lit(validation_end_date)
)

print(f"Observed dates: {number_of_dates}")
print(f"Training through: {train_end_date}")
print(f"Validation through: {validation_end_date}")
print(f"Testing begins: {observed_dates[train_date_count + validation_date_count]}")

# COMMAND ----------

split_summary_df = (
    train_df.select(
        F.lit("Train").alias("split"),
        F.count("*").alias("rows"),
        F.min("query_date_local").alias("start_date"),
        F.max("query_date_local").alias("end_date")
    )
    .unionByName(
        validation_df.select(
            F.lit("Validation").alias("split"),
            F.count("*").alias("rows"),
            F.min("query_date_local").alias("start_date"),
            F.max("query_date_local").alias("end_date")
        )
    )
    .unionByName(
        test_df.select(
            F.lit("Test").alias("split"),
            F.count("*").alias("rows"),
            F.min("query_date_local").alias("start_date"),
            F.max("query_date_local").alias("end_date")
        )
    )
    .withColumn(
        "row_percentage",
        F.round(F.col("rows") / modeling_df.count() * 100, 2)
    )
)

display(split_summary_df)

# COMMAND ----------

split_row_count = (
    train_df.count()
    + validation_df.count()
    + test_df.count()
)

assert split_row_count == modeling_df.count(), (
    "Split row counts does not match the modeling dataset."
)

print(f"Chronological split validation passed: {split_row_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Preprocessing
# MAGIC
# MAGIC Multi-category string features are indexed and one-hot encoded. Boolean
# MAGIC features are converted to binary numeric values. The preprocessing pipeline is
# MAGIC fitted only on the training set to prevent data leakage.

# COMMAND ----------

# Define preprocessing groups
from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, VectorAssembler)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

string_categorical_features = [
    "name",
    "source",
    "destination",
    "query_day_name_local"
]

boolean_features = [
    "is_weekend",
    "rain_at_either_location"
]

boolean_numeric_features = [
    f"{column_name}_numeric"
    for column_name in boolean_features
]

model_numeric_features = (
    numeric_features
    + boolean_numeric_features
)

print(f"Numeric model features: {len(model_numeric_features)}")
print(f"String categorical features: {len(string_categorical_features)}")

# COMMAND ----------

def convert_boolean_features(input_df):
    output_df = input_df

    for column_name in boolean_features:
        output_df = output_df.withColumn(
            f"{column_name}_numeric",
            F.col(column_name).cast("double")
        )
    
    return output_df

train_input_df = convert_boolean_features(train_df)
validation_input_df = convert_boolean_features(validation_df)
test_input_df = convert_boolean_features(test_df)

# Verify
display(
    train_input_df.select(
        "is_weekend",
        "is_weekend_numeric",
        "rain_at_either_location",
        "rain_at_either_location_numeric"
    ).distinct()
)

# COMMAND ----------

# Create the preprocessing stages
indexed_feature_columns = [
    f"{column_name}_index"
    for column_name in string_categorical_features
]

encoded_feature_columns = [
    f"{column_name}_encoded"
    for column_name in string_categorical_features
]

categorical_indexers = [
    StringIndexer(
        inputCol=column_name,
        outputCol=f"{column_name}_index",
        handleInvalid="keep"
    )
    for column_name in string_categorical_features
]

one_hot_encoder = OneHotEncoder(
    inputCols=indexed_feature_columns,
    outputCols=encoded_feature_columns,
    handleInvalid="keep",
    dropLast=True
)

feature_assembler = VectorAssembler(
    inputCols=(model_numeric_features + encoded_feature_columns),
    outputCol="unscaled_features",
    handleInvalid="error"
)

preprocessing_pipeline = Pipeline(
    stages=(
        categorical_indexers
        + [one_hot_encoder, feature_assembler]
    )
)

# COMMAND ----------

# Fit only on training data
preprocessing_model = preprocessing_pipeline.fit(train_input_df)

train_prepared_df = preprocessing_model.transform(train_input_df)
validation_prepared_df = preprocessing_model.transform(validation_input_df)
test_prepared_df = preprocessing_model.transform(test_input_df)

# COMMAND ----------

# Validate the prepared datasets
prepared_split_summary = [
    (
        "Train",
        train_prepared_df.count(),
        train_df.count()
    ),
    (
        "Validation",
        validation_prepared_df.count(),
        validation_df.count()
    ),
    (
        "Test",
        test_prepared_df.count(),
        test_df.count()
    )
]

for split_name, prepared_rows, original_rows in prepared_split_summary:
    assert prepared_rows == original_rows, (
        f"{split_name} row count changed during preprocessing."
    )

    print(
        f"{split_name}: {prepared_rows:,} rows retained"
    )

# COMMAND ----------

feature_vector_size = (
    train_prepared_df
    .select(
        F.size(
            vector_to_array("unscaled_features")
        ).alias("feature_count")
    )
    .first()["feature_count"]
)

print(f"Assembled feature count: {feature_vector_size}")

display(train_prepared_df.select(
    target_column,
    "unscaled_features"
    ).limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Model
# MAGIC
# MAGIC A constant baseline predicts the median training price for every fare quote.
# MAGIC Subsequent regression models must outperform this baseline on the validation
# MAGIC period.

# COMMAND ----------

# Create a regression-evaluation function
from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_regression_predictions(predictions_df, model_name, split_name):
    metric_results = {"model": model_name,
                      "split": split_name}
    
    for metric_name in ["mae", "rmse", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol=target_column,
            predictionCol="prediction",
            metricName=metric_name
        )

        metric_results[metric_name] = float(
            evaluator.evaluate(predictions_df)
        )

    return metric_results

# COMMAND ----------

# Calculate the training median
training_median_price = (
    train_df
    .approxQuantile(
        target_column,
        [0.5],
        0.001
    )[0]
)

print(f"Training median price: ${training_median_price:,.2f}")

# COMMAND ----------

# Generate baseline predictions
baseline_train_predictions_df = (
    train_prepared_df
    .withColumn(
        "prediction",
        F.lit(training_median_price)
    )
)

baseline_validation_predictions_df = (
    validation_prepared_df
    .withColumn(
        "prediction",
        F.lit(training_median_price)
    )
)

# COMMAND ----------

# Evaluate the baseline
baseline_results = [
    evaluate_regression_predictions(
        baseline_train_predictions_df,
        "Training Median Baseline",
        "Train"
    ),
    evaluate_regression_predictions(
        baseline_validation_predictions_df,
        "Training Median Baseline",
        "Validation"
    )
]

baseline_results_df = (
    spark.createDataFrame(baseline_results)
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
)

display(baseline_results_df)

# COMMAND ----------

training_mean_price = (
    train_df
    .agg(
        F.avg(target_column).alias("mean_price")
    )
    .first()["mean_price"]
)

print(f"Training mean price: ${training_mean_price:,.2f}")

# COMMAND ----------

mean_baseline_train_predictions_df = (
    train_prepared_df
    .withColumn(
        "prediction",
        F.lit(training_mean_price)
    )
)

mean_baseline_validation_predictions_df = (
    validation_prepared_df
    .withColumn(
        "prediction",
        F.lit(training_mean_price)
    )
)

# COMMAND ----------

# Evaluate and combine both baselines
mean_baseline_results = [
    evaluate_regression_predictions(
        mean_baseline_train_predictions_df,
        "Training Mean Baseline",
        "Train"
    ),
    evaluate_regression_predictions(
        mean_baseline_validation_predictions_df,
        "Training Mean Baseline",
        "Validation"
    )
]

all_baseline_results = (
    baseline_results + mean_baseline_results
)

all_baseline_results_df = (
    spark.createDataFrame(all_baseline_results)
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
    .orderBy("model", "split")
)

display(all_baseline_results_df)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

feature_scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withStd=True,
    withMean=False
)

feature_scaler_model = feature_scaler.fit(train_prepared_df)

train_scaled_df = feature_scaler_model.transform(train_prepared_df)
validation_scaled_df = feature_scaler_model.transform(validation_prepared_df)
test_scaled_df = feature_scaler_model.transform(test_prepared_df)

print("Feature scaling completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Linear regression provides an interpretable first model and measures how much
# MAGIC of the fare variation can be explained through additive relationships between
# MAGIC the selected predictors and quoted price.

# COMMAND ----------

# Train the initial linear regression
from pyspark.ml.regression import LinearRegression

linear_regression = LinearRegression(
    featuresCol="features",
    labelCol=target_column,
    predictionCol="prediction",
    regParam=0.0,
    elasticNetParam=0.0,
    maxIter=100,
    standardization=False
)

linear_regression_model =linear_regression.fit(train_scaled_df)

print("Linear Regression training completed.")

print(f"Intercept: "
      f"{linear_regression_model.intercept:,.4f}"
    )

print(
    f"Number of Coefficients: "
    f"{len(linear_regression_model.coefficients)}"
)

# COMMAND ----------

# Generate predictions
linear_train_predictions_df = (linear_regression_model.transform(train_scaled_df))
linear_validation_predictions_df = (linear_regression_model.transform(validation_scaled_df))

# COMMAND ----------

# Evaluate linear regression
linear_regression_results = [
    evaluate_regression_predictions(
        linear_train_predictions_df,
        "Linear Regression",
        "Train"
    ),
    evaluate_regression_predictions(
        linear_validation_predictions_df,
        "Linear Regression",
        "Validation"
    )
]

model_comparison_results = (
    all_baseline_results +
    linear_regression_results
)

model_comparison_df = (
    spark.createDataFrame(model_comparison_results)
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
    .orderBy(
        "split",
        "rmse"
    )
)

display(model_comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear Regression Performance
# MAGIC
# MAGIC Linear regression substantially outperformed both constant-price baselines. On the validation set, it achieved an MAE of **$1.74**, an RMSE of **$2.47**, and an R² of **0.9296**, explaining approximately 93% of quoted-price variation. Training and validation results were nearly identical, indicating strong generalization with no evident overfitting across the chronological split.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# Train a Random Forest
from pyspark.ml.regression import RandomForestRegressor

random_forest = RandomForestRegressor(
    featuresCol="unscaled_features",
    labelCol=target_column,
    predictionCol="prediction",
    numTrees=50,
    maxDepth=10,
    minInstancesPerNode=5,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=42
)

random_forest_model = random_forest.fit(train_prepared_df)

print("Random Forest training completed.")

print(f"Number of trees: {random_forest_model.getNumTrees}")

# COMMAND ----------

# Generate train and validation predictions
random_forest_train_predictions_df = (random_forest_model.transform(train_prepared_df))
random_forest_validation_predictions_df = (random_forest_model.transform(validation_prepared_df))

# COMMAND ----------

# Evaluate random forest
random_forest_results = [
    evaluate_regression_predictions(
        random_forest_train_predictions_df,
        "Random Forest",
        "Train"
    ),
    evaluate_regression_predictions(
        random_forest_validation_predictions_df,
        "Random Forest",
        "Validation"
    )
]

model_comparison_results = (
    all_baseline_results +
    linear_regression_results +
    random_forest_results
)

model_comparison_df = (
    spark.createDataFrame(model_comparison_results)
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
    .orderBy(
        "split",
        "rmse"
    )
)

display(model_comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Performance
# MAGIC
# MAGIC Random Forest substantially outperformed the constant-price baselines but did
# MAGIC not outperform Linear Regression. It achieved a validation MAE of **$2.59**,
# MAGIC an RMSE of **$3.37**, and an R² of **0.8685**. Similar training and validation
# MAGIC results indicate stable generalization, although the current model may be
# MAGIC underfitting because of its conservative depth and feature-sampling settings.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Experiments

# COMMAND ----------

random_forest_candidates = [
    {
        "model_name": "Random Forest - Depth 14",
        "numTrees": 75,
        "maxDepth": 14,
        "minInstancesPerNode": 2,
        "featureSubsetStrategy": "onethird",
        "subsamplingRate": 0.9
    },
    {
        "model_name": "Random Forest - Depth 16",
        "numTrees": 100,
        "maxDepth": 16,
        "minInstancesPerNode": 2,
        "featureSubsetStrategy": "all",
        "subsamplingRate": 0.8
    }
]

# COMMAND ----------

# Train and evaluate each candidate
tuned_random_forest_results = []
tuned_random_forest_models = {}

for configuration in random_forest_candidates:
    model_name = configuration["model_name"]

    estimator = RandomForestRegressor(
        featuresCol="unscaled_features",
        labelCol=target_column,
        predictionCol="prediction",
        numTrees=configuration["numTrees"],
        maxDepth=configuration["maxDepth"],
        minInstancesPerNode=configuration["minInstancesPerNode"],
        featureSubsetStrategy=configuration["featureSubsetStrategy"],
        subsamplingRate=configuration["subsamplingRate"],
        seed=42
    )

    fitted_model = estimator.fit(train_prepared_df)

    train_predictions_df = fitted_model.transform(train_prepared_df)
    validation_predictions_df = fitted_model.transform(validation_prepared_df)

    tuned_random_forest_results.extend([
        evaluate_regression_predictions(
            train_predictions_df,
            model_name,
            "Train"
        ),
        evaluate_regression_predictions(
            validation_predictions_df,
            model_name,
            "validation"
        )
    ])

    tuned_random_forest_models[model_name] = fitted_model

    print(f"Completed: {model_name}")

# COMMAND ----------

depth_14_random_forest_model = (
    tuned_random_forest_models[
        "Random Forest - Depth 14"
    ]
)

actual_tree_count = len(
    depth_14_random_forest_model.trees
)

print(
    f"Requested trees: 75\n"
    f"Trees actually trained: {actual_tree_count}"
)

# COMMAND ----------

model_comparison_df = (
    model_comparison_df
    .withColumn(
        "split",
        F.initcap("split")
    )
)

# COMMAND ----------

depth_14_comparison_df = (
    spark.createDataFrame(
        linear_regression_results
        + random_forest_results
        + tuned_random_forest_results
    )
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
    .orderBy("split", "rmse")
)

display(depth_14_comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tuned Random Forest Performance
# MAGIC
# MAGIC Increasing Random Forest complexity produced a substantial improvement. The
# MAGIC depth-14 model achieved a validation MAE of **$1.17**, an RMSE of **$1.75**,
# MAGIC and an R² of **0.9647**, outperforming Linear Regression across all evaluation
# MAGIC metrics. Training and validation performance remained closely aligned,
# MAGIC indicating that the improvement came from capturing useful nonlinear
# MAGIC relationships rather than evident overfitting. The performance gain came with
# MAGIC a larger 109 MB model, creating a tradeoff between predictive accuracy and
# MAGIC model size.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient-Boosted Trees
# MAGIC
# MAGIC Gradient-Boosted Trees may approach or exceed the tuned Random Forest while using fewer, smaller trees.

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

gradient_boosted_trees = GBTRegressor(
    featuresCol="unscaled_features",
    labelCol=target_column,
    predictionCol="prediction",
    maxIter=30,
    maxDepth=8,
    stepSize=0.05,
    minInstancesPerNode=5,
    subsamplingRate=0.8,
    maxBins=32,
    seed=42
)

gradient_boosted_trees_model = (gradient_boosted_trees.fit(train_prepared_df))

print("Gradient-Boosted Trees training completed.")
print(f"Number of trees: {len(gradient_boosted_trees_model.trees)}")

# COMMAND ----------

# Generate predictions
gbt_train_predictions_df = (gradient_boosted_trees_model.transform(train_prepared_df))
gbt_validation_predictions_df = (gradient_boosted_trees_model.transform(validation_prepared_df))

# COMMAND ----------

gradient_boosted_trees_results =[
    evaluate_regression_predictions(
        gbt_train_predictions_df,
        "Gradient-Boosted Trees",
        "Train"
    ),
    evaluate_regression_predictions(
        gbt_validation_predictions_df,
        "Gradient-Boosted Trees",
        "Validation"
    )
]

final_validation_comparison_df = (
    spark.createDataFrame(
        all_baseline_results
        + linear_regression_results
        + random_forest_results
        + tuned_random_forest_results
        + gradient_boosted_trees_results
    )
    .select(
        "model",
        F.initcap("split").alias("split"),
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
    .orderBy("split", "rmse")
)

display(final_validation_comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Model Selection
# MAGIC
# MAGIC The tuned depth-14 Random Forest produced the best validation metrics, but its
# MAGIC advantage over Gradient-Boosted Trees was minimal: less than one cent in MAE,
# MAGIC approximately three cents in RMSE, and 0.0012 in R². The Random Forest also
# MAGIC exceeded 109 MB. Gradient-Boosted Trees was therefore selected as the final
# MAGIC model because it provides nearly equivalent predictive performance with a
# MAGIC smaller and more practical ensemble.

# COMMAND ----------

# Final training and test evaluation
development_df = (
    train_df
    .unionByName(validation_df)
)

development_input_df = convert_boolean_features(development_df)

final_test_input_df = convert_boolean_features(test_df)

print(f"Development rows: {development_df.count():,}")
print(f"Final test rows: {test_df.count():,}")

# COMMAND ----------

# Refit preprocessing without using test data
final_processing_model = (
    preprocessing_pipeline.fit(development_input_df)
)

development_prepared_df = (final_processing_model.transform(development_input_df))
final_test_prepared_df = (final_processing_model.transform(final_test_input_df))

print("Final preprocessing completed.")

# COMMAND ----------

# Retrain the selected GBT configuration
final_gbt_estimator = GBTRegressor(
    featuresCol="unscaled_features",
    labelCol=target_column,
    predictionCol="prediction",
    maxIter=30,
    maxDepth=8,
    stepSize=0.05,
    minInstancesPerNode=5,
    subsamplingRate=0.8,
    maxBins=32,
    seed=42
)

final_gbt_model = final_gbt_estimator.fit(development_prepared_df)

print("Final GBT model trained.")

# COMMAND ----------

# Evaluate the untouched test period
final_test_predictions_df = (final_gbt_model.transform(final_test_prepared_df))

final_test_result = evaluate_regression_predictions(
    final_test_predictions_df,
    "Final Gradient-Boosted Trees",
    "Test"
)

final_test_results_df = (
    spark.createDataFrame([final_test_result])
    .select(
        "model",
        "split",
        F.round("mae", 4).alias("mae"),
        F.round("rmse", 4).alias("rmse"),
        F.round("r2", 4).alias("r2")
    )
)

display(final_test_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Test Performance
# MAGIC
# MAGIC The selected Gradient-Boosted Trees model achieved a test MAE of **$1.19**, a
# MAGIC test RMSE of **$1.81**, and a test R² of **0.9625**. Its test performance was
# MAGIC close to its validation performance, indicating that the model generalized
# MAGIC well to the untouched final time period. On average, predicted quoted prices
# MAGIC differed from actual quoted prices by approximately $1.19.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Residual Analysis

# COMMAND ----------

# First create the error columns
final_test_diagnostics_df = (
    final_test_predictions_df
    .withColumn(
        "residual",
        F.col(target_column) - F.col("prediction")
    )
    .withColumn(
        "absolute_error",
        F.abs(F.col("residual"))
    )
    .withColumn(
        "squared_error",
        F.pow(F.col("residual"), 2)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, a positive residual means the model underpredicted the quoted price; a negative residual means it overpredicted.

# COMMAND ----------

# Overall error distribution
overall_error_summary_df = (
    final_test_diagnostics_df
    .agg(
        F.count("*").alias("test_quotes"),
        F.round(F.avg("residual"), 4).alias("average_residual"),
        F.round(F.min("prediction"), 2).alias("minimum_prediction"),
        F.round(F.max("prediction"), 2).alias("maximum_prediction"),
        F.sum((F.col("prediction") < 0).cast("int")).alias("negative_predictions"),
        F.round(F.expr("percentile_approx(absolute_error, 0.50)"), 4).alias("median_absolute_error"),
        F.round(F.expr("percentile_approx(absolute_error, 0.90)"), 4).alias("p90_absolute_error"),
        F.round(F.expr("percentile_approx(absolute_error, 0.95)"), 4).alias("p95_absolute_error"),
        F.round(F.max("absolute_error"), 4).alias("maximum_absolute_error")
    )
)

display(overall_error_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overall Error Distribution
# MAGIC
# MAGIC The final model showed almost no systematic bias, with an average residual of
# MAGIC **$0.01** and no negative price predictions. Half of the test predictions were
# MAGIC within **$0.81** of the actual quoted price, while 90% were within **$2.58**
# MAGIC and 95% were within **$3.41**. A maximum absolute error of **$49.65**
# MAGIC indicates that a small number of unusual or high-priced quotes require
# MAGIC additional inspection.

# COMMAND ----------

# Error by provider and product
product_error_summary_df = (
    final_test_diagnostics_df
    .groupBy(
        "cab_type",
        "name"
    )
    .agg(
        F.count("*").alias("test_quotes"),
        F.round(F.avg(target_column), 2).alias("average_actual_price"),
        F.round(F.avg("prediction"), 2).alias("average_predicted_price"),
        F.round(F.avg("absolute_error"), 4).alias("mae"),
        F.round(F.sqrt(F.avg("squared_error")), 4).alias("rmse"),
        F.round(F.avg("residual"), 4).alias("average_residual")
    )
    .orderBy(F.desc("mae"))
)

display(product_error_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product-Level Performance
# MAGIC
# MAGIC The final model remained well calibrated across all Uber and Lyft products,
# MAGIC with average predicted prices within approximately **$0.12** of average actual
# MAGIC prices. Product-level MAE ranged from **$0.70** for Lyft to **$1.75** for
# MAGIC UberXL. UberXL had the largest prediction variability, with an RMSE of
# MAGIC **$2.87**, while no product showed substantial systematic overprediction or
# MAGIC underprediction. The results indicate that strong overall performance was not
# MAGIC driven by only one provider or service tier.

# COMMAND ----------

# Inspect the largest individual errors
largest_test_errors_df = (
    final_test_diagnostics_df
    .select(
        "id",
        "query_date_local",
        "cab_type",
        "name",
        "source",
        "destination",
        "distance",
        "surge_multiplier",
        target_column,
        F.round("prediction", 2).alias("prediction"),
        F.round("residual", 2).alias("residual"),
        F.round("absolute_error", 2).alias("absolute_error")
    )
    .orderBy(
        F.desc("absolute_error")
    )
    .limit(20)
)

display(largest_test_errors_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Largest Prediction Errors
# MAGIC
# MAGIC The largest test errors were concentrated entirely among Uber quotes with a
# MAGIC recorded surge multiplier of `1.0`. In each case, the actual quoted price was
# MAGIC substantially higher than the model prediction. The largest error occurred for
# MAGIC an UberXL quote priced at **$68.50**, compared with a predicted price of
# MAGIC **$18.85**. Because the source data contains no Uber surge multiplier above
# MAGIC `1.0`, the model lacks a recorded signal that explains these unusual price
# MAGIC spikes. These observations were retained because there is no evidence that
# MAGIC they are invalid, but they represent an important limitation of the available
# MAGIC features.

# COMMAND ----------

# Create a model-artifact volume
spark.sql("""
    CREATE VOLUME IF NOT EXISTS
    rideshare_elt.gold.ml_artifacts
""")

artifact_root = (
    "/Volumes/rideshare_elt/gold/"
    "ml_artifacts/price_prediction"
)

# COMMAND ----------

# Save the fitted models
preprocessing_model_path = (
    f"{artifact_root}/preprocessing_model"
)

gbt_model_path = (
    f"{artifact_root}/final_gbt_model"
)

final_processing_model.write().overwrite().save(
    preprocessing_model_path
)

final_gbt_model.write().overwrite().save(
    gbt_model_path
)

print("Models saved successfully.")
print(f"Preprocessing model: {preprocessing_model_path}")
print(f"GBT model: {gbt_model_path}")

# COMMAND ----------

# Save the evaluation results
final_test_results_path = (
    f"{artifact_root}/results/final_test_metrics"
)

product_error_results_path = (
    f"{artifact_root}/results/product_error_metrics"
)

overall_error_results_path = (
    f"{artifact_root}/results/overall_error_metrics"
)

final_test_results_df.write.format(
    "delta"
).mode(
    "overwrite"
).save(
    final_test_results_path
)

product_error_summary_df.write.format(
    "delta"
).mode(
    "overwrite"
).save(
    product_error_results_path
)

overall_error_summary_df.write.format(
    "delta"
).mode(
    "overwrite"
).save(
    overall_error_results_path
)

print("Evaluation results saved successfully.")

# COMMAND ----------

from pyspark.ml import PipelineModel
from pyspark.ml.regression import GBTRegressionModel

artifact_root = (
    "/Volumes/rideshare_elt/gold/"
    "ml_artifacts/price_prediction"
)

final_preprocessing_model = PipelineModel.load(
    f"{artifact_root}/preprocessing_model"
)

final_gbt_model = GBTRegressionModel.load(
    f"{artifact_root}/final_gbt_model"
)

final_test_results_df = (
    spark.read.format("delta")
    .load(
        f"{artifact_root}/results/final_test_metrics"
    )
)

product_error_summary_df = (
    spark.read.format("delta")
    .load(
        f"{artifact_root}/results/product_error_metrics"
    )
)

overall_error_summary_df = (
    spark.read.format("delta")
    .load(
        f"{artifact_root}/results/overall_error_metrics"
    )
)

print("Saved models and results loaded successfully.")