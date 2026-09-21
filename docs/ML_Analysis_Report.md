# Ride Fare Prediction: Machine Learning Analysis

## Executive Summary

This analysis develops a supervised regression model that predicts an Uber or
Lyft quoted fare from the ride product, trip characteristics, query time,
recorded surge multiplier, locations, and matched endpoint weather. The model
uses the curated Gold table produced by the Databricks ELT pipeline and
preserves its grain of one row per valid fare quote.

The selected Gradient-Boosted Trees (GBT) model achieved the following results
on the untouched test period:

| Metric | Test result |
| --- | ---: |
| Mean absolute error (MAE) | **$1.1948** |
| Root mean squared error (RMSE) | **$1.8133** |
| R-squared (R²) | **0.9625** |

The model explains approximately 96% of quoted-price variation in the final
time period. Half of its predictions were within $0.81 of the actual quote,
90% were within $2.58, and 95% were within $3.41.

## Objective and Scope

- **Task:** Supervised regression
- **Target:** `price`
- **Source:** `rideshare_elt.gold.rides_weather_enriched`
- **Population:** 637,976 valid Uber and Lyft fare quotes
- **Period:** November 25 through December 18, 2018
- **Unit of analysis:** One fare quote

The target is the quoted price, not a completed-ride charge or realized
revenue. The model is therefore designed to reproduce patterns in historical
fare quotes rather than forecast ride demand or bookings.

## Source and Target Validation

Before modeling, the notebook confirms that:

- The Gold table contains 637,976 rows and 637,976 unique quote IDs.
- The target contains no null or negative values.
- Quoted prices range from $2.50 to $97.50.
- Removing modeling-only exclusions does not change the dataset grain.

`_gold_created_at` is removed because it is pipeline metadata.
`price_per_mile` is also removed because it is calculated from the target:

```text
price_per_mile = price / distance
```

Including this field would leak the answer into the model.

## Exploratory Analysis and Feature Selection

The exploratory stage reviews the target distribution, quoted prices by
provider and product, numeric correlations with price, categorical
cardinalities, and missing values.

The correlation matrix revealed substantial duplication between source and
destination weather measurements. Because the two endpoints are geographically
close, corresponding temperature, pressure, humidity, wind, rain, and cloud
measurements are strongly correlated. The final feature set retains one
representative source measurement or an engineered endpoint feature instead of
both copies.

### Final Numeric Features

- `distance`
- `surge_multiplier`
- `query_hour_local`
- `source_clouds`
- `source_pressure`
- `source_rain_amount`
- `source_humidity`
- `source_wind`
- `average_endpoint_temperature`
- `endpoint_temperature_difference`

### Final Categorical Features

- `name`
- `source`
- `destination`
- `query_day_name_local`
- `is_weekend`
- `rain_at_either_location`

`source` and `destination` are retained separately. The derived `route` field
is excluded because it duplicates those two inputs. `cab_type` is excluded
because the provider-specific product in `name` already identifies the service
and provider combination.

## Leakage-Safe Chronological Split

Records are split by `query_date_local` rather than randomly. This design
trains the model on earlier quotes and evaluates it on later quotes, preventing
future observations from influencing earlier predictions.

| Split | Rows | Date range | Share |
| --- | ---: | --- | ---: |
| Train | 406,638 | Nov 25–Dec 12, 2018 | 63.74% |
| Validation | 123,933 | Dec 13–Dec 15, 2018 | 19.43% |
| Test | 107,405 | Dec 16–Dec 18, 2018 | 16.84% |

The test period remains untouched throughout model development and is used
only after the final model configuration is selected.

## Preprocessing

The preprocessing pipeline performs the following operations:

1. Casts `is_weekend` and `rain_at_either_location` to numeric indicators.
2. Applies `StringIndexer` to product, source, destination, and local weekday.
3. One-hot encodes the indexed categorical features.
4. Assembles numeric and encoded inputs into `unscaled_features`.
5. Standardizes the feature vector only for Linear Regression.

Preprocessing is initially fitted only on the training split. After model
selection, it is refitted on the combined train and validation development
period before the final test evaluation.

## Model Development and Validation

Two constant-price baselines establish minimum performance expectations. The
analysis then compares Linear Regression, Random Forest, a more complex tuned
Random Forest, and Gradient-Boosted Trees.

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Training median baseline | $7.3932 | $9.8007 | -0.1091 |
| Training mean baseline | $7.5557 | $9.3061 | 0.0000 |
| Linear Regression | $1.7435 | $2.4697 | 0.9296 |
| Initial Random Forest | $2.5939 | $3.3745 | 0.8685 |
| Tuned Random Forest, depth 14 | **$1.1732** | **$1.7478** | **0.9647** |
| Gradient-Boosted Trees | $1.1821 | $1.7780 | 0.9635 |

The initial Random Forest underfit the data. Increasing its depth and number of
trees substantially improved performance without creating a material gap
between training and validation metrics.

## Final Model Selection

The tuned depth-14 Random Forest achieved the best validation metrics, but its
advantage over GBT was very small: less than one cent in MAE, approximately
three cents in RMSE, and 0.0012 in R². The tuned Random Forest also exceeded
109 MB during training.

GBT was selected as the final model because it delivered nearly identical
validation accuracy with a smaller and more practical ensemble. Its principal
configuration is:

| Parameter | Value |
| --- | ---: |
| Trees / iterations | 30 |
| Maximum depth | 8 |
| Step size | 0.05 |
| Minimum instances per node | 5 |
| Subsampling rate | 0.8 |
| Maximum bins | 32 |
| Random seed | 42 |

After selection, preprocessing and GBT were refitted on the combined training
and validation data. The final model was then evaluated once on the untouched
test period.

## Final Test Evaluation

| Model | Split | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Final Gradient-Boosted Trees | Test | **$1.1948** | **$1.8133** | **0.9625** |

The test metrics remain close to the validation metrics, indicating that the
selected model generalized well to the final chronological period.

## Residual Diagnostics

Residuals are defined as:

```text
residual = actual price - predicted price
```

A positive residual indicates underprediction, while a negative residual
indicates overprediction.

| Diagnostic | Result |
| --- | ---: |
| Test quotes | 107,405 |
| Average residual | $0.0129 |
| Median absolute error | $0.8144 |
| 90th-percentile absolute error | $2.5849 |
| 95th-percentile absolute error | $3.4069 |
| Maximum absolute error | $49.6521 |
| Minimum prediction | $4.06 |
| Maximum prediction | $87.93 |
| Negative predictions | 0 |

The average residual is close to zero, so the model has little overall
directional bias. The maximum error shows that a small number of unusual,
high-priced quotes remain difficult to predict.

### Product-Level Performance

Average predicted prices remained within approximately $0.12 of average actual
prices for every product. Product-level MAE ranged from $0.70 for Lyft to $1.75
for UberXL. UberXL was the most difficult product, with an RMSE of $2.87, while
no product showed substantial systematic overprediction or underprediction.

### Largest Errors

The 20 largest test errors were all Uber quotes with a recorded surge
multiplier of `1.0`, and all were substantial underpredictions. The largest was
an UberXL quote with an actual price of $68.50 and a prediction of $18.85.

The source data contains no Uber surge multiplier above `1.0`. Consequently,
the model has no recorded surge signal capable of explaining these apparent
Uber price spikes. The records were retained because there is no evidence that
they are invalid, but they expose an important limitation of the available
features.

## Feature Importance

One-hot-encoded feature importances are aggregated back to their original
fields for interpretation.

| Feature | Importance |
| --- | ---: |
| Ride product (`name`) | 69.43% |
| Distance | 18.70% |
| Surge multiplier | 7.91% |
| Destination | 1.98% |
| Source | 1.54% |
| All time and weather features combined | 0.44% |

Ride product, distance, and recorded surge multiplier account for
approximately 96% of total model importance. Location makes a smaller
contribution, while time and weather variables add little predictive value in
this dataset. Feature importance describes predictive usefulness within this
model and should not be interpreted as causation.

## Saved Artifacts

The fitted preprocessing model, final GBT model, and evaluation summaries are
persisted under the Unity Catalog volume:

```text
/Volumes/rideshare_elt/gold/ml_artifacts/price_prediction
```

Saved artifacts include:

- `preprocessing_model`
- `final_gbt_model`
- `results/final_test_metrics`
- `results/product_error_metrics`
- `results/overall_error_metrics`

The models can be restored with Spark `PipelineModel` and
`GBTRegressionModel`; evaluation outputs are stored in Delta format. This
allows later analysis without retraining the model.

## Limitations

- The dataset covers only November 25 through December 18, 2018, so the model
  does not establish long-term or seasonal performance.
- Records are fare quotes rather than completed rides or realized revenue.
- The source contains no Uber surge multiplier above `1.0`, limiting the
  model's ability to explain unusual Uber price spikes.
- Feature importance is model-specific and non-causal.
- Performance is measured on a later period from the same historical dataset;
  production monitoring would still be required to detect drift.
- MLflow experiment tracking and production deployment are not yet part of the
  implemented workflow.

## Conclusion

The final GBT model substantially outperformed both constant baselines and
generalized consistently across the chronological validation and test periods.
Its $1.19 test MAE and 0.9625 R² demonstrate strong predictive performance for
the observed fare-quote population. Ride product, distance, and surge dominate
the model, while the residual analysis identifies unrecorded Uber price spikes
as the primary unresolved source of extreme error.

## Reproducibility

- [View the machine-learning notebook](../notebooks/05_ML_Price_Prediction.py)
- [Review the Gold dataset data dictionary](Data_Dictionary.md)
- [Review the Databricks ELT pipeline](Databricks_ELT_Pipeline.md)
