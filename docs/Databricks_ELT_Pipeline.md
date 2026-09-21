# Databricks ELT Pipeline

## Overview

This pipeline transforms public Uber and Lyft fare estimates and Boston weather observations into a governed, analytics-ready dataset.

The pipeline follows a Bronze, Silver, and Gold medallion architecture in Databricks:

```text
Source CSV files
    |
    v
Unity Catalog Volumes
    |
    v
Bronze Delta tables
    |
    v
Silver clean and quarantine tables
    |
    v
Gold fare and weather dataset
    |
    v
Tableau-ready CSV export
```

The final dataset contains one row per valid fare quote and supports both the
Tableau pricing dashboard and the completed predictive fare-modeling workflow.

The source records represent queried fare estimates, not completed rides, bookings, revenue, or rider demand.

## Technology Stack

- Databricks
- Apache Spark
- PySpark
- Spark SQL
- Delta Lake
- Unity Catalog
- Databricks Auto Loader
- Structured Streaming checkpoints
- Tableau Desktop

## Pipeline Notebooks

The notebooks should be executed in numeric order.

| Notebook | Layer | Purpose |
|---|---|---|
| `00_Environment_Validation.py` | Source validation | Inspect source files, schemas, nulls, identifiers, and numeric quality rules |
| `01_Bronze_Ingestion.py` | Bronze | Incrementally ingest the raw CSV files into governed Delta tables |
| `02_Silver_Transformations.py` | Silver | Clean, standardize, validate, enrich, and quarantine records |
| `03_Gold_Analytics.py` | Gold | Match fare quotes with source and destination weather observations |
| `04_Tableau_Export.py` | Delivery | Validate and export the Gold dataset for Tableau |

## Unity Catalog Design

```text
rideshare_elt
|-- bronze
|   |-- cab_rides_raw
|   `-- weather_raw
|-- silver
|   |-- cab_rides_clean
|   |-- cab_rides_quarantine
|   `-- weather_clean
`-- gold
    `-- rides_weather_enriched
```

Unity Catalog volumes store the source files, Auto Loader checkpoints, and Tableau export.

The source directories are:

```text
/Volumes/rideshare_elt/bronze/cab_rides_files
/Volumes/rideshare_elt/bronze/weather_files
```

The Tableau export is written to:

```text
/Volumes/rideshare_elt/gold/tableau_exports/rideshare_gold.csv
```

## Source Validation

The environment-validation notebook confirms that the expected files are available before ingestion. It previews the datasets, infers their source schemas for inspection, and calculates initial quality metrics.

### Cab-Ride Checks

- Total record count
- Distinct ride ID count
- Null count for every column
- Non-positive distance count
- Negative price count
- Surge multiplier values below `1`

### Weather Checks

- Total record count
- Distinct location count
- Distinct timestamp count
- Null count for every column
- Humidity values outside the range from `0` to `1`
- Non-positive pressure values
- Negative wind values

These checks establish the source-data baseline before the governed tables are created.

## Bronze Layer

The Bronze layer preserves the original source values and adds ingestion metadata. Business transformations are intentionally deferred to the Silver layer.

### Inputs

- `cab_rides.csv`
- `weather.csv`

### Outputs

- `rideshare_elt.bronze.cab_rides_raw`
- `rideshare_elt.bronze.weather_raw`

### Ingestion Design

Both CSV sources are ingested using Databricks Auto Loader with:

- Explicit source schemas
- CSV header recognition
- Rescue-mode schema evolution
- A `_rescued_data` column for unexpected fields or incompatible values
- Separate checkpoints for the fare and weather streams
- An `availableNow` trigger for bounded incremental processing
- Append writes to Delta tables

The checkpoint directories allow Auto Loader to remember previously processed files and avoid ingesting the same file again during subsequent runs.

### Ingestion Metadata

The Bronze tables add:

- `_ingested_at`
- `_source_file`
- `_source_file_name`
- `_source_file_modified_at`
- `_rescued_data`

This metadata supports source tracing, operational troubleshooting, and ingestion auditing.

### Bronze Validation

| Dataset | Rows | Important nullable field | Null rows | Rescued rows |
|---|---:|---|---:|---:|
| Cab rides | 693,071 | `price` | 55,095 | 0 |
| Weather | 6,276 | `rain` | 5,382 | 0 |

Zero rescued rows indicate that the supplied source columns matched the declared ingestion schemas.

Missing prices and rain measurements remain unchanged in Bronze so that the raw source values are preserved.

## Silver Layer

The Silver layer converts the raw records into clean, standardized, and analytics-ready tables.

### Outputs

| Table | Rows | Purpose |
|---|---:|---|
| `rideshare_elt.silver.cab_rides_clean` | 637,976 | Valid fare quotes available for analysis |
| `rideshare_elt.silver.cab_rides_quarantine` | 55,095 | Fare records rejected by the pricing-quality rules |
| `rideshare_elt.silver.weather_clean` | 6,276 | Standardized weather observations |

The clean and quarantined fare records reconcile to all 693,071 Bronze fare records.

### Fare Transformations

The fare pipeline:

- Trims provider, product, source, and destination text.
- Interprets the 13-digit fare timestamp as milliseconds.
- Creates a UTC query timestamp.
- Converts the UTC timestamp to Boston local time using `America/New_York`.
- Derives local date, hour, weekday name, and weekend fields.
- Creates a Boolean surge indicator when `surge_multiplier > 1`.
- Calculates price per mile when price is available and distance is positive.
- Adds a Silver transformation timestamp.

### Fare Quality Rules

Each fare record receives a quality status.

| Condition | Quality status |
|---|---|
| Missing or blank ride ID | `QUARANTINE_MISSING_ID` |
| Missing price | `QUARANTINE_MISSING_PRICE` |
| Negative price | `QUARANTINE_INVALID_PRICE` |
| Zero or negative distance | `QUARANTINE_INVALID_DISTANCE` |
| All checks passed | `VALID` |

Invalid records are retained in a quarantine Delta table instead of being deleted. This preserves auditability while keeping the analytical table reliable.

The current 55,095 quarantined records correspond to fare quotes with missing prices.

### Weather Transformations

The weather pipeline:

- Trims location text.
- Interprets the 10-digit weather timestamp as seconds.
- Creates UTC and Boston-local timestamps.
- Derives local date, hour, weekday name, and hourly time fields.
- Preserves whether the original rain measurement was missing.
- Creates an analytics-ready rain amount.
- Creates a Boolean rain indicator.
- Adds a Silver transformation timestamp.

An unreported rain measurement is assigned an analytical `rain_amount` of `0.0`, while `rain_was_missing` preserves the original missing-data condition. This prevents the transformation from erasing the distinction between a reported zero and a missing source value.

### Weather Quality Rules

Weather records are validated for:

- Missing locations
- Invalid timestamps
- Missing temperatures
- Humidity outside the range from `0` to `1`
- Cloud cover outside the range from `0` to `1`
- Duplicate location and timestamp keys
- Incorrect rain-value replacement

All 6,276 weather records passed the current Silver quality rules.

## Gold Layer

The Gold layer combines each valid fare quote with the nearest available weather observation at both ends of the route.

### Input Tables

- `rideshare_elt.silver.cab_rides_clean`
- `rideshare_elt.silver.weather_clean`

### Output Table

- `rideshare_elt.gold.rides_weather_enriched`

### Expected Grain

The Gold table maintains exactly:

```text
One row per valid fare quote
```

The expected and validated row count is 637,976.

### Weather-Matching Method

For each fare quote, the pipeline:

1. Finds weather observations for the same source location.
2. Limits candidates to observations within 60 minutes before or after the fare-query timestamp.
3. Calculates the absolute time difference between each observation and the query.
4. Ranks candidate observations by the smallest time difference.
5. Uses the most recent timestamp as the tie-breaker.
6. Retains the nearest source-weather observation.
7. Repeats the same process for the destination location.

Both matches use the fare-query timestamp. The destination match does not use a separately estimated arrival time.

### Gold Enrichments

The Gold dataset includes:

- Provider and ride-product attributes
- Source and destination
- Directional route
- Distance
- Quoted price
- Surge multiplier and surge indicator
- Price per mile
- UTC and Boston-local query fields
- Source weather measurements
- Destination weather measurements
- Average endpoint temperature
- Endpoint temperature difference
- Rain-at-either-location indicator
- Gold refresh timestamp

### Gold Validation

| Metric | Result |
|---|---:|
| Rows | 637,976 |
| Unique fare-quote IDs | 637,976 |
| Curated columns | 36 |
| Source-weather match rate | 100% |
| Destination-weather match rate | 100% |
| Weather-tolerance violations | 0 |

The validation confirms that weather matching did not change the fare-quote row count, duplicate IDs, or retain a match outside the 60-minute tolerance.

## Tableau Export

The Tableau export notebook removes the pipeline-only `_gold_created_at` field from the curated Gold dataset.

### Export Results

| Metric | Result |
|---|---:|
| Exported rows | 637,976 |
| Unique fare-quote IDs | 637,976 |
| Exported columns | 35 |
| Export file | `rideshare_gold.csv` |
| Export size | 199.52 MB |

Spark first writes the dataset into a temporary directory. The notebook confirms that Spark created exactly one CSV part file, moves and renames that file, and then removes the temporary directory.

The exported CSV is read back and validated for:

- Row count
- Unique fare-quote IDs
- Column count and order
- Missing prices
- Missing source weather
- Missing destination weather

Reading the exported file back provides a final validation of the delivered dataset rather than relying only on the in-memory DataFrame.

## Data Lineage

| Source | Transformation | Output |
|---|---|---|
| `cab_rides.csv` | Auto Loader ingestion | `rideshare_elt.bronze.cab_rides_raw` |
| `weather.csv` | Auto Loader ingestion | `rideshare_elt.bronze.weather_raw` |
| Bronze fare table | Standardization, validation, and quarantine | Clean and quarantined Silver fare tables |
| Bronze weather table | Standardization and validation | Silver weather table |
| Silver fare and weather tables | Nearest-weather matching and enrichment | Gold analytical table |
| Gold table | Metadata removal and validation | Tableau-ready CSV |
| Tableau-ready CSV | Interactive visual analysis | Packaged Tableau workbook |

## Repeatability and Validation Controls

- Explicit source schemas reduce unintended type inference.
- Auto Loader checkpoints track files that were already processed.
- Rescue-mode ingestion captures unexpected fields or incompatible values.
- Bronze ingestion metadata preserves source lineage.
- Silver clean and quarantine records reconcile to the Bronze population.
- Duplicate fare IDs and weather business keys are checked.
- Gold input dimensions are validated before weather matching.
- Weather matches are limited to the configured 60-minute tolerance.
- Gold output retains one record for every valid fare quote.
- The Tableau export is read back and independently validated.
- The notebooks stop execution when required tables, volumes, dimensions, or quality checks are incorrect.

## Assumptions and Limitations

- The source records are queried fare estimates rather than completed rides.
- Records with missing prices cannot support pricing analysis and are retained in quarantine.
- An unreported rain value becomes an analytical rain amount of zero, while a separate field preserves that the source value was missing.
- Weather observations may occur before or after the query but must fall within the 60-minute tolerance.
- Source and destination matches both use the fare-query timestamp; the pipeline does not estimate arrival time.
- The current input is a historical source snapshot rather than a live operational feed.
- Auto Loader makes Bronze ingestion incremental, but the current Silver and Gold notebooks rebuild their target tables.
- Large source and Tableau-export CSV files are excluded from Git.

## Pipeline Outcome

The completed pipeline provides:

- Governed Bronze, Silver, and Gold Delta tables
- Explicit schemas and ingestion lineage
- Standardized fare, location, and time attributes
- Auditable quality rules and fare-record quarantine
- Source and destination weather enrichment
- Reconciled record counts across layers
- A validated Tableau export
- A validated modeling source for the completed GBT fare-prediction workflow
