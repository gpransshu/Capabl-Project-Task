# NYC Taxi Trip Duration Prediction

A machine learning project that predicts the duration (in seconds) of New York City taxi trips from pickup/dropoff location, time, and passenger information. Built for the Capabl Data Science Internship (SIP).

## Overview

**Dataset.** The project uses the classic New York City Taxi Trip Duration dataset, loaded from two local CSV files:

- `train.csv` — trip records with the target column `trip_duration` (duration in seconds)
- `test.csv` — trip records without the duration, used for final predictions

Each record contains: `id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, pickup/dropoff `latitude` and `longitude`, and `store_and_fwd_flag`.

**Approach.** The notebook walks through the full pipeline:

1. **Exploratory analysis** — summary statistics, checks for missing values, duplicates, and zero-passenger rows (dropped as inaccurate).
2. **Feature engineering** — geodesic pickup-to-dropoff distance (via `geopy`), plus `month`, day-of-week name, and hour-of-day extracted from `pickup_datetime`; zero-distance rows are dropped.
3. **Visualization** — distributions of vendor, `store_and_fwd_flag`, weekday, hour, and month with matplotlib/seaborn.
4. **Encoding** — `LabelEncoder` for month names and one-hot encoding for `store_and_fwd_flag`.
5. **Modeling** — an 80/20 train/test split (`random_state=2529`) and three regression models:
   - Linear Regression (`sklearn.linear_model.LinearRegression`)
   - Lasso Regression (`alpha=0.001`)
   - Ridge Regression (`alpha=0.001`)

**Results (as recorded in the notebook).** The saved outputs report an R² score of ~0.029 for all three models and a mean absolute percentage error of ~95% for Linear Regression. Predictions on `test.csv` are generated at the end of the notebook.

## Tech Stack

- Python 3 (Jupyter notebook, ipykernel)
- pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (preprocessing, train/test split, linear models, metrics)
- geopy (geodesic distance between coordinates)

No dependency manifest (requirements.txt) is included; the libraries above are all that the notebook imports.

## Project Structure

```
.
├── README.md                   # This file
└── training and testing.ipynb  # End-to-end pipeline: EDA, feature engineering, model training, and evaluation
```

At runtime the notebook also expects `train.csv` and `test.csv` in the working directory (not committed to the repo).

## How to Run

1. Install the Python libraries listed in the Tech Stack section.
2. Obtain the New York City Taxi Trip Duration dataset and place `train.csv` and `test.csv` in the repository root.
3. Open the notebook in Jupyter:

   ```bash
   jupyter notebook "training and testing.ipynb"
   ```

4. Run the cells from top to bottom. Evaluation outputs (R², MAPE) and test-set predictions are produced by the final cells.
