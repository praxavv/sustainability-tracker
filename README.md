## Running the Notebook

You can run the notebook directly on Google Colab using this link:

[Open Sustainability Tracker Notebook](https://colab.research.google.com/drive/1A08BiAyesG5OAGZ-7HDGx5KnaBXnmsvt?usp=sharing)

# Sustainability-Tracker — Interactive Weather Forecasting with BigQuery AI

**One-line:** A compact, interactive demo notebook that turns NOAA GSOD (2018–2024) data into visual insights and short-term temperature forecasts using a pre-trained XGBoost model.

---

## What it is

* Exploratory notebook that walks from raw GSOD samples → cleaned daily aggregates → yearly summaries → interactive visualizations (including a 3D Plotly globe) → model inference and validation.
* Intended as a **light, interactive demo**: core data engineering and training pipelines were done separately; this notebook showcases the workflow, visual UX and inference interface.

---

## Key features

* Quick data inspection (sample & cleaned CSVs, JSON metadata)
* Yearly aggregate visualizations: mean/min/max temperature bands, precipitation, wind, correlation heatmap
* Interactive 3D weather globe (Plotly) with station hover cards (temp, dew point, wind, precip, pressure)
* Pretrained model inference: `temp_c_boosted.pkl` loaded directly from GitHub for example predictions
* Rolling-window validation summary (5 windows): **RMSE ≈ 2.5°C**, **MAE ≈ 1.8°C**, **R² ≈ 0.91**

---

## Files referenced in the notebook

* `Data/sample_gsod_last10.csv` — sample raw rows
* `Data/cleaned-data.csv` — cleaned daily observations
* `Data/Globe.parquet` — station / aggregated metrics used for globe
* `temp_c_boosted.pkl` — pretrained XGBoost regressor (inference)
* `gsod_train_features.parquet`, `gsod_daily_aggregated.parquet` — lookup / nearest-station data
* `cross-validation-rolling-window.json`, `Test-Predictions.csv` — validation metadata & predictions

---

## Quick start (Colab or local)

1. Install essentials:

```bash
pip install pandas numpy matplotlib seaborn plotly joblib scikit-learn xgboost requests geopy ipywidgets
# In Colab: also run `!pip install ipywidgets` and enable widgets if needed
```

2. Open `Sustainability-Tracker.ipynb` in Colab / Jupyter.
3. Run cells top-to-bottom. The notebook fetches sample files and the model directly from the repo/GDrive as needed.

---

## Example: load model & run a single prediction

```python
import requests, joblib
from io import BytesIO
import pandas as pd

url = "https://github.com/praxavv/sustainability-tracker/raw/main/temp_c_boosted.pkl"
resp = requests.get(url); resp.raise_for_status()
model = joblib.load(BytesIO(resp.content))

X = pd.DataFrame([{
    'lat': 19.0785, 'lon': 72.8782, 'month': 9, 'day': 3,
    'dewp_c': 25, 'wind_speed_m_s': 2, 'visibility_km': 10,
    'precipitation_mm': 0, 'pressure_hpa': 1012
}])
print("Prediction (°C):", round(model.predict(X)[0], 2))
```

---

## 3D globe — controls & tips

The globe is built with Plotly and is interactive. Short navigation guide:

**Trackpad / laptop**

* Rotate: two-finger drag horizontally (or click+drag).
* Zoom: two-finger vertical drag (or pinch).
* Span / inspect: double-click, hold and drag to box-select.
* Reset view: click the *home* icon in the top-right of the Plotly toolbar.

**Mouse / desktop**

* Rotate: left-click + drag.
* Zoom: scroll wheel (or use the zoom tool in the toolbar).
* Hover: move cursor over a station to see the station card (temp, dewpoint, wind, pressure, etc.).
* Reset view: home icon in the Plotly toolbar.

Navigation behavior may vary slightly by browser. For best interactivity use a modern browser and a laptop/desktop with a decent GPU for smoother rendering.

---

## Model & validation (summary)

* Model: XGBoost regressor trained on feature-engineered GSOD daily aggregates.
* Validation: rolling time-window cross validation (5 windows).
* Typical performance (demo): **RMSE ≈ 2.5°C**, **MAE ≈ 1.8°C**, **R² ≈ 0.91** — meaning the predictor captures most variance and usually misses by a couple degrees Celsius.

---

## Notes & limitations

* This notebook is a **demo**. Productionization (low latency inference, full station coverage, uncertainty quantification, retraining pipelines) is not included here.
* Predictions rely on nearest training row to fill missing features for arbitrary coordinates/dates — useful for demos, but not a substitute for a full feature service or ensemble forecasts.
* Visualizations use sampled / pre-aggregated files to keep the notebook lightweight.

---

## Where to go next

* Swap the demo model for a served endpoint (TF/XGBoost server or cloud prediction API) for lower latency.
* Add uncertainty estimates (quantile regression or ensembles).
* Expand station coverage and ingest live GSOD updates via BigQuery streaming or scheduled ETL.

---

## License & attribution

See the repository for licensing. Data sourced from NOAA GSOD (2018–2024) via BigQuery.

---
