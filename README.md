## Running the Notebook

You can run the notebook directly on Google Colab using this link:

[Open Sustainability Tracker Notebook](https://colab.research.google.com/drive/1A08BiAyesG5OAGZ-7HDGx5KnaBXnmsvt?usp=sharing)


Sustainability-Tracker.ipynb
Sustainability-Tracker.ipynb_
Project Title: Sustainability-Tracker

Interactive Weather Forecasting with BigQuery AI

Problem Statement: Accurate short-term temperature prediction is essential for agriculture, energy planning, and disaster preparedness. Traditional weather dashboards either overwhelm users with raw data or hide the logic behind black-box forecasts. This project tackles the challenge of making weather prediction both accurate and transparent by combining historical GSOD data, machine learning models, and interactive visualizations into a single, user-friendly platform.

Note: This is a sample notebook built for demonstration. The heavy lifting (data engineering, model training, and full-scale pipelines) was done in the original notebook—here we focus on a lighter, interactive version to showcase the workflow.


[ ]
# --- Loading Raw Data ---
import pandas as pd
import json
import requests

# Fetching sample data
csv_url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/sample_gsod_last10.csv"
json_url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/gsod_metadata.json"

# Load CSV
df = pd.read_csv(csv_url)

# Load JSON
response = requests.get(json_url)
meta = response.json()

# Display metadata
print("\nDataset Metadata:")
print(f"Total Rows: {meta['total_rows']}")
print(f"Shape: {tuple(meta['shape'])}")
print(f"Column Count: {meta['column_count']}")
print("Columns:", meta['columns'])

# Display last 10 rows
print("\nLast 10 rows:")
display(df.T)
Data Exploration: We leveraged NOAA’s GSOD dataset (2018–2024) hosted on BigQuery as the foundation for our analysis. The exploration phase began by examining dataset scale—counting the total rows and inspecting the last ten records—to confirm consistency and completeness. We then extracted metadata, including overall shape, column count, and variable names, to build a clear understanding of the dataset’s structure before moving into preprocessing and modeling.


[ ]
# --- Cleaned Data ---
import pandas as pd
import json
import requests

# Cleaned Data
csv_url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/cleaned-data.csv"
json_url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/cleaned-metadata.json"

# Load CSV
df = pd.read_csv(csv_url)

# Load JSON
response = requests.get(json_url)
meta = response.json()

# Display metadata
print("Cleaned Data:\n")
print(f"Total Rows: {meta['total_rows']}")
print(f"Shape: {tuple(meta['shape'])}")
print(f"Column Count: {meta['column_count']}")
print("Columns:", meta['columns'])

# Display last 10 rows
print("\nLast 10 rows:")
display(df.T)
Step 3: Preprocessed Daily Dataset The gsod_daily_aggregated table holds cleaned and standardized daily weather observations, ready for analysis and modeling.

Preprocessing highlights:

a. Converted units for consistency (°F → °C, mph → m/s)

b. Removed invalid or missing values

c. Enriched records by joining station metadata (name, country, latitude, longitude)

To validate these transformations, we inspected the last 10 rows and confirmed the overall dataset shape, ensuring the preprocessing pipeline produced a reliable foundation for modeling.

Note on Precision:

Predictions are reported with high decimal precision to preserve accuracy. While this may produce outputs with many decimal places, the detail ensures results remain statistically robust.


[ ]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for visualization
url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/yearly_metadata.json"
df_yearly = pd.read_json(url)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (15,10), 'font.size': 12})

fig, axs = plt.subplots(2,2)

# 1. Temperature trends with min-max band
axs[0,0].plot(df_yearly.year, df_yearly.avg_temp, color='tomato', label='Mean Temp')
axs[0,0].fill_between(df_yearly.year, df_yearly.min_temp, df_yearly.max_temp,
                      color='tomato', alpha=0.2, label='Min-Max')
axs[0,0].set_title("Annual Temperature Trends (°C)")
axs[0,0].set_xlabel("Year"); axs[0,0].set_ylabel("Temperature (°C)")
axs[0,0].legend()

# 2. Precipitation trends
axs[0,1].bar(df_yearly.year, df_yearly.total_precip, color='royalblue', alpha=0.7)
axs[0,1].plot(df_yearly.year, df_yearly.avg_precip, color='navy', marker='o', label='Avg Precip')
axs[0,1].set_title("Annual Precipitation (mm)")
axs[0,1].set_xlabel("Year"); axs[0,1].set_ylabel("Precipitation (mm)")
axs[0,1].legend()

# 3. Wind trends
axs[1,0].plot(df_yearly.year, df_yearly.avg_wind, color='seagreen', marker='s')
axs[1,0].set_title("Average Annual Wind Speed (m/s)")
axs[1,0].set_xlabel("Year"); axs[1,0].set_ylabel("Wind Speed (m/s)")

# 4. Correlation Heatmap
corr = df_yearly[['avg_temp','avg_precip','avg_wind']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1,1], square=True)
axs[1,1].set_title("Climate Metrics Correlation")

plt.tight_layout()
plt.show()
Yearly Climate Statistics

Daily data was aggregated to compute annual statistics across temperature, precipitation, and wind.

Visualizations include:

Temperature: annual mean with min–max range

Precipitation: yearly totals and average trends

Wind: average annual speed

Correlation: relationships between key climate metrics

Together, these plots offer a high-level view of long-term climate patterns.

Note: The visualizations are based on NOAA datasets and are intended to provide insights into climate trends. Actual local conditions may differ and should be interpreted with context.


[ ]
import numpy as np, pandas as pd, plotly.graph_objects as go

# config / data
R = 1
url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/Globe.parquet"
df = pd.read_parquet(url)

# sphere coords
lat = np.radians(df['latitude'].values); lon = np.radians(df['longitude'].values)
x = R * np.cos(lat) * np.cos(lon)
y = R * np.cos(lat) * np.sin(lon)
z = R * np.sin(lat)

# hover text (vectorized via apply)
hover_text = df.apply(
    lambda r: (
        f"<b>{r.station_name} ({r.country_code})</b><br><br>"
        f"Avg Temp: {r.temp_c} °C<br>"
        f"Avg Dewp: {r.dewp_c} °C<br>"
        f"Avg Wind: {r.wind_speed_m_s} m/s<br>"
        f"Avg Visibility: {r.visibility_km} km<br>"
        f"Avg Precipitation: {r.precipitation_mm} mm<br>"
        f"Avg Pressure: {r.pressure_hpa} hPa"
    ),
    axis=1
)

# globe surface grid
phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:100j]
xs = R * np.sin(phi) * np.cos(theta)
ys = R * np.sin(phi) * np.sin(theta)
zs = R * np.cos(phi)

BLUE, WHITE, NEON_ORANGE = "#0066FF", "#FFFFFF", "#FF6A00"

fig = go.Figure(data=[
    go.Surface(x=xs, y=ys, z=zs, colorscale=[[0,"black"],[1,"black"]],
               showscale=False, hoverinfo="skip", opacity=1.0),
    go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(
            size=2, sizemode="diameter", color=df["temp_c"].values,
            colorscale=[[0.0, BLUE], [0.44, WHITE], [1.0, NEON_ORANGE]],
            cmin=-40, cmax=50,
            colorbar=dict(title="Avg Temp (°C)", thickness=15, len=0.5, x=1.05, y=0.8),
            opacity=1.0, line=dict(width=0.2, color="black")
        ),
        hoverinfo="skip", showlegend=False
    ),
    go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=10, sizemode="diameter", color="rgba(0,0,0,0)", opacity=0.0),
        hovertext=hover_text, hoverinfo="text",
        hoverlabel=dict(bgcolor="black", font_size=12, font_color="#39FF14", bordercolor="#39FF14", namelength=0),
        showlegend=False
    )
])

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor="black", aspectmode="data",
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
    ),
    paper_bgcolor="black", plot_bgcolor="black",
    margin=dict(l=0, r=0, t=50, b=0),
    title="🌐 Weather Stations Globe",
    font=dict(size=10, color="white"),
    hovermode="closest", hoverdistance=2
)

fig.show()
This code builds a fully interactive 3D weather globe: latitude/longitude are mapped onto a sphere, stations are plotted as temperature-colored points, and hovering reveals detailed climate stats like dew point, wind speed, and pressure. It’s essentially turning raw data into a visual planetary dashboard.

Navigation tips (laptops): Drag with two fingers horizontally to rotate 🌍, Drag with two fingers vertically to zoom in/out 🔎, double-click hold and drag to span and click the home icon in the top right corner to get back in the original position 🔄.


[ ]
import requests
import joblib
import pandas as pd
from io import BytesIO

# 1️⃣ Fetching the model from GitHub directly into memory
url = "https://github.com/praxavv/sustainability-tracker/raw/main/temp_c_boosted.pkl"
response = requests.get(url)
response.raise_for_status()

# 2️⃣ Loading the model from bytes without saving to disk
model = joblib.load(BytesIO(response.content))

# 3️⃣ Example prediction
example_input = pd.DataFrame([{
    'lat': 19.0785, 'lon': 72.8782, 'month': 9, 'day': 3,
    'dewp_c': 25, 'wind_speed_m_s': 2, 'visibility_km': 10,
    'precipitation_mm': 0, 'pressure_hpa': 1012
}])

pred = model.predict(example_input)
print("Example prediction:", round(pred[0], 2))
We queried historical weather data from BigQuery, selected relevant features, and trained an XGBoost regressor to predict daily temperature. The model was saved as temp_c_boosted.pkl for inference and demonstration.


[ ]
import pandas as pd
import gdown
import requests
import json

# --------------------- Config ---------------------
gdrive_file_id = "1snwv4uqwb-A-cFr54oqUwqwPPIAt-MT9"
gdrive_filename = "Test-Predictions.csv"
github_meta_url = "https://github.com/praxavv/sustainability-tracker/raw/refs/heads/main/Data/cross-validation-rolling-window.json"

# --------------------- Fetch GitHub Metadata ---------------------
response = requests.get(github_meta_url)
meta_json = response.json()

# Per-window performance
df_window_perf = pd.DataFrame(meta_json["per_window_performance"])
print("Per-window performance:")
print(df_window_perf.to_string(index=False))

# Overall validation performance
print("\n🔮 Overall validation performance:")
df_overall_perf = pd.Series(meta_json["overall_validation_performance"])
print(df_overall_perf)

# --------------------- Fetching Google Drive CSV ---------------------
gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", gdrive_filename, quiet=True)
df_drive = pd.read_csv(gdrive_filename)

# --------------------- Predictions ---------------------
print("\n==== Predictions (latest per country, max 5 rows) ====")
df_drive['date'] = pd.to_datetime(df_drive['date'])
cols = ['station_name', 'country_code', 'date', 'prediction', 'actual_temp']
df_sorted = df_drive[cols].sort_values(['country_code', 'date'], ascending=[True, False])

desired_countries = ['US', 'CN', 'IN', 'RU', 'JP']
selected = []
seen = set()

for cc in desired_countries:
    subset = df_sorted[df_sorted['country_code'] == cc]
    if subset.empty:
        continue
    latest_date = subset['date'].max()
    row = subset[subset['date'] == latest_date].head(1)
    selected.append(row)
    seen.add(cc)

if len(selected) < 5:
    remaining = df_sorted[~df_sorted['country_code'].isin(seen)]
    remaining_latest = (
        remaining.sort_values('date', ascending=False)
        .drop_duplicates('country_code')
        .head(5 - len(selected))
    )
    selected.append(remaining_latest)

# Final output as DataFrame
if selected:
    df_custom_head = pd.concat(selected, ignore_index=True)
    df_custom_head['date'] = pd.to_datetime(df_custom_head['date']).dt.strftime('%Y-%m-%d')
else:
    df_custom_head = pd.DataFrame(columns=cols)  # empty fallback

display(df_custom_head)
The model went through five different validation “windows” (think of them like time-slices of unseen data), and it handled them with impressive consistency. RMSE stays in the low-to-mid 2’s, which means the model’s temperature predictions usually miss the actual value by just a couple degrees Celsius. The MAE hovering around ~1.8 shows that most errors are even smaller on average.

The crown jewel is the R² score: all windows stay well above 0.87, peaking at 0.97 in val_1, which tells us the model explains over 90% of the variance in the data. The explained variance metric mirrors that, showing stability across slices.

🔮 Big picture: With an overall RMSE ≈ 2.5 and R² ≈ 0.91, this boosted model isn’t just guessing — it’s capturing real structure in the weather data while keeping errors impressively tight.


[ ]
import matplotlib.pyplot as plt

# Calculating absolute error
df_drive['abs_error'] = abs(df_drive['prediction'] - df_drive['actual_temp'])

plt.hist(df_drive['abs_error'], bins=100, log=True)
plt.xlabel('Absolute Error (°C)')
plt.ylabel('Count (log scale)')
plt.title('Error Distribution (Sep–Dec 2024)')
plt.show()
This chart shows how often different error magnitudes occur, with most predictions staying within a small temperature miss.


[ ]
# Note: !pip install ipywidgets for better sync

import os, urllib.request, joblib, numpy as np, pandas as pd
from geopy.geocoders import Nominatim
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from datetime import datetime

# --- config / downloads ---
MODEL_URL = "https://github.com/praxavv/sustainability-tracker/raw/main/temp_c_boosted.pkl"
MODEL_PATH = "temp_c_boosted.pkl"
TRAIN_S3 = "https://praxavv.s3.eu-north-1.amazonaws.com/gsod_train_features.parquet"
DAILY_AGG_S3 = "https://praxavv.s3.eu-north-1.amazonaws.com/gsod_daily_aggregated.parquet"

if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
model = joblib.load(MODEL_PATH)

expected = list(getattr(model, "feature_names_in_", []))
geolocator = Nominatim(user_agent="geoapi")

train_df = pd.read_parquet(TRAIN_S3)
daily_df = pd.read_parquet(DAILY_AGG_S3)

# --- prediction logic ---
def run_prediction(city, country, date):
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        month, day = dt.month, dt.day
        out = ["<h3>Weather Prediction Dashboard 🌡️</h3>"]

        loc = geolocator.geocode(f"{city}, {country}")
        if not loc:
            display(HTML(f"<p style='color:red;'>⚠️ Could not geocode {city}, {country}</p>")); return
        lat, lon = round(loc.latitude, 2), round(loc.longitude, 2)
        out.append(f"<p>✅ Coordinates for <b>{city.strip()}, {country.strip()}</b> → lat={lat}, lon={lon}</p>")

        subset = train_df[(train_df['month']==month) & (train_df['day']==day)].copy()
        if subset.empty:
            display(HTML("<p style='color:red;'>⚠️ No training row found for this date.</p>")); return
        subset['dist'] = np.hypot(subset['lat'] - lat, subset['lon'] - lon)
        nearest = subset.loc[subset['dist'].idxmin()].round(2).to_dict()
        out.append(f"<p>✅ Features filled from nearest training row at lat={nearest['lat']}, lon={nearest['lon']}</p>")

        row = {
            f: (nearest.get(f) if f in nearest else
                (lat if f == "lat" else lon if f == "lon" else
                 dt.year if f == "year" else month if f == "month" else
                 day if f == "day" else np.nan))
            for f in expected
        }

        X = pd.DataFrame([row], columns=expected)
        pred = round(float(model.predict(X)[0]), 2)
        out.append(f"<p>✅ Model Prediction = <b>{pred:.2f}°C</b></p>")

        daily_df['dist'] = np.hypot(daily_df['lat'] - lat, daily_df['lon'] - lon)
        nearest_station = daily_df.loc[daily_df['dist'].idxmin(), 'station_name']
        out.append(f"<p>✅ Nearest Station → <b>{nearest_station}</b></p>")
        out.append(f"<h2>🌡️ Predicted Temp on {date}: {pred:.2f}°C</h2>")

        output_widget.clear_output(wait=True)
        with output_widget:
            display(HTML("".join(out)))

    except Exception as e:
        clear_output(wait=True)
        display(HTML(f"<p style='color:red;'>⚠️ Error: {e}</p>"))

# --- widget ---
FIELD_WIDTH, DESC_WIDTH = "300px", "110px"
city_input = widgets.Text(description="City:", placeholder="e.g., New York",
                          layout=widgets.Layout(width=FIELD_WIDTH),
                          style={"description_width": DESC_WIDTH})
country_input = widgets.Text(description="Country Code:", placeholder="(e.g. US, IN)",
                             layout=widgets.Layout(width=FIELD_WIDTH),
                             style={"description_width": DESC_WIDTH})
date_input = widgets.DatePicker(description="Date:", layout=widgets.Layout(width=FIELD_WIDTH),
                                style={"description_width": DESC_WIDTH})
predict_button = widgets.Button(description="Predict 🌡️", button_style="success",
                                layout=widgets.Layout(width="160px"))
output_widget = widgets.Output()

def on_predict_clicked(b):
    with output_widget:
        if not date_input.value:
            display(HTML("<p style='color:red;'>⚠️ Please select a date.</p>")); return
        run_prediction(city_input.value.strip(), country_input.value.strip(),
                       date_input.value.strftime("%Y-%m-%d"))

predict_button.on_click(on_predict_clicked)

inputs = widgets.VBox([city_input, country_input, date_input,
                       widgets.HBox([widgets.Box(layout=widgets.Layout(flex="1")), predict_button],
                                    layout=widgets.Layout(width="304.5px"))])
dashboard = widgets.VBox([inputs, output_widget], layout=widgets.Layout(align_items="flex-start"))
display(dashboard)
Impact Statement

By converting cleaned NOAA GSOD data (2018–2024) into an interactive globe with integrated predictive models, Sustainability-Tracker empowers diverse stakeholders—farmers, energy planners, and city officials—to visually explore forecasts, run scenario experiments, and make more informed operational decisions. When paired with the experiment widget, the platform facilitates rapid hypothesis testing and localised decision support, lowering the barrier between climate data and action.

Conclusion

This sample notebook showcases how historical data, ML, and interactive visualization can work together to produce short-term temperature forecasts, with scalable BigQuery access, batch + interactive inference, and a reproducible workflow—while leaving room to improve coverage, uncertainty, and latency in future iterations.

Gemini
Sustainability-Tracker: Interactive Weather Forecasting with BigQuery AI
Project Title: Sustainability-Tracker
Problem Statement
Accurate short-term temperature prediction is essential for agriculture, energy planning, and disaster preparedness. Traditional weather dashboards often present raw data or hide the underlying forecast logic. This project addresses the need for accurate and transparent weather prediction by integrating historical GSOD data, machine learning models, and interactive visualizations into a user-friendly platform.

This notebook serves as a sample demonstrating the workflow. The core data engineering, model training, and full-scale pipelines were developed separately; this version focuses on interactive elements and showcasing the results.

Data Sources
The project utilizes NOAA’s GSOD dataset (2018–2024) hosted on BigQuery.

Raw Data: The initial data exploration involved examining the scale and structure of the raw GSOD dataset, confirming consistency and completeness.
Cleaned Data: The data underwent preprocessing steps including:
Unit conversion (°F to °C, mph to m/s)
Removal of invalid or missing values
Enrichment with station metadata (name, country, latitude, longitude)
Data Exploration and Visualization
The notebook includes visualizations to explore climate patterns:

Yearly Climate Statistics: Aggregated daily data to compute annual statistics for temperature, precipitation, and wind. Visualizations include:

Annual mean temperature with min-max range
Yearly precipitation totals and average trends
Average annual wind speed
Correlation heatmap between key climate metrics
Interactive 3D Weather Globe: This visualization maps weather stations onto a 3D globe. Stations are colored by average temperature, and hovering over a station reveals detailed climate statistics (dew point, wind speed, pressure, etc.).

Navigation Tips for the 3D Globe (laptops):

Rotate: Drag horizontally with two fingers.
Zoom: Drag vertically with two fingers.
Pan: Double-click, hold, and drag.
Reset View: Click the "home" icon in the top-right corner.
Machine Learning Model
An XGBoost regressor was trained on historical weather data from BigQuery to predict daily temperature. The trained model (temp_c_boosted.pkl) is used for inference.

Model Validation
The model was validated using a rolling window approach across five time-slices of unseen data. Key performance metrics include:

RMSE (Root Mean Squared Error): Generally in the low-to-mid 2s (°C), indicating typical prediction errors of a couple of degrees Celsius.
MAE (Mean Absolute Error): Around ~1.8 (°C), showing smaller average errors.
R² (Coefficient of Determination): Consistently above 0.87 (peaking at 0.97), indicating the model explains over 90% of the variance in the data.
The overall validation performance shows an RMSE of approximately 2.5°C and an R² of about 0.91.

Interactive Prediction Dashboard
The notebook features an interactive widget that allows users to get temperature predictions for a specific city, country, and date. The dashboard uses the trained XGBoost model and finds the nearest training data point to fill in necessary features for the prediction.

Impact Statement
Sustainability-Tracker empowers stakeholders (farmers, energy planners, city officials) to visualize forecasts, run scenario experiments, and make informed decisions by providing an interactive platform based on cleaned NOAA GSOD data and integrated predictive models. The interactive experiment widget facilitates rapid hypothesis testing and localized decision support.

Conclusion
This sample notebook demonstrates the integration of historical data, machine learning, and interactive visualization for short-term temperature forecasting. It highlights scalable data access via BigQuery (though the sample uses local files), both batch and interactive inference capabilities, and a reproducible workflow. Future iterations could focus on improving coverage, uncertainty quantification, and prediction latency.

Colab paid products - Cancel contracts here

Can you write a professional readme.md for this notebook? just look at the notebook and explain it. Also cover topics like controls for navigation inside the 3d globe plot. All the best
Absolutely! I can generate a professional README.md for your notebook. It will cover the project's purpose, data, methods, visualizations (including instructions for the 3D globe), and the interactive prediction dashboard.

 
What can I help you build?
0 / 2000
Gemini can make mistakes so double-check it and use code with caution. Learn more
