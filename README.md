# Daily Time Series Explorer

## Goal
Streamlit app that allows the user to import their own time series CSV file and that gives a quick overview of time series features
HEADS UP: Only works with daily time series for now.
Powered by: Streamlit

## Features
- Bring own dataset or load sample data that I have provided 
- Overview of the series: periodicity, number of obs., start, end, etc.
- Moving Averages
- Decomposition: Trend, Seasonality, etc.
- Test for time series Stationarity
- Autocorrelation
- Add holidays to use as external regressors
- Forecasting with basic-ish model (Prophet)
- Detect and flag anomalies

## Assumptions and requirements
- VERY IMPORTANT: **It only works with daily time series so far**, haven't tested with other frequencies (but it shouldn't take much effort to be able to handle monthly or weekly series, at least)
- If there are any missing values, fill with pandas' `fillna(method = "backfill")`
- If you want to bring your own (daily) data, there are a few requirements:
    - Comma-separated CSV file
    - First column has to be the date (and has to be named `date`), second column is the value, and it doesn't matter the name

## Next steps
DISCLAIMER: This was developed as a (not-so) little side-project to practice Python and to showcase some skills in ML, deployment and web apps, so I don't expect to keep this one super updated and add that many features, but if you liked what you've seen and want to give me feedback or suggestions, it would be very much appreciated.
Here are some possible future directions I could go:
- Implement Level drift detection
- Improve model validation methods
- Add more robust residuals diagnostics
- Include an automated framework like `pycaret` or `H2o` for training and hypertuning several models at once
- Test more Time Series features based on `pytimetk` or `tsfresh`

Some screenshots:

![image](https://github.com/rafabelokurows/time_series_explorer/assets/55976107/2812c83c-c0c1-4448-a5e3-eb914992e1c5)

![image](https://github.com/rafabelokurows/time_series_explorer/assets/55976107/03161477-6f32-455c-bfdd-ba4dea7c3577)

![image](https://github.com/rafabelokurows/time_series_explorer/assets/55976107/6852f274-f5eb-4657-a37b-b859c4c8e0ad)
