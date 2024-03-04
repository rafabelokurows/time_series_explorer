#### SETUP ####
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pytimetk as tk
import holidays
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet import Prophet

color_pal = sns.color_palette()
line_color_1 = '#6b9080'

#### STREAMLIT PAGE #### 
st.set_page_config(page_title="Time Series Analysis", layout = 'wide', initial_sidebar_state = 'auto',page_icon = '📈')

st.title("Time Series Auto Explorer")
st.markdown("Load your own CSV file to see basic stats, perform statistical checks and a simple little forecasting for your time series")

#### SIDEBAR ####
uploaded_file = st.sidebar.file_uploader("Choose a Time Series dataset",help="Teste",accept_multiple_files=False)
option = st.sidebar.selectbox(
    "Don't have any datasets? Try some sample data",
    ('DEI - Economic Indicator for Portugal (2020-current)', 'EUR to USD rates (2004-current)','Amazon Stock data (2013-current)',None),index=None,
   placeholder="Select dataset")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if option is not None:
    st.sidebar.write('You selected: ', option)
elif uploaded_file is not None:
    st.sidebar.write('You brought your own data!')
    st.sidebar.success("File uploaded successfully!")
    #st.sidebar.subheader("Name of your file: "+uploaded_file.name)
st.sidebar.markdown("  \n  \n**Author**: Rafael Belokurows")
st.sidebar.markdown("rafabelokurows@gmail.com")
st.sidebar.markdown("**Version:** 0.1")

#### MAIN STUFF ####
def frequency(df):
    diffs = (df.index[1:] - df.index[:-1])
    min_delta = diffs.min()
    mask = (diffs == min_delta)[:-1] & (diffs[:-1] == diffs[1:])
    pos = np.where(mask)[0][0]
    idx = df.index
    freq = pd.infer_freq(idx[pos: pos + 3])
    if freq == 'D':
        return 'Daily'
    elif freq == 'W':
        return 'Weekly'
    elif freq == 'M':
        return 'Monthly'
    else:
        return None

def read_data(filename):
    ts_data = pd.read_csv(filename)
    ts_data = ts_data.set_index('date')
    ts_data.index = pd.to_datetime(ts_data.index) #converting date to DateTimeIndex
    ts_data.rename(columns={ts_data.columns[0]: 'value'},inplace=True)
    return ts_data


if uploaded_file is not None:
    filename = uploaded_file
    ts_data = read_data(filename)
elif option == "DEI - Economic Indicator for Portugal (2020-current)":
    filename = "C:\\Users\\BELOKUROWSR\\Desktop\\time_series_explorer\\dei.csv"
    ts_data = read_data(filename)
elif option == "EUR to USD rates (2004-current)":
    filename = "C:\\Users\\BELOKUROWSR\\Desktop\\time_series_explorer\\eur_to_usd.csv"
    ts_data = read_data(filename)
elif option == "Amazon Stock data (2013-current)":
    filename = "C:\\Users\\BELOKUROWSR\\Desktop\\time_series_explorer\\amazon_stock.csv"
    ts_data = read_data(filename)
else:
    st.info("Waiting for dataset...",icon="ℹ️")

if uploaded_file is not None or option is not None:
    #### LOADING FILES ####
    #ts_data = pd.read_csv(uploaded_file,sep=";",parse_dates=['date'],date_format = "%d/%m/%Y") #parsing date as it was in Portuguese date format

    #df = pd.read_csv(uploaded_file)
    #df = pd.concat([ts_data.head(),pd.DataFrame([None]*len(ts_data.columns)),ts_data.tail()]).drop(columns=[0])
    #df.index = df.index.normalize()
    st.subheader("Basic stats")
    #col1, col2, col3 = st.columns(3, gap="small")
    df_metrics = ts_data[['value']].agg(['mean','min','count','max','median'])
    main_container = st.container(border=False)
    main1, main2, main3, main4,main5 = main_container.columns(5, gap="small")
    main1.markdown("Beginning of the series")
    main1.table(ts_data.head())
    
    main2.markdown("End of the series")
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)
    main2.table(ts_data.tail())
    main3.markdown("Summary Statistics")
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(3) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)
    main3.table(df_metrics)
    na_values = ts_data.value.isna().sum()
    # Conducting the Augmented Dickey-Fuller test to check for stationarity
    if na_values > 0:
        ts_data['value'] = ts_data['value'].fillna(method="bfill")
    adf_test = adfuller(ts_data['value'], autolag='AIC')
    # Outputting the results
    adf_output = pd.Series(adf_test[0:2], index=['Test Statistic', 'p-value'])
    for key, value in adf_test[4].items():
        adf_output[f'Critical Value ({key})'] = value
    adf_out_df = pd.DataFrame(adf_output,columns=['result'])
    main4.markdown("ADF test for Stationarity")
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(4) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)
    main4.table(adf_out_df)

    def highlight_significant(val):
        if val < 0.05:
            return 'background-color: yellow'
        else:
            return ''
    #main4.dataframe(adf_out_df,column_config={"_index":None,"index":"ADF test - Stationarity"})

    p_value = adf_output.reset_index().query("index == 'p-value'").iloc[0, 1]
    if p_value > 0.05:
        main5.text(f"Time Series looks to be Non-Stationary\np-value = {p_value:.3f}")
    else:
        main5.text(f"Time Series is Stationary\np-value = {p_value:.3f}")
    freq_timeseries = frequency(ts_data)
    main5.text(f'Time Series Frequency: {frequency(ts_data)}')
    
    main5.text(f'Missing values: {na_values}')

    ts_data['weekly_mean'] = ts_data['value'].resample('W', label='left', closed='left').mean()
    ts_data['rolling_mean'] = ts_data.value.rolling(window=60).mean()

    # Create the line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter( x=ts_data.index,y= ts_data.value, name='Value',
                        line_shape='linear',line_color=line_color_1,opacity=0.8))
    fig.add_trace(go.Scatter( x=ts_data.index,y= ts_data.rolling_mean, name='60-day rolling window',
                        line_shape='linear',line_color='#669bbc'))
    fig.add_trace(go.Scatter( x=ts_data.index,y= ts_data.weekly_mean, name="Weekly Mean",
                        line_shape='spline',line_color='#f28482'))
    fig.update_traces(connectgaps=True)
    # Customize the layout
    fig.update_layout(
        title="Time Series",
        yaxis_title="Value",
        height=750
    )

    decomposition = sm.tsa.seasonal_decompose(ts_data['value'], model='additive',period=365)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, mode='lines', name='Trend',line_color=line_color_1))
    fig_trend.update_layout(
        title="Trend Component",
        xaxis_title="",
        yaxis_title="",
        height=250
    )
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, mode='lines', name='Seasonal',line_color=line_color_1))
    fig_seasonal.update_layout(
        title="Seasonal Component",
        xaxis_title="",
        yaxis_title="",
        height=250
    )
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, mode='lines', name='Residuals',line_color=line_color_1))
    fig_residual.update_layout(
        title="Residuals",
        xaxis_title="",
        yaxis_title="",
        height=250
    )

    col4, col5 = st.columns([0.7, 0.3], gap="small")
    col4.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)
    col5.plotly_chart(fig_trend)
    col5.plotly_chart(fig_seasonal)
    col5.plotly_chart(fig_residual)

    st.subheader("Further Analysis")

    col6, col7, col8 = st.columns([0.30, 0.25,0.45], gap="small")

    fig_hist = px.histogram(ts_data, x='value',  marginal=None, color_discrete_sequence=['#6b9080'],histnorm='density',nbins = 50,
                    hover_data=ts_data.columns,height=520,title= "Distribution")
    col6.plotly_chart(fig_hist,theme="streamlit", use_container_width=True)

    #https://community.plotly.com/t/plot-pacf-plot-acf-autocorrelation-plot-and-lag-plot/24108/3
    from statsmodels.tsa.stattools import pacf, acf
    df_pacf = pacf(ts_data['value'], nlags=30)
    df_acf = acf(ts_data['value'], nlags=30)
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Scatter(
        x= np.arange(len(df_acf)),
        y= df_acf,
        name= 'PACF',line_color=line_color_1
        ))
    fig_acf.update_xaxes(rangeslider_visible=False)
    fig_acf.update_layout(
        title="Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
            autosize=False,
        #     width=500,
            height=250,
        )
    col7.plotly_chart(fig_acf,theme="streamlit", use_container_width=True)
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Scatter(
        x= np.arange(len(df_pacf)),
        y= df_pacf,
        name= 'PACF',line_color=line_color_1
        ))
    fig_pacf.update_xaxes(rangeslider_visible=False)
    fig_pacf.update_layout(
        title="Partial Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
            autosize=False,
        #     width=500,
            height=250,
        )
    col7.plotly_chart(fig_pacf,theme="streamlit", use_container_width=True)

    anomalize_df = tk.anomalize(
        data          = ts_data.reset_index(),
        date_column   = 'date',
        value_column  = 'value',
        iqr_alpha     = 0.03, # using the default
        clean_alpha   = 0.75, # using the default
        clean         = "min_max",
        max_anomalies = 0.1,
        trend=None
    )

    fig_anomaly = tk.plot_anomalies(
        data        = anomalize_df,
        date_column = 'date',
        engine      = 'plotly',
        title       = 'Anomaly Detection',
        line_color=line_color_1
    )
    fig_anomaly.update_layout(height = 520,showlegend=False)
    col8.plotly_chart(fig_anomaly,theme="streamlit", use_container_width=True)

    st.subheader("Modeling and Preparation - Features added to the Time Series")
    hol_pt = holidays.country_holidays('PT',  years=range(ts_data.index.min().year,
                                                                ts_data.index.max().year+1))
    ts_data = ts_data \
        .reset_index()\
        .augment_lags(
            date_column  = 'date',
            value_column = 'value',
            lags         = (1, 3)
        )\
        .augment_timeseries_signature(date_column = 'date')
    ts_data['is_holiday'] = ts_data.index.isin(hol_pt)
    st.dataframe(ts_data.head(10), use_container_width=True)

    ts_train = ts_data.reset_index().rename(columns={'date':'ds','value':'y'}).loc[:,['ds','y','is_holiday','date_wday','date_month']]

    #https://facebook.github.io/prophet/docs/quick_start.html#python-api
    st.subheader("Forecasting with Prophet")
    model = Prophet(changepoint_prior_scale= 0.5, seasonality_prior_scale=0.01)
    model.add_regressor('is_holiday')
    model.add_regressor('date_wday')
    model.add_regressor('date_month')
    model.fit(ts_train)

    future = model.make_future_dataframe(periods=90)\
            .augment_timeseries_signature(date_column = 'ds')\
            .rename(columns={"ds_wday":"date_wday","ds_month":"date_month"})
    future['is_holiday'] = future.index.isin(hol_pt)
    #future.tail()
    forecast = model.predict(future)
    teste = model.plot_components(forecast)
    plotly_prophet = plot_plotly(model, forecast,xlabel='', ylabel='Value',uncertainty=True, plot_cap=True)
    plotly_prophet.update_layout(showlegend=False)
    plotly_prophet.data[0].marker.color = "#F8C9FE"
    plotly_prophet.data[2].line.color = "#4B8A6F"
    plotly_prophet.layout.xaxis.rangeslider.visible = False
    st.plotly_chart(plotly_prophet, use_container_width=True)



st.markdown('-----------------------------------------------------')
st.markdown('Developed by [Rafael Belokurows](https://github.com/rafabelokurows) - 2024')
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
#st.sidebar.header('Filter listings')
#outside = st.sidebar.checkbox('View listings outside of Porto municipality')
#values = st.sidebar.slider("Price range ($)", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (50., 300.))
#min_nights_values = st.sidebar.slider('Minimum Nights', 0, 30, (1))
