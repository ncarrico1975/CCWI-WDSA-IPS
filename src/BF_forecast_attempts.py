import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import sqrt
import datetime
import xgboost as xgb

# df_holidays = pd.read_excel('documents/Holidays.xlsx',parse_dates=True)    #Importa medições caudal entrada
df_holidays = pd.read_csv('documents/Holidays.txt',parse_dates=True)    #Importa medições caudal entrada
# holidays = np.genfromtxt('documents/Holidays.txt', )

df_inflow = pd.read_excel('documents/InflowData_1.xlsx')    #Importa medições caudal entrada

transf_dict_inflow = {
    "Date-time CET-CEST (DD/MM/YYYY HH:mm)": "datetime",
    "DMA A (L/s)": "dma_A",
    "DMA B (L/s)": "dma_B",
    "DMA C (L/s)": "dma_C",
    "DMA D (L/s)": "dma_D",
    "DMA E (L/s)": "dma_E",
    "DMA F (L/s)": "dma_F",
    "DMA G (L/s)": "dma_G",
    "DMA H (L/s)": "dma_H",
    "DMA I (L/s)": "dma_I",
    "DMA J (L/s)": "dma_J"  
}
df_inflow.rename(columns= transf_dict_inflow, inplace=True) #corrige nome colunas
df_inflow['datetime'] = pd.to_datetime(df_inflow['datetime'],format='%d/%m/%Y %H:%M')
df_inflow.set_index('datetime', inplace = True) #passa timestamps a index
# print(df_inflow)



df_weather = pd.read_excel('documents/WeatherData_1.xlsx')  #importa medições meteorológicas

transf_dict_weather = {
    "Date-time CET-CEST (DD/MM/YYYY HH:mm)": "datetime",
    "Rainfall depth (mm)": "prcp",
    "Air temperature (°C)": "atmp",
    "Air humidity (%)": "ahum",
    "Windspeed (km/h)": "wspd"
}
df_weather.rename(columns= transf_dict_weather, inplace=True)   #corrige nome colunas
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'],format='%d/%m/%Y %H:%M')
df_weather.set_index('datetime', inplace = True) #passa timestamps a index
# print(df_weather)



# pd.options.plotting.backend = "plotly"

# fig_inflow = df_inflow.plot()
# # fig_weather = df_weather.plot()

# fig_inflow.show()
# # fig_weather.show()



unique_days_inflow = df_inflow.index.map(lambda t: t.date()).unique()   #get list of unique days
unique_days_weather = df_weather.index.map(lambda t: t.date()).unique()   #get list of unique days

# print('There are ',len(unique_days_inflow),' days of inflow data:', unique_days_inflow)
# print('There are ',len(unique_days_weather),' days of weather data', unique_days_weather)



#######################    parameters
number_of_days_in_history = 30
days_to_forecast = 6
last_day_in_history = '2022-07-18'

forecasting_methods = [
                    'naive_avg',
                    'naive_median',
                    'naive_ewma',
                    'XGBOOST_naive',
]

forecasting_dmas = [
                    'dma_A',
                    'dma_B',
                    'dma_C',
                    # 'dma_D',
                    # 'dma_E',
                    # 'dma_F',
                    # 'dma_G',
                    # 'dma_H',
                    # 'dma_I',
                    # 'dma_J'
]

dict_colors ={
    "dma_A": "blue",
    "dma_B": "red",
    "dma_C": "green",
    "dma_D": "orange",
    "dma_E": "yellow",
    "dma_F": "purple",
    "dma_G": "black",
    "dma_H": "cyan",
    "dma_I": "darkred",
    "dma_J": "teal",

}

###########################






def naive(df_train=None,test_length='single_day',forecast_type='avg',history_type='day_by_day'): #média, mediana, ewma

    dma_name = df_train.columns[0]

    #history_type can be "day_by_day" or "weekday_vs_weekend"

    df_train = df_train.copy()

    df_train = df_train.dropna()

    if history_type == 'day_by_day':
        df_train['day_of_week'] = df_train.index.day_name()
    elif history_type == 'weekday_vs_weekend':
        df_train['day_of_week'] = df_train.index.weekday

    df_train['hour'] = df_train.index.hour
    df_train['minute'] = df_train.index.minute


    descriptive_statistics_by_dayweek_hour_minute = df_train.groupby(['day_of_week', 'hour', 'minute']).describe()

    # print('Descriptive Statistics by Day and Hour: \n', descriptive_statistics_by_dayweek_hour_minute)

    frequency = int(np.diff(df_train.index).min()/ np.timedelta64(1, 's'))#in seconds
    number_measurements_in_a_day = int(24 / (frequency/3600)) 

    if test_length=='single_day': #1 dia
        forecasting_timestamps = pd.date_range(start=df_train.index[-1]+pd.to_timedelta(str(frequency)+'s'),freq=str(frequency)+'s', periods=number_measurements_in_a_day)
    else:#os periodos pedidos
        forecasting_timestamps = pd.date_range(start=df_train.index[-1]+pd.to_timedelta(str(frequency)+'s'),freq=str(frequency)+'s', periods=test_length)

    predictions = pd.DataFrame(index=forecasting_timestamps,columns=[dma_name])

    for index, row in predictions.iterrows():
        hour = index.hour
        minute = index.minute

        #Cria bloco de dados, que depende se o tipo de historico é "day_by_day" ou "weekday_vs_weekend"
        if history_type == 'day_by_day':
            day_of_week = index.day_name()
            block = df_train[(df_train['day_of_week']==day_of_week) & 
                             (df_train['hour']==hour) & 
                             (df_train['minute']==minute)]

        elif history_type == 'weekday_vs_weekend':
            day_of_week = index.weekday()

            if day_of_week >=0 and day_of_week<=4:#dias de semana
                block = df_train[(df_train['day_of_week']>=0) & 
                                (df_train['day_of_week']<=4) & 
                                (df_train['hour']==hour) & 
                                (df_train['minute']==minute)]
            elif day_of_week ==5 or day_of_week==6: #fins de semana
                block = df_train[(df_train['day_of_week']>4) & 
                                (df_train['hour']==hour) & 
                                (df_train['minute']==minute)]

        

        if forecast_type=='avg':
            value = block[dma_name].mean()
        elif forecast_type=='ewma':
            value = block[dma_name].ewm(com=0.5,min_periods=block[dma_name].size).mean()[-1]
        elif forecast_type=='median':
            value = block[dma_name].median()

        predictions.loc[index,dma_name] = value

    return predictions

def measure_rmse(test, predicted):

    predicted_array=predicted.tolist()
    test_array=test.tolist()

    test_nans = np.argwhere(np.isnan(test_array))[:,0].tolist() #check if there are missing values. Both arrays must present the same size
    predicted_nans = np.argwhere(np.isnan(predicted_array))[:,0].tolist()
    total_nans = test_nans+predicted_nans

    if len(total_nans) != 0:
        test = np.delete(test, total_nans) 
        predicted = np.delete(predicted, total_nans) 

    mse = np.mean((test - predicted)**2)
    rmse = sqrt(mse)

    return rmse

def xgboost_naive(history_inflow_df=None, history_weather_df = None, expected_weather_df = None, df_holidays = None, test_length='single_day'):

    dma_name = history_inflow_df.columns[0]


    history_inflow_df=history_inflow_df.copy()
    history_inflow_df = history_inflow_df.dropna()
    history_weather_df = history_weather_df.copy()
    expected_weather_df = expected_weather_df.copy()


    def create_features_xgboost(history_inflow_df, history_weather_df, df_holidays):
        history_inflow_df = history_inflow_df.copy()
        history_inflow_df['hour'] = history_inflow_df.index.hour
        history_inflow_df['dayofweek'] = history_inflow_df.index.dayofweek


        complete_holiday_list = df_holidays.holiday.values
        complete_holiday_list = [datetime.datetime.strptime(date, '%d/%m/%Y').date() for date in complete_holiday_list]

        for i in range(history_inflow_df.shape[0]):
            timestamp = history_inflow_df.index[i]
            date = timestamp.date()
            if date in complete_holiday_list:
                history_inflow_df.loc[timestamp,'holiday']=1
            else:
                history_inflow_df.loc[timestamp,'holiday']=0


        merged_df = pd.merge(history_inflow_df, history_weather_df, left_index=True, right_index=True,how='left')
        return merged_df

    df_features = create_features_xgboost(history_inflow_df, history_weather_df, df_holidays)

    features=[               #selecionar que features entram para o xgboost
                'hour',
                'dayofweek',
                'holiday',
                'prcp',
                'atmp',
                'ahum', 
                'wspd',
                ]
    target=dma_name


    X = df_features[features]
    y = df_features[target]

    frac=0.8
    frac_id = int(frac * len(X))

    X_train = X.iloc[:frac_id,:]
    X_test = X.iloc[frac_id:,:]

    y_train = y.iloc[:frac_id]
    y_test = y.iloc[frac_id:]

    frequency = int(np.diff(history_inflow_df.index).min()/ np.timedelta64(1, 's'))#in seconds
    number_measurements_in_a_day = int(24 / (frequency/3600)) 

    if test_length=='single_day': #1 dia
        forecasting_timestamps = pd.date_range(start=history_inflow_df.index[-1]+pd.to_timedelta(str(frequency)+'s'),freq=str(frequency)+'s', periods=number_measurements_in_a_day)
    else:#os periodos pedidos
        forecasting_timestamps = pd.date_range(start=history_inflow_df.index[-1]+pd.to_timedelta(str(frequency)+'s'),freq=str(frequency)+'s', periods=test_length)


    X_pred = pd.DataFrame(index=forecasting_timestamps)
    X_pred = create_features_xgboost(X_pred, expected_weather_df, df_holidays)
    X_pred = X_pred[features]

    regressor_XGBOOST = xgb.XGBRegressor(base_score=0.5, 
                                        booster='gbtree',    
                                        n_estimators=10000,
                                        early_stopping_rounds=100,
                                        objective='reg:squarederror',
                                        max_depth=6,
                                        learning_rate=0.1)

    regressor_XGBOOST.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        verbose=False)

    y_pred = regressor_XGBOOST.predict(X_pred) 
    y_pred_df = pd.DataFrame(index=X_pred.index, columns=[dma_name],data=y_pred)

    
    return y_pred_df





#Ranges to forecast
range_of_days_to_forecast = pd.date_range(pd.to_datetime(last_day_in_history)+pd.Timedelta(days=1),freq='d',periods=days_to_forecast)
range_to_forecast = pd.date_range(pd.to_datetime(last_day_in_history)+pd.Timedelta(days=1),freq='H',periods=days_to_forecast*24)



#Empty DF's to receive forecasts
if 'naive_avg' in forecasting_methods: df_forecasts_naive_average = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_median' in forecasting_methods: df_forecasts_naive_median = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_ewma' in forecasting_methods: df_forecasts_naive_ewma = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'XGBOOST_naive' in forecasting_methods: df_forecasts_xgboost_naive = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)


#Empty df's to receive error metric results
if 'naive_avg' in forecasting_methods: naive_avg_rmse_by_day = pd.DataFrame(columns=forecasting_dmas, index=range_of_days_to_forecast.date)
if 'naive_median' in forecasting_methods: naive_median_rmse_by_day = pd.DataFrame(columns=forecasting_dmas, index=range_of_days_to_forecast.date)
if 'naive_ewma' in forecasting_methods: naive_ewma_rmse_by_day = pd.DataFrame(columns=forecasting_dmas, index=range_of_days_to_forecast.date)
if 'XGBOOST_naive' in forecasting_methods: xgboost_naive_rmse_by_day = pd.DataFrame(columns=forecasting_dmas, index=range_of_days_to_forecast.date)





for day_to_forecast in range_of_days_to_forecast:   #para cada dia

    history_inflow_df = df_inflow.loc[(df_inflow.index > day_to_forecast - pd.Timedelta(days = number_of_days_in_history)) & 
                                      (df_inflow.index < day_to_forecast)]  #obtem o histórico de medições para N dias 
    history_weather_df = df_weather.loc[(df_weather.index > day_to_forecast - pd.Timedelta(days = number_of_days_in_history)) & 
                                        (df_weather.index < day_to_forecast)]   #Obtem o histórico de weather_data para N dias
    expected_weather_df = df_weather.loc[(df_weather.index > day_to_forecast) & (df_weather.index < day_to_forecast + pd.Timedelta(days = 1))]

    true_df = df_inflow.loc[df_inflow.index.date == day_to_forecast.date()]


    for dma in forecasting_dmas:    #para cada dma

        if 'naive_avg' in forecasting_methods:    
            forecast_naive_average = naive(df_train=history_inflow_df.loc[:,[dma]], 
                                        test_length='single_day',
                                        forecast_type='avg',
                                        history_type='day_by_day')
            df_forecasts_naive_average.loc[forecast_naive_average.index,dma] = forecast_naive_average.iloc[:,0]

            x=forecast_naive_average.index
            y_pred=forecast_naive_average[dma].values
            y_true=true_df.loc[x,dma].values
            naive_avg_rmse_by_day.loc[x[0].date(),dma] = measure_rmse(y_true, y_pred)


        if 'naive_median' in forecasting_methods:    
            forecast_naive_median = naive(df_train=history_inflow_df.loc[:,[dma]], 
                                        test_length='single_day',
                                        forecast_type='median',
                                        history_type='day_by_day')
            df_forecasts_naive_median.loc[forecast_naive_median.index,dma] = forecast_naive_median.iloc[:,0]

            x=forecast_naive_median.index
            y_pred=forecast_naive_median[dma].values
            y_true=true_df.loc[x,dma].values
            naive_median_rmse_by_day.loc[x[0].date(),dma] = measure_rmse(y_true, y_pred)


        if 'naive_ewma' in forecasting_methods:    
            forecast_naive_ewma = naive(df_train=history_inflow_df.loc[:,[dma]], 
                                        test_length='single_day',
                                        forecast_type='ewma',
                                        history_type='day_by_day')
            df_forecasts_naive_ewma.loc[forecast_naive_ewma.index,dma] = forecast_naive_ewma.iloc[:,0]

            x=forecast_naive_ewma.index
            y_pred=forecast_naive_ewma[dma].values
            y_true=true_df.loc[x,dma].values
            naive_ewma_rmse_by_day.loc[x[0].date(),dma] = measure_rmse(y_true, y_pred)


        if 'XGBOOST_naive' in forecasting_methods:
            forecast_xgboost_naive = xgboost_naive(history_inflow_df=history_inflow_df.loc[:,[dma]], 
                                                   history_weather_df = history_weather_df,
                                                   expected_weather_df = expected_weather_df, 
                                                   df_holidays = df_holidays, 
                                                   test_length='single_day') 
            df_forecasts_xgboost_naive.loc[forecast_xgboost_naive.index,dma] = forecast_xgboost_naive.iloc[:,0]

            x=forecast_xgboost_naive.index
            y_pred=forecast_xgboost_naive[dma].values
            y_true=true_df.loc[x,dma].values
            xgboost_naive_rmse_by_day.loc[x[0].date(),dma] = measure_rmse(y_true, y_pred)


#plots

# Create a list to store traces
traces = []


for dma in df_inflow[forecasting_dmas].columns:    #real measurements
    trace = go.Scatter(x=df_inflow.index, y=df_inflow[dma], mode='lines', name=f'Real - {dma}',opacity=0.5, line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
    traces.append(trace)

if 'naive_avg' in forecasting_methods:
    for dma in df_forecasts_naive_average.columns:    #forecasts_naive_average
        trace = go.Scatter(x=df_forecasts_naive_average.index, y=df_forecasts_naive_average[dma], mode='lines', name=f'Naive Avg - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)

if 'naive_median' in forecasting_methods:
    for dma in df_forecasts_naive_median.columns:    #forecasts_naive_average
        trace = go.Scatter(x=df_forecasts_naive_median.index, y=df_forecasts_naive_median[dma], mode='lines', name=f'Naive Median - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)

if 'naive_ewma' in forecasting_methods:
    for dma in df_forecasts_naive_ewma.columns:    #forecasts_naive_ewma
        trace = go.Scatter(x=df_forecasts_naive_ewma.index, y=df_forecasts_naive_ewma[dma], mode='lines', name=f'Naive Ewma - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)

if 'XGBOOST_naive' in forecasting_methods:
    for dma in df_forecasts_xgboost_naive.columns:    #forecasts_naive_ewma
        trace = go.Scatter(x=df_forecasts_xgboost_naive.index, y=df_forecasts_xgboost_naive[dma], mode='lines', name=f'Naive XGBOOST - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)


fig = go.Figure(data=traces)

fig.show()

print('naive_avg_rmse_by_day \n',naive_avg_rmse_by_day)
print('naive_median_rmse_by_day \n',naive_median_rmse_by_day)
print('naive_ewma_rmse_by_day \n',naive_ewma_rmse_by_day)
print('xgboost_naive_rmse_by_day \n',xgboost_naive_rmse_by_day)

