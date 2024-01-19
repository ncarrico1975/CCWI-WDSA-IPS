import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import sqrt
import datetime
from methods.naive import naive
from methods.xgboost import xgboost_naive
from methods.svr import svr
from methods.quevedo import quevedo
from methods.metrics import BattleMetrics_per_dma
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.layers import (BatchNormalization, Dense,  
                TimeDistributed, Bidirectional, 
                SimpleRNN, GRU, LSTM, Dropout)

from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.layers import Dense, SimpleRNN, Dropout




### FOR NOW LETS KEEP THE LSTM MODEL HERE. ALL THE OTHER MODELS ARE IN SPECIFIC FOLDERS AND FILES

#### Parameters for LSTM ####
WINDOW     = 10
EPOCHS     = 200 #number of complete passes 
PATIENCE   = 10  #number of epochs with no improvement after which training will be stopped
BATCH_SIZE = 32  #number of training samples to work through before the model’s internal parameters are updated
#############################

### Auxiliary functions to train LSTM model
def one_step_forecast(df, window):
    d = df.values
    x = []
    n = len(df)
    idx = df.index[:-window]
    for start in range(n-window):
        end = start + window
        x.append(d[start:end])
    cols = [f'x_{i}' for i in range(1, window+1)]
    x = np.array(x).reshape(n-window, -1)
    y = df.iloc[window:].values
    df_xs = pd.DataFrame(x, columns=cols, index=idx)
    df_y = pd.DataFrame(y.reshape(-1), columns=['y'], index=idx)
    return pd.concat([df_xs, df_y], axis=1).dropna()

def split_data(df, val_split=0.15, n_days_to_forecast=7):
    #test data = days to forecast * 24
    n_test = n_days_to_forecast*24
    n_train_val = len(df)-n_test
    n = int(n_train_val * val_split)
    train, val, test = df[:-n], df[-n:n_test], df[n_test:]
    return train,val


class Standardize:
    def __init__(self, df, n_days_test, split=0.15):
        self.data = df
        self.split = split
    
    def split_data(self):
        n_test = n_days_test*24 #size of test data
        train, test = self.data.iloc[:-n_test], self.data.iloc[-n_test:]
        n = int(len(train) * self.split)
        train, val = train.iloc[:-n], train.iloc[-n:]
        assert len(train) + len(val)+len(test) == len(self.data)
        return train, val, test

   
    def _transform(self, data):
        data_s = (data - self.mu)/self.sigma
        return data_s
    
    def fit_transform(self):
        train,  val, test = self.split_data()
        self.mu, self.sigma = train.mean(), train.std()
        train_s = self._transform(train)
        val_s = self._transform(val)
        test_s = self._transform(test)
        return train_s,  val_s, test_s
         
    def inverse(self, data):
        return (data * self.sigma)+self.mu

'''
 Create the features_target_ts function that 
 takes a dataset and returns an x split (independent variables or features) 
 and y split (dependent or target variables):
'''

def features_target_ts(*args):
    y = [col.pop('y').values.reshape(-1, 1) for col in args]
    x = [col.values.reshape(*col.shape, 1)
                   for col in args]
    return *y, *x


def create_model(train, units, dropout=0.2):
    model = keras.Sequential()
    model.add(LSTM(units=units,
                   input_shape=(train.shape[1], 
                                train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def train_model_ts(model, 
                x_train, y_train, x_val, y_val, 
                epochs = EPOCHS , 
                patience= PATIENCE, 
                batch_size= BATCH_SIZE):
                
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[RootMeanSquaredError(), 
                           MeanAbsoluteError()])
    
    es = keras.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    min_delta=0, 
                    patience=patience)
    
    history = model.fit(x_train,y_train, 
              shuffle=False, epochs=epochs,
              batch_size=batch_size, 
              validation_data=(x_val, y_val),
              callbacks=[es], verbose=1)
    return history
    

def lstm(df_train = None, window = WINDOW, n_days_test = 7): 

    dma_name = df_train.columns[0]
    
    inflow = df_train[dma_name]

    inflow_one_step = one_step_forecast(inflow, window)

    scale_inflow = Standardize(inflow_one_step, n_days_test)

    train,val, test = scale_inflow.fit_transform()
    
    (y_train, y_val, y_test, x_train, x_val, x_test) = features_target_ts(train, val, test)

    tf.keras.backend.clear_session()
    
    model_lstm = create_model(train=x_train, units=32)

    history_lstm = train_model_ts(model_lstm, x_train, y_train, x_val, y_val)

    predictions = model_lstm.predict(x_test)
    #to scale back
    mu, sigma =np.mean(inflow), np.std(inflow)
    predictions = predictions*sigma
    predictions = predictions+mu

    return predictions
#######################333




df_holidays = pd.read_csv('documents/Holidays.txt',parse_dates=True)    #import holidays
# df_inflow = pd.read_excel('documents/InflowData_1.xlsx')    #import original inflow data
df_inflow = pd.read_csv('treated_series/inflow_completed_in_R.csv')     #import reconstructed inflow data
df_weather = pd.read_excel('documents/WeatherData_1.xlsx')  #import original weather data
# df_weather = pd.read_csv('treated_series/weather_completed_in_R.csv')  #import reconstructed weather data

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
df_inflow.rename(columns= transf_dict_inflow, inplace=True) #to rename columns
df_inflow['datetime'] = pd.to_datetime(df_inflow['datetime'],format='%d/%m/%Y %H:%M')  #datetime with correct format
df_inflow.set_index('datetime', inplace = True) #to make the timestamps become row indices


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



#######################    parameters
last_day_train = '2022-07-17' #last day considered for our training/validation data
n_days_train   = 60           # length of our train/validation data (counting from the last_day_train into the past)
n_days_test    = 7            # lenght of our test data (counting from the next day to the last_day_train)



# comment out the models you don't want to run
forecasting_methods = [
                    'naive_avg',
                    'naive_median',
                    'naive_ewma',
                    'xgboost_naive',
                    'lstm',
                    'svr',
                    'quevedo',
                    ]

# comment out the variables dmas you don't want to run
forecasting_dmas = [
                    'dma_A',
                    'dma_B',
                    'dma_C',
                    'dma_D',
                    'dma_E',
                    'dma_F',
                    'dma_G',
                    'dma_H',
                    'dma_I',
                    'dma_J'
                    ]


#just for plotting
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



#prepare the expected range of forecast (the datetime index)
last_day_train = pd.to_datetime(last_day_train)
range_to_forecast = pd.date_range(pd.to_datetime(last_day_train)+pd.Timedelta(days=1),freq='H',periods=n_days_test*24)


#Empty DF's to receive forecasts
if 'lstm' in forecasting_methods: df_forecasts_lstm = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_avg' in forecasting_methods: df_forecasts_naive_avg = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_median' in forecasting_methods: df_forecasts_naive_median = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_ewma' in forecasting_methods: df_forecasts_naive_ewma = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'xgboost_naive' in forecasting_methods: df_forecasts_xgboost_naive = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'svr' in forecasting_methods: df_forecasts_svr = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'quevedo' in forecasting_methods: df_forecasts_quevedo = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)

#Empty df's to receive error metric results
global_rmse = pd.DataFrame(index=forecasting_methods,columns=forecasting_dmas)
if 'lstm' in forecasting_methods: lstm_rmse = pd.DataFrame(columns=forecasting_dmas)
if 'naive_avg' in forecasting_methods: naive_avg_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_median' in forecasting_methods: naive_median_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_ewma' in forecasting_methods: naive_ewma_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'xgboost_naive' in forecasting_methods: xgboost_naive_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'svr' in forecasting_methods: svr_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'quevedo' in forecasting_methods: quevedo_rmse = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)

#Empty df's to receive battle metric results
global_battle_metrics = pd.DataFrame(index=forecasting_methods,columns=np.append(forecasting_dmas,'sum'))
if 'lstm' in forecasting_methods: lstm_battle_metrics= pd.DataFrame(columns=forecasting_dmas)
if 'naive_avg' in forecasting_methods: naive_avg_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_median' in forecasting_methods: naive_median_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'naive_ewma' in forecasting_methods: naive_ewma_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'xgboost_naive' in forecasting_methods: xgboost_naive_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'svr' in forecasting_methods: svr_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)
if 'quevedo' in forecasting_methods: quevedo_battle_metrics = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)



#to slice dataframes according to the time range defined for the training data
history_inflow_df = df_inflow.loc[(df_inflow.index > last_day_train-pd.Timedelta(days = n_days_train)) &
                                 (df_inflow.index < last_day_train+pd.Timedelta(days = 1))]
history_weather_df = df_weather.loc[(df_weather.index > last_day_train-pd.Timedelta(days = n_days_train)) &
                                 (df_weather.index < last_day_train+pd.Timedelta(days = 1))]
true_df = df_inflow.loc[(df_inflow.index >= last_day_train+pd.Timedelta(days = 1)) &
                                  (df_inflow.index<=last_day_train+pd.Timedelta(days = n_days_test+1))]
expected_weather_df = df_weather.loc[(df_weather.index >= last_day_train+pd.Timedelta(days = 1)) &
                                  (df_weather.index<=last_day_train+pd.Timedelta(days = n_days_test+1))]

#Main loop
for dma in forecasting_dmas:    # for each dma

    if 'lstm' in forecasting_methods:
         #y_pred will be an array storing the forecast obtained from LSTM
         y_pred = lstm(df_train = history_inflow_df.loc[:,[dma]], 
                       window=WINDOW, 
                       n_days_test=n_days_test)
         
         x = true_df.index # it will store the indices of the range to forecast
         df_forecasts_lstm.loc[x,dma] = y_pred[:,0]
         y_true =true_df.loc[x,dma].values # true values of the range to forecast
         
         lstm_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
         global_rmse.loc['lstm',dma] = lstm_rmse.loc[0,dma]
         if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
             lstm_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
             global_battle_metrics.loc['lstm',dma] = lstm_battle_metrics.loc[0,dma]
     
    if 'naive_avg' in forecasting_methods:
        y_pred = naive(df_train=history_inflow_df.loc[:,[dma]],
                       test_length=24*n_days_test,
                       forecast_type='avg',
                       pattern_type='day_by_day')   # day_by_day or weekday_vs_weekend
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_naive_avg.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        naive_avg_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['naive_avg',dma] = naive_avg_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            naive_avg_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['naive_avg',dma] = naive_avg_battle_metrics.loc[0,dma]

    if 'naive_median' in forecasting_methods:
        y_pred = naive(df_train=history_inflow_df.loc[:,[dma]],
                       test_length=24*n_days_test,
                       forecast_type='median',
                       pattern_type='day_by_day')   # day_by_day or weekday_vs_weekend
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_naive_median.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        naive_median_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['naive_median',dma] = naive_median_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            naive_median_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['naive_median',dma] = naive_median_battle_metrics.loc[0,dma]

    if 'naive_ewma' in forecasting_methods:
        y_pred = naive(df_train=history_inflow_df.loc[:,[dma]],
                       test_length=24*n_days_test,
                       forecast_type='ewma',
                       pattern_type='day_by_day')   # day_by_day or weekday_vs_weekend
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_naive_ewma.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        naive_ewma_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['naive_ewma',dma] = naive_ewma_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            naive_ewma_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['naive_ewma',dma] = naive_ewma_battle_metrics.loc[0,dma]

    if 'xgboost_naive' in forecasting_methods:
        y_pred = xgboost_naive(history_inflow_df=history_inflow_df.loc[:,[dma]], 
                                history_weather_df = history_weather_df,
                                expected_weather_df = expected_weather_df, 
                                df_holidays = df_holidays, 
                                test_length=24*n_days_test)
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_xgboost_naive.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        xgboost_naive_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['xgboost_naive',dma] = xgboost_naive_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            xgboost_naive_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['xgboost_naive',dma] = xgboost_naive_battle_metrics.loc[0,dma]

    if 'svr' in forecasting_methods:
        y_pred = svr(history_inflow_df=history_inflow_df.loc[:,[dma]], 
                     test_length=24*n_days_test,
                     df_holidays = df_holidays,
                     D=5)
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_svr.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        svr_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['svr',dma] = svr_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            svr_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['svr',dma] = svr_battle_metrics.loc[0,dma]

    if 'quevedo' in forecasting_methods:
        y_pred = quevedo(history_inflow_df=history_inflow_df.loc[:,[dma]], 
                     test_length=24*n_days_test,
                     df_holidays = df_holidays,
                     pattern_type='day_by_day')   # day_by_day or weekday_vs_weekend
        
        x = true_df.index # it will store the indices of the range to forecast
        df_forecasts_quevedo.loc[x,dma] = y_pred
        y_true =true_df.loc[x,dma].values # true values of the range to forecast

        quevedo_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)
        global_rmse.loc['quevedo',dma] = svr_rmse.loc[0,dma]

        if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
            quevedo_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
            global_battle_metrics.loc['quevedo',dma] = quevedo_battle_metrics.loc[0,dma]

# Create a list to store traces
traces = []
for dma in df_inflow[forecasting_dmas].columns:    #real measurements
    trace = go.Scatter(x=df_inflow.index, y=df_inflow[dma], mode='lines', name=f'Real - {dma}',opacity=0.5, line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
    traces.append(trace)

#Plots
if 'lstm' in forecasting_methods:
    for dma in df_forecasts_lstm.columns:    #forecasts_lstm
        trace = go.Scatter(x=df_forecasts_lstm.index, y=df_forecasts_lstm[dma], mode='markers', name=f'LSTM - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)

if 'naive_avg' in forecasting_methods:
    for dma in df_forecasts_naive_avg.columns:    #forecasts_naive_avg
        trace = go.Scatter(x=df_forecasts_naive_avg.index, y=df_forecasts_naive_avg[dma], mode='markers', name=f'Naive Avg - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace) 

if 'naive_median' in forecasting_methods:
    for dma in df_forecasts_naive_median.columns:    #forecasts_naive_median
        trace = go.Scatter(x=df_forecasts_naive_median.index, y=df_forecasts_naive_median[dma], mode='markers', name=f'Naive Median - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace) 

if 'naive_ewma' in forecasting_methods:
    for dma in df_forecasts_naive_ewma.columns:    #forecasts_naive_ewma
        trace = go.Scatter(x=df_forecasts_naive_ewma.index, y=df_forecasts_naive_ewma[dma], mode='markers', name=f'Naive Ewma - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)         

if 'xgboost_naive' in forecasting_methods:
    for dma in df_forecasts_xgboost_naive.columns:    #forecasts_xgboost_naive
        trace = go.Scatter(x=df_forecasts_xgboost_naive.index, y=df_forecasts_xgboost_naive[dma], mode='markers', name=f'Xgboost Naive - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)     

if 'svr' in forecasting_methods:
    for dma in df_forecasts_svr.columns:    #forecasts_xgboost_naive
        trace = go.Scatter(x=df_forecasts_svr.index, y=df_forecasts_svr[dma], mode='markers', name=f'SVR - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace) 

if 'quevedo' in forecasting_methods:
    for dma in df_forecasts_quevedo.columns:    #forecasts_xgboost_naive
        trace = go.Scatter(x=df_forecasts_quevedo.index, y=df_forecasts_quevedo[dma], mode='markers', name=f'Quevedo - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace) 


fig = go.Figure(data=traces)

fig.show()

#Print errors for each model
if 'lstm' in forecasting_methods:
    print('RMSE for the LSTM model (for all selected DMAs) \n\n',lstm_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the LSTM model (for all selected DMAs)\n\n', lstm_battle_metrics)
        print('Overall battle metrics = ', lstm_battle_metrics.sum(axis =1).loc[0])

if 'naive_avg' in forecasting_methods:
    print('RMSE for the Naive Avg model (for all selected DMAs) \n\n',naive_avg_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the Naive Avg model (for all selected DMAs)\n\n', naive_avg_battle_metrics)
        print('Overall battle metrics = ', naive_avg_battle_metrics.sum(axis =1).loc[0])

if 'naive_median' in forecasting_methods:
    print('RMSE for the Naive Median model (for all selected DMAs) \n\n',naive_median_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the Naive Median model (for all selected DMAs)\n\n', naive_median_battle_metrics)
        print('Overall battle metrics = ', naive_median_battle_metrics.sum(axis =1).loc[0])

if 'naive_ewma' in forecasting_methods:
    print('RMSE for the Naive Avg model (for all selected DMAs) \n\n',naive_ewma_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the Naive Ewma model (for all selected DMAs)\n\n', naive_ewma_battle_metrics)
        print('Overall battle metrics = ', naive_ewma_battle_metrics.sum(axis =1).loc[0])

if 'xgboost_naive' in forecasting_methods:
    print('RMSE for the Xgboost Naive model (for all selected DMAs) \n\n',xgboost_naive_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the Xgboost Naive model (for all selected DMAs)\n\n', xgboost_naive_battle_metrics)
        print('Overall battle metrics = ', xgboost_naive_battle_metrics.sum(axis =1).loc[0])

if 'svr' in forecasting_methods:
    print('RMSE for the SVR model (for all selected DMAs) \n\n',svr_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the SVR model (for all selected DMAs)\n\n', svr_battle_metrics)
        print('Overall battle metrics = ', svr_battle_metrics.sum(axis =1).loc[0])

if 'quevedo' in forecasting_methods:
    print('RMSE for the Quevedo model (for all selected DMAs) \n\n',quevedo_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the Quevedo model (for all selected DMAs)\n\n', quevedo_battle_metrics)
        print('Overall battle metrics = ', quevedo_battle_metrics.sum(axis =1).loc[0])



#Print global errors
print('Global RMSE: \n\n',global_rmse)
print('\n\n')
if n_days_test==7:
    global_battle_metrics['sum'] = global_battle_metrics.sum(axis=1).values
    print('Global BattleMetrics \n\n',global_battle_metrics)
