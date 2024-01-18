import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from datetime import  datetime, timedelta
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df_inflow = pd.read_csv('inflow_completed_in_R.csv')
df_holidays = pd.read_csv('Holidays.txt',parse_dates = True)    

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
df_inflow['datetime'] = pd.to_datetime(df_inflow['datetime'],format='%d/%m/%Y %H:%M')
df_inflow.set_index('datetime', inplace = True) #to make the timestamps become row indices


#######################    parameters
last_day_train = '2022-07-17' #last day considered for our training/validation data
n_days_train   = 120           # length of our train/validation data (counting from the last_day_train into the past)
n_days_test    = 7            # lenght of our test data (counting from the next daty to the last_day_train



# comment out the models you don't want to run
forecasting_methods = [
                    #'naive_avg',
                    #'naive_median',
                    #'naive_ewma',
                    #'XGBOOST_naive',
                    'lstm',
                    ]

# comment out the variables dmas you don't want to run
forecasting_dmas = [
                    'dma_A',
                    'dma_B',
                    #'dma_C',
                    #'dma_D',
                    #'dma_E',
                    # 'dma_F',
                    # 'dma_G',
                    # 'dma_H',
                    # 'dma_I',
                    # 'dma_J'
                    ]

#### Parameters for LSTM ####
WINDOW     = 10
EPOCHS     = 200 #number of complete passes 
PATIENCE   = 10  #number of epochs with no improvement after which training will be stopped
BATCH_SIZE = 32  #number of training samples to work through before the model’s internal parameters are updated
#############################




###########################

#just to convert the type to datetime
last_day_train = pd.to_datetime(last_day_train)

#auxiliary functions to train LSTM model
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

def split_data(df, val_split=0.15):
    #test data = days to forecast * 24
    n_test = n_days_to_forecast*24
    n_train_val = len(df)-n_test
    n = int(n_train_val * val_split)
    train, val, test = df[:-n], df[-n:n_test], df[n_test:]
    return train,val


class Standardize:
    def __init__(self, df, n_days_test, split=0.15 ):
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
    

def lstm(df_train = None, window = WINDOW, n_days_test = n_days_test): 

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

range_to_forecast = pd.date_range(pd.to_datetime(last_day_train)+pd.Timedelta(days=1),freq='H',periods=n_days_test*24)


#Empty DF's to receive forecasts
if 'lstm' in forecasting_methods: df_forecasts_lstm = pd.DataFrame(index = range_to_forecast, columns = forecasting_dmas)

#Empty df's to receive error metric results
if 'lstm' in forecasting_methods: lstm_rmse = pd.DataFrame(columns=forecasting_dmas)

#Empty df's to receive battle metric results
if 'lstm' in forecasting_methods: lstm_battle_metrics= pd.DataFrame(columns=forecasting_dmas)

# To compute Battle's metrics
# WARNING, for this to work, forecast range should be an entire week

def MeanError1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
    
    SumError = np.sum(Allerrors)
    AvgError = SumError / len(Allerrors)
    return AvgError

def MaxError1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
    
    MaxError = np.max(Allerrors)
    return MaxError

def MeanErrorExcluding1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24, len(y_pred)):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
        
    SumError = np.sum(Allerrors)
    AvgError = SumError / len(Allerrors)
    return AvgError

def BattleMetrics_per_dma(y_pred, y_true):
    return MeanError1stday(y_pred, y_true)+MaxError1stday(y_pred, y_true)+ MeanErrorExcluding1stday(y_pred, y_true)

#to slice dataframe according to the time range defined for the training data
history_inflow_df = df_inflow.loc[(df_inflow.index > last_day_train-pd.Timedelta(days = n_days_train)) &
                                 (df_inflow.index < last_day_train+pd.Timedelta(days = 1))]

true_df = df_inflow.loc[(df_inflow.index >= last_day_train+pd.Timedelta(days = 1)) &
                                  (df_inflow.index<=last_day_train+pd.Timedelta(days = n_days_test+1))]


for dma in forecasting_dmas:    # for each dma

    if 'lstm' in forecasting_methods:
         #y_pred will be an array storing the forecast obtained from LSTM
         y_pred = lstm(df_train = history_inflow_df.loc[:,[dma]], window=WINDOW, n_days_test=n_days_test)
         x = true_df.index # it will store the indices of the range to forecast
         df_forecasts_lstm.loc[x,dma] = y_pred[:,0]
         y_true =true_df.loc[x,dma].values # true values of the range to forecast
         
         lstm_rmse.loc[0,dma] = mean_squared_error(y_true, y_pred,squared=False)

         if n_days_test == 7: #because only in this case it makes sense to compute battle's metrics
             lstm_battle_metrics.loc[0,dma]= BattleMetrics_per_dma(y_pred, y_true)
             

# Create a list to store traces
traces = []
for dma in df_inflow[forecasting_dmas].columns:    #real measurements
    trace = go.Scatter(x=df_inflow.index, y=df_inflow[dma], mode='lines', name=f'Real - {dma}',opacity=0.5, line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
    traces.append(trace)


if 'lstm' in forecasting_methods:
    for dma in df_forecasts_lstm.columns:    #forecasts_lstm
        trace = go.Scatter(x=df_forecasts_lstm.index, y=df_forecasts_lstm[dma], mode='markers', name=f'LSTM - {dma}', line=dict(color=dict_colors[dma], width=1.5, dash='solid'))
        traces.append(trace)

fig = go.Figure(data=traces)

fig.show()


if 'lstm' in forecasting_methods:
    print('RMSE for the LSTM model (for all selected DMAs) \n\n',lstm_rmse)
    print('\n\n')
    if n_days_test==7:
        print('Battle metrics for the LSTM model (for all selected DMAs)\n\n', lstm_battle_metrics)
        print('Overall battle metrics = ', lstm_battle_metrics.sum(axis =1).loc[0])
