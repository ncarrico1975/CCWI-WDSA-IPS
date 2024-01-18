import datetime
import pandas as pd
import numpy as np
import xgboost as xgb

def xgboost_naive(history_inflow_df=None, history_weather_df = None, expected_weather_df = None, df_holidays = None, test_length=24*7):

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

    
    return y_pred_df.iloc[:,0].values
