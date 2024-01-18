import pandas as pd
import numpy as np



def naive(df_train=None,test_length=24*7,forecast_type='avg',pattern_type='day_by_day'): #média, mediana, ewma

    dma_name = df_train.columns[0]

    #history_type can be "day_by_day" or "weekday_vs_weekend"

    df_train = df_train.copy()

    df_train = df_train.dropna()

    if pattern_type == 'day_by_day':
        df_train['day_of_week'] = df_train.index.day_name()
    elif pattern_type == 'weekday_vs_weekend':
        df_train['day_of_week'] = df_train.index.weekday

    df_train['hour'] = df_train.index.hour
    df_train['minute'] = df_train.index.minute


    frequency = int(np.diff(df_train.index).min()/ np.timedelta64(1, 's'))#in seconds

    forecasting_timestamps = pd.date_range(start=df_train.index[-1]+pd.to_timedelta(str(frequency)+'s'),freq=str(frequency)+'s', periods=test_length)

    predictions = pd.DataFrame(index=forecasting_timestamps,columns=[dma_name])

    for index, row in predictions.iterrows():
        hour = index.hour
        minute = index.minute

        #Cria bloco de dados, que depende se o tipo de historico é "day_by_day" ou "weekday_vs_weekend"
        if pattern_type == 'day_by_day':
            day_of_week = index.day_name()
            block = df_train[(df_train['day_of_week']==day_of_week) & 
                             (df_train['hour']==hour) & 
                             (df_train['minute']==minute)]

        elif pattern_type == 'weekday_vs_weekend':
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

    return predictions.iloc[:,0].values   #return just the values
