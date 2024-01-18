import pandas as pd
import numpy as np
import datetime
from sklearn import svm




#### SVR1
def svr(history_inflow_df=None, test_length=24*7, df_holidays=None,D=5):

    holiday = df_holidays
    holiday.columns = ['date']

    history_inflow_df['date'] = history_inflow_df.index
    history_inflow_df.columns = ['flow','date']

    #Aqui desenvolve-se tudo
    df=history_inflow_df
    # criar duas novas colunas
    df['Date'] = pd.to_datetime(df['date']).dt.date
    df['Time'] = pd.to_datetime(df['date']).dt.time
    # print('New Dataset (split date and time):')
    # print(df)
    df=df.reset_index()

    del df['date']
    df = df.rename({'Date': 'date', 'Time': 'time', 'value':'flow'}, axis='columns')
    df=df[['date','time','flow']]
    # print('New Dataset (split date and time) 2:')
    # print(df)
    
    # holiday['date'] = pd.to_datetime(holiday['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # # to extract all unique values (time) present in dataframe
    time_unique_val=df.time.unique()
    time_unique_ind=np.arange(24)
    
    #in order to have a mapping between the time of day and its index
    time_unique=pd.DataFrame({'time':time_unique_val, 'time_unique_ind':time_unique_ind})
    # print('Unique timesteps')
    # print(time_unique)
    
    #creates a column with the time index
    df['time_ind'] = df['time'].map(time_unique.set_index('time')['time_unique_ind'])
    # print('New Dataset (split date and time with unique timestep):')
    # print(df)
    
    #extra column indicating day of week
    #0: mon, 1:tue, ..., 5:sat, 6:sun
    df['dayofweek'] = pd.to_datetime(df['date'],dayfirst=True)
    df['dayofweek'] = df['dayofweek'].dt.dayofweek
    
    # if day is a holiday, then dayofweek is -1
    df.loc[df.date.isin(holiday.date), 'dayofweek'] = -1
    # print('New Dataset (split date, with unique timestep, dayofweek):')
    # print(df)
    
    # in order to construct a new dataframe with D lagged values
    df_lagged = df.copy()
    for i in range(1, D+1):
        shifted = df['flow'].shift(i)
        df_lagged = pd.concat((df_lagged, shifted), axis=1)
        
    # just to label the columns of the resulting dataframe
    lagged_cols=["n-"+ str(x) for x in range(1,D+1)]
    colnames= ["date","time","flow","time_ind","dayofweek"]+lagged_cols
    df_lagged.columns=colnames
    # print('Lagged df (lag of 5 measurements)')
    # print(df_lagged)
    
    # in order to drop the rows corresponding do the first day
    #to get rid of the NaN values
    df_lagged = df_lagged.iloc[D:]
    # print('Lagged df without first day')
    # print(df_lagged)
    
    #df_weekdays: contains data related to weekdays 
    #df_sat: contains data related to saturdays
    #df_sun: contains data related to sundays and holidays
    df_weekdays = df_lagged[(df_lagged['dayofweek']>=0) & (df_lagged['dayofweek']<=4)]
    # print("Weekdays df", df_weekdays)
    df_sat = df_lagged[df_lagged['dayofweek']==5]
    # print("Saturdays df", df_sat)
    df_sun = df_lagged[(df_lagged['dayofweek']==-1) | (df_lagged['dayofweek']==6)]
    # print("Sunday df", df_sun)
    

    #Começa o processo de regressão


    # for testing, we will use min_date_train to max_date_train for constructing regressors
    regressors_weekdays,std_weekdays=creates_regressors_and_std(df_weekdays)
    # print("Regressor Weekdays", regressors_weekdays)
    # print(len(regressors_weekdays))
    # print("Standard deviation Weekdays", std_weekdays)
    regressors_saturdays,std_saturdays=creates_regressors_and_std(df_sat)
    # print("Regressor Saturdays", regressors_saturdays)
    # print(len(regressors_saturdays))
    # print("Standard deviation Saturdays", std_saturdays)
    regressors_sundays, std_sundays=creates_regressors_and_std(df_sun)
    # print("Regressor Sundays", regressors_sundays)
    # print(len(regressors_sundays))
    # print("Standard deviation Sundays", std_sundays)
    
    regressors = [regressors_weekdays, regressors_saturdays, regressors_sundays]

    # print('df_lagged')
    # print(df_lagged)


    lagged_cols=["n-"+ str(x) for x in range(1,D+1)]
    colnames= ["date","time","flow","time_ind","dayofweek"]+lagged_cols
    df2predict = pd.DataFrame(columns=colnames,
                              index=range(df_lagged.index[-1]+1,df_lagged.index[-1]+1+test_length),
                              )
    # print('df2predict')
    # print(df2predict)


    date_1_aux = datetime.datetime.combine(df_lagged.iloc[-1,0],df_lagged.iloc[-1,1])
    date_2_aux = datetime.datetime.combine(df_lagged.iloc[-2,0],df_lagged.iloc[-2,1])
    spacing = (date_1_aux - date_2_aux)
    # print('Spacing')
    # print(spacing)

    start_date = datetime.datetime.combine(df_lagged.iloc[-1,0],df_lagged.iloc[-1,1])+spacing
    dates2predict = pd.date_range(start=start_date, 
                                  periods=test_length,
                                  freq=spacing)
    # print('dates2predict')
    # print(dates2predict)

    df2predict['date'] = dates2predict.date
    df2predict['time'] = dates2predict.time

    # print('df2predict')
    # print(df2predict)

    df2predict['time_ind'] = df2predict['time'].map(time_unique.set_index('time')['time_unique_ind'])
    # print('df2predict')
    # print(df2predict)


    df2predict['dayofweek'] = pd.to_datetime(df2predict['date'],dayfirst=True)
    df2predict['dayofweek'] = df2predict['dayofweek'].dt.dayofweek
    # print('df2predict')
    # print(df2predict)

    # if day is a holiday, then dayofweek is -1
    df2predict.loc[df2predict.date.isin(holiday.date), 'dayofweek'] = -1
    # print('df2predict')
    # print(df2predict)

    df2predict.iloc[0,5:] = df_lagged.iloc[-D:,2].iloc[::-1]
    # print('df2predict')
    # print(df2predict)

    y_pred, df2predict =predict_flow(df2predict, regressors)

    # print('df2predict')
    # print(df2predict)

    # print('y_pred')
    # print(y_pred)

    return y_pred

def predict_flow(df,regressors):
    regressors_weekdays, regressors_saturdays, regressors_sundays = regressors
    
    predictions=[]
    for i in range(len(df)):
        # print('Forecasting step: ',df.index[i])
        if ((df.iloc[i,4]>=0) and (df.iloc[i,4] <=4)):
            model=regressors_weekdays[df.iloc[i,3]]
        elif (df.iloc[i,4]==5): # i.e. saturdays
            model=regressors_saturdays[df.iloc[i,3]]
        else: #i.e., sundays and holidays
            model=regressors_sundays[df.iloc[i,3]]
        
        x2model = df.iloc[i,5:].values.reshape(1, -1)
        # print(x2model)

        y = model.predict(x2model)[0]
        # print(y)

        df.iloc[i,2] = y
        if i < len(df) - 1:
            df.iloc[i+1,5] = y
            df.iloc[i+1,6:] = df.iloc[i,5:-1]
        # print(df)
        predictions.append(y)


    # print(predictions)
    return predictions,df

def svr_model(data,epsilon):
    y=data['flow'].copy()
    X=data.iloc[:,5:] #contains only the D previous values
    # print(X,len(X))


    # fig1 = plt.figure()
    # for i in range(len(X)):
    #     plt.plot([-5,-4,-3,-2,-1],X.iloc[i,:], 'o-',label=X.index[i])
    # plt.legend()


    #70% train, 30% test
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    X_train = X
    y_train = y
    # print(X_train,len(X_train))


    # ########### necesita de Xtest e permite otimizar SVR
    # if (activate_grid_search==1):
    #     params=grid_search(data)
    #     svr_model = svm.SVR(C=params['C'], epsilon=epsilon, gamma=params['gamma'], kernel=params['kernel'])
    # else:
    svr_model = svm.SVR(C=10.0, epsilon=epsilon)
    svr_model.fit(X_train, y_train)
    return svr_model

def create_data_for_model(data, time_ind):
    sel_data=data[data['time_ind']==time_ind]
    return sel_data

def creates_regressors_and_std(df):
    data1=df
    regressors = []
    std=[]
    
    for i in range(24):
        data=create_data_for_model(data1, i) # vou chamar de selecionar as datas
        std.append(data['flow'].std())

        epsilon=data['flow'].std()*0.5 #the tube
        # print(data)
        regressors.append(svr_model(data, epsilon))
    return regressors,std
