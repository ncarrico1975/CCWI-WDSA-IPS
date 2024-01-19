import datetime
import pandas as pd
import numpy as np
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def measure_rmse(test, predicted):
	return sqrt(mean_squared_error(test, predicted))

def quevedo(history_inflow_df=None, test_length=24*7, df_holidays=None,pattern_type='day_by_day'):

    history_inflow_df = history_inflow_df.copy()
    df_holidays = df_holidays.copy()

    
    history_inflow_df['date'] = history_inflow_df.index
    history_inflow_df.columns = ['value','date']
    train_simulation = history_inflow_df

    holiday = df_holidays
    holiday.columns = ['date']


    n_measures_in_day = 24



    #create array with dates to forecast - Quevedo is runned individually once per day
    freq = 24/n_measures_in_day*60
    freq_str = str(int(freq))+'min'
    first_date = train_simulation['date'].iloc[-1]+datetime.timedelta(minutes=freq)

    datelist = pd.date_range(start=first_date, periods=test_length,freq=freq_str)
    # print(datelist)



    #lista de dias individuais que o quevedo vai receber 
    unique_days = []
    for x in datelist.date:
        # check if exists in unique_list or not
        if x not in unique_days:
            unique_days.append(x)
    # print(unique_days)



    #for each individual day use quevedo 
    for x in range(len(unique_days)):#para cada um destes dias

        date1 = np.datetime64(unique_days[x]) + np.timedelta64(0, 's')#vou buscar o dia
        train_simulation_temp = train_simulation[train_simulation['date'] < date1]  #DF temporaria apenas com historico anterior ao dia a prever

        new_dates1, new_values1 = main_quevedo(train_simulation_temp, date1,holiday,n_measures_in_day,pattern_type)  #chamo o quevedo para o historico temporario / recebo 1 dia inteiro
        tempDf1 = pd.DataFrame(columns=['date', 'value'])   #DF quevedo
        tempDf1['date'] = new_dates1
        tempDf1['value'] = new_values1

        # print('Quevedo results ')
        # print(tempDf1)

        # print(new_values1)
        tempDf2 = tempDf1[(tempDf1['date'] > train_simulation['date'].iloc[-1]) & (tempDf1['date'] <= datelist[-1])] #Filtro apenas as medições que me interessam (que nao existem no train simulation)

        # print('Quevedo results filtered')
        # print(tempDf2)



        train_simulation = pd.concat([train_simulation, tempDf2[tempDf2['date'] > train_simulation['date'].iloc[-1]]], ignore_index=True)   

        # print('History updated')
        # print(train_simulation)

    result_df = train_simulation['value'].iloc[-test_length:].values
    return result_df

class ARIMA_QV:
    """Scikit-learn like interface for Holt-Winters method."""

    def __init__(self, a1=0.5, a2=0.1, a3=0.5, a4=0.5):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4


    def fit(self, series):
        # note that unlike scikit-learn's fit method, it doesn't learn
        # the optimal model paramters, alpha, beta, gamma instead it takes
        # whatever the value the user specified the produces the predicted time
        # series, this of course can be changed.

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4

        self.series = series

        ####

        b1 = a1 - (2*math.cos(2*math.pi/7)+1)
        b2 = a2 - (2*math.cos(2*math.pi/7)+1)*a1 + 2 * math.cos(2*math.pi/7)+1
        b3 = a3 - (2*math.cos(2*math.pi/7)+1)*a2 + (2 * math.cos(2*math.pi/7)+1)*a1-1
        b4 = a4 - (2*math.cos(2*math.pi/7)+1)*a3 + (2 * math.cos(2*math.pi/7)+1)*a2-a1
        b5 = -(2*math.cos(2*math.pi/7)+1)*a4 + (2*math.cos(2*math.pi/7)+1)*a3-a2
        b6 = (2*math.cos(2*math.pi/7)+1)*a4-a3
        b7 = -a4

        predictions = []
        [predictions.append(None) for i in range(7)]



        for i in range(7, len(series)):

            f = - b1 * series[i-1] - b2 * series[i-2] -b3 * series[i-3] - b4 * series[i-4] - b5 * series[i-5] - b6 * series[i-6] - b7 * series[i-7]
            predictions.append(f)

            # print()
        self.predictions_ = predictions
        return self



    def predict(self, n_preds=1):


        series = self.series
        predictions = self.predictions_

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4

        ####

        b1 = a1 - (2*math.cos(2*math.pi/7)+1)
        b2 = a2 - (2*math.cos(2*math.pi/7)+1)*a1 + 2 * math.cos(2*math.pi/7)+1
        b3 = a3 - (2*math.cos(2*math.pi/7)+1)*a2 + (2 * math.cos(2*math.pi/7)+1)*a1-1
        b4 = a4 - (2*math.cos(2*math.pi/7)+1)*a3 + (2 * math.cos(2*math.pi/7)+1)*a2-a1
        b5 = -(2*math.cos(2*math.pi/7)+1)*a4 + (2*math.cos(2*math.pi/7)+1)*a3-a2
        b6 = (2*math.cos(2*math.pi/7)+1)*a4-a3
        b7 = -a4

        i = len(series)

        f = - b1 * series[i-1] - b2 * series[i-2] -b3 * series[i-3] - b4 * series[i-4] - b5 * series[i-5] - b6 * series[i-6] - b7 * series[i-7]



        predictions.append(f)

        return predictions

def timeseries_cv_score(params, series):
    a1, a2, a3, a4 = params

    model = ARIMA_QV(a1, a2, a3, a4)
    model.fit(series)
    predictions = model.predictions_

    rmse = measure_rmse(series[7:], predictions[7:])
    return rmse

def main_quevedo(hist, date1,holiday,n_measures_in_day,pattern_type):


    values_h = hist['value']
    dates_h = hist['date']
    spacing = (dates_h.iloc[1] - dates_h.iloc[0])

    ###Estimar o periodo de um dia, ou seja, o nr de medições que ocorrem num dia
    ###Como o 1º dia do historico pode vir incompleto, vamos buscar a contagem do 2 dia
    hist.index = hist['date']
    hist.index = pd.to_datetime(hist.index)
    history_count_day = hist.groupby(hist.index.date).count()

    del hist['date']

    ### Quantas medições ocorrem numa hora
    N = 24 / n_measures_in_day

    #agregado diário
    history_sum_day = hist.groupby(hist.index.date).sum()

    #correção do N
    for idx, day in enumerate(history_sum_day.value):
        history_sum_day.iloc[idx] = history_sum_day.iloc[idx] * N

    # print('History_sum_day \n', history_sum_day)
    #############

    holiday = holiday['date']
    # holiday = [np.datetime64(x) for x in holiday]

    hol = 0
    last_day = date1

    if last_day in holiday:
        hol = 1

    if hol == 0:
        x = [0.6, 0.5, 0.6, 0.5]
        data = history_sum_day[1:].values
        data = [item for sublist in data for item in sublist]

        opt = minimize(timeseries_cv_score, x0=x,
                       args=(data),
                       method='TNC')

        # print('original parameters: {}'.format(str(x)))
        # print('best parameters: {}'.format(str(opt.x)))

        # print('end')

        a1, a2, a3, a4 = opt.x
        model = ARIMA_QV(a1, a2, a3, a4)
        model.fit(data)

        predictions = model.predict(n_preds=1)

    else:
        # print('É feriado')

        #get all the sundays
        sundays = history_sum_day.copy()
        sundays = sundays.reset_index()
        sundays['index'] = pd.to_datetime(sundays['index'])
        sundays['weekday'] = sundays['index'].dt.dayofweek
        # print(sundays)
        sundays = sundays.loc[sundays['weekday'] == 6]
        # print(sundays)
        sundays = sundays.reset_index(drop=True)
        del sundays['weekday']
        vals = sundays['value'].values

        model = SimpleExpSmoothing(vals)
        model_fit = model.fit()
        predictions = model_fit.predict(len(vals), len(vals))

        # print(predictions)

    # plt.figure(1)
    # plt.plot(history_sum_day.index, history_sum_day.values,'b.-', label='Histórico')
    # plt.plot(history_sum_day.index[:-1], predictions[:-1], 'g.-', label='Fit')
    # plt.plot(history_sum_day.index[-1], predictions[-1], 'r.-', label='Forcasted')
    # plt.legend()
    # rmse = history_sum_day.values[-1]-predictions[-1]
    # plt.title(label='ARIMA for daily forecast // SE= ' + str(round(rmse[0], 2)))

    #######Pattern
    hist['date'] = hist.index
    hist['weekday'] = hist['date'].apply(lambda x: x.weekday())#monday 0, sunday 6
    # print('hist with weekday \n', hist)









    ####Aplicar o padrão ao novo dia

    new_day = predictions[-1]
    new_day_wd = date1.astype(datetime.datetime).weekday()#monday 0 sunday 6
    # print(new_day_wd,hol)
    pat = None
    ################
    # plt.figure(2)

    if pattern_type == 'weekday_vs_weekend':

        if new_day_wd < 5 and hol == 0: # if it is a weekday
            ###weekday
            weekdays_only = hist[hist['weekday'] < 5]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Weekdays')

        elif new_day_wd == 5 and hol == 0:
            ###saturday
            saturday_only = hist[hist['weekday'] == 5]
            del saturday_only['weekday']
            del saturday_only['date']
            # print('saturday only \n', saturday_only)
            saturday_only_patt = saturday_only.groupby(saturday_only.index.time).mean()
            # print('saturday only patt mean \n', saturday_only_patt)
            sum_sat_patt = saturday_only_patt.sum()
            # print(sum_sat_patt)
            for i in range(len(saturday_only_patt)):
                saturday_only_patt.iloc[i] = saturday_only_patt.iloc[i] / sum_sat_patt
            # print('saturday only patt \n', saturday_only_patt)

            pat = saturday_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(saturday_only_patt, label='Saturdays')

        elif new_day_wd == 6 or hol == 1:
            ###sunday
            sunday_only = hist[hist['weekday'] == 6]
            del sunday_only['weekday']
            del sunday_only['date']
            # print('sunday only \n', sunday_only)
            sunday_only_patt = sunday_only.groupby(sunday_only.index.time).mean()
            # print('sunday only patt mean \n', sunday_only_patt)
            sum_sun_patt = sunday_only_patt.sum()
            # print(sum_sun_patt)
            for i in range(len(sunday_only_patt)):
                sunday_only_patt.iloc[i] = sunday_only_patt.iloc[i] / sum_sun_patt
            # print('saturday only patt \n', sunday_only_patt)

            pat = sunday_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(sunday_only_patt, label='Sundays')

    elif pattern_type == 'day_by_day':
        if new_day_wd == 0 and hol == 0:
            ###Monday
            weekdays_only = hist[hist['weekday'] == 0]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Monday')

        elif new_day_wd == 1 and hol == 0:
            ###Thuesday
            weekdays_only = hist[hist['weekday'] == 1]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Thuesday')

        elif new_day_wd == 2 and hol == 0:
            ###Wednesday
            weekdays_only = hist[hist['weekday'] == 2]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Wednesday')

        elif new_day_wd == 3 and hol == 0:
            ###Thursday
            weekdays_only = hist[hist['weekday'] == 3]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Thursday')

        elif new_day_wd == 4 and hol == 0:
            ###Friday
            weekdays_only = hist[hist['weekday'] == 4]
            del weekdays_only['weekday']
            del weekdays_only['date']
            # print('weekday only \n', weekdays_only)
            weekdays_only_patt = weekdays_only.groupby(weekdays_only.index.time).mean()
            # print('weekday only patt mean \n', weekdays_only_patt)
            sum_wd_patt = weekdays_only_patt.sum()
            # print(sum_wd_patt)
            for i in range(len(weekdays_only_patt)):
                weekdays_only_patt.iloc[i] = weekdays_only_patt.iloc[i] / sum_wd_patt
            # print('weekday only patt \n', weekdays_only_patt)


            pat = weekdays_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(weekdays_only_patt, label='Friday')

        elif new_day_wd == 5 and hol == 0:
            ###saturday
            saturday_only = hist[hist['weekday'] == 5]
            del saturday_only['weekday']
            del saturday_only['date']
            # print('saturday only \n', saturday_only)
            saturday_only_patt = saturday_only.groupby(saturday_only.index.time).mean()
            # print('saturday only patt mean \n', saturday_only_patt)
            sum_sat_patt = saturday_only_patt.sum()
            # print(sum_sat_patt)
            for i in range(len(saturday_only_patt)):
                saturday_only_patt.iloc[i] = saturday_only_patt.iloc[i] / sum_sat_patt
            # print('saturday only patt \n', saturday_only_patt)

            pat = saturday_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(saturday_only_patt, label='Saturdays')

        elif new_day_wd == 6 or hol == 1:
            ###sunday
            sunday_only = hist[hist['weekday'] == 6]
            del sunday_only['weekday']
            del sunday_only['date']
            # print('sunday only \n', sunday_only)
            sunday_only_patt = sunday_only.groupby(sunday_only.index.time).mean()
            # print('sunday only patt mean \n', sunday_only_patt)
            sum_sun_patt = sunday_only_patt.sum()
            # print(sum_sun_patt)
            for i in range(len(sunday_only_patt)):
                sunday_only_patt.iloc[i] = sunday_only_patt.iloc[i] / sum_sun_patt
            # print('saturday only patt \n', sunday_only_patt)

            pat = sunday_only_patt
            for i in range(len(pat)):
                pat.iloc[i] = pat.iloc[i] * new_day / N
            
            # plt.plot(sunday_only_patt, label='Sundays')

    else:
        print('Erro nos padrões')



    pat_values = pat.values
    pat_values = [item for sublist in pat_values for item in sublist]

    pat['date'] = pat.index
    time1 = pat['date'].values
    time1 = time1[0]
    new_date = datetime.datetime.combine(date1.astype(datetime.datetime), time1)

    pat_dates = pd.date_range(
        start=new_date, periods=24/N, freq=spacing)

    new_dates = [np.datetime64(x) for x in dates_h]


    return pat_dates, pat_values
