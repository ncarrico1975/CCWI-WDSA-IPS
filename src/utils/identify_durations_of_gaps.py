import math

def durations_gaps(df,variable,durations_of_short_failure):
    long_duration_failure = {}
    short_duration_failure = {}

    counter = 0
    dates_array=[]
    for index, value in df[variable].items(): 
        if math.isnan(value):
            counter+=1
            dates_array.append(index)
        elif math.isnan(value)==False:
            if counter > durations_of_short_failure:
                if counter!=0 and dates_array!=[]:
                    long_duration_failure[f'{dates_array[0]} >>> {dates_array[-1]}'] = (counter, (df[variable].index[-1]-index).days)
                    counter = 0
                    dates_array=[]
            if counter <= durations_of_short_failure:
                if counter!=0 and dates_array!=[]:
                    if counter == 1:
                        short_duration_failure[f'{dates_array[0]}'] = (counter, (df[variable].index[-1]-index).days)
                        counter = 0
                        dates_array=[]
                    else:
                        short_duration_failure[f'{dates_array[0]} >>> {dates_array[-1]}'] = (counter, (df[variable].index[-1]-index).days)
                        counter = 0
                        dates_array=[]

    return short_duration_failure, long_duration_failure
