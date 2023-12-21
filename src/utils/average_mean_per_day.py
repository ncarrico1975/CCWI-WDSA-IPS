import math
from statistics import mean, median

def average_per_weekday(df,variable,method='average'):
    week_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    keys_of_dict_week_day=[]
    for wd in week_days:
        for h in range(0,24):
            keys_of_dict_week_day.append(f'{wd} - {h}')
    
    dict_week_day = {key: [] for key in keys_of_dict_week_day}
    # print('END')
    for i, v in df[variable].items():
        if math.isnan(v)==False:
            dict_week_day[f'{week_days[i.weekday()]} - {i.hour}'].append(v)

    dict_week_day_hour_average = {}
    for key in keys_of_dict_week_day:
        dict_week_day_hour_average[key] = mean(dict_week_day[key])

    return dict_week_day, dict_week_day_hour_average