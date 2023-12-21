# Description: Fills in the gaps with the average associated with the day of the week and its time
import math

def reconstruction(df,variable,short_duration_failure,long_duration_failure,values_per_week_day_hour_average):
    week_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for i,v in df[variable].items():
        if math.isnan(v):
            df[variable].loc[i] = values_per_week_day_hour_average[f'{week_days[i.weekday()]} - {i.hour}']
    
    return df