def reconstruct_df_daily(df, history):
    df = df.copy()
    df.sort_index(inplace=True)
 
    history = history.copy()
    history.sort_index(inplace=True)
   
    list_of_nans = df.index[df['value'].isnull()]
 
    df['day_of_week'] = df.index.day_name()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
 
    history['day_of_week'] = history.index.day_name()
    history['hour'] = history.index.hour
    history['minute'] = history.index.minute
 
    descriptive_statistics = history.groupby(['day_of_week', 'hour','minute']).describe()
 
    for index_missing_measurement in list_of_nans:
        minute = index_missing_measurement.minute
        hour = index_missing_measurement.hour
        day_of_week = index_missing_measurement.day_name()
 
        avg = descriptive_statistics.loc[day_of_week,hour,minute][1]
        df.loc[index_missing_measurement,'value'] = avg
 
 
    df.sort_index(inplace=True)
 
    return df[['value']],list_of_nans


def reconstruct_df_history(df):
    df = df.copy()
 
    df.sort_index(inplace=True)
   
 
    list_of_nans = df.index[df['value'].isnull()]
 
 
    df['day_of_week'] = df.index.day_name()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
 
    descriptive_statistics = df.groupby(['day_of_week', 'hour','minute']).describe()
 
    for index_missing_measurement in list_of_nans:
        minute = index_missing_measurement.minute
        hour = index_missing_measurement.hour
        day_of_week = index_missing_measurement.day_name()
 
        avg = descriptive_statistics.loc[day_of_week,hour,minute][1]
        df.loc[index_missing_measurement,'value'] = avg
 
 
    df.sort_index(inplace=True)
 
    return df[['value']],list_of_nans