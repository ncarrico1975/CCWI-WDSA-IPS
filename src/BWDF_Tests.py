import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.average_mean_per_day import average_per_weekday
from utils.identify_durations_of_gaps import durations_gaps
from utils.reconstruction_strategies import reconstruction

df = pd.read_excel('documents/InflowData_1.xlsx')

transf_dict = {
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

df.rename(columns= transf_dict, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'],format='%d/%m/%Y %H:%M')
df.set_index('datetime', inplace = True)


short_duration_failure, long_duration_failure = durations_gaps(df,'dma_A',2)
values_per_weekday, values_per_week_day_hour_average = average_per_weekday(df,'dma_A')

df = reconstruction(df,'dma_A',short_duration_failure,long_duration_failure,values_per_week_day_hour_average)

short_duration_failure, long_duration_failure = durations_gaps(df,'dma_A',2)
values_per_weekday, values_per_week_day_hour_average = average_per_weekday(df,'dma_A')