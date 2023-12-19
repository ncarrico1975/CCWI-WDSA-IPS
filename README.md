# Battle of Water Demand Forecasting (BWDF)


### Background and research direction
The Battle of Water Demand Forecasting (BWDF), organized in the context of the 3rd International WDSA-CCWI Joint Conference in Ferrara, Italy (https://wdsa-ccwi2024.it), aims at comparing the effectiveness of methods for the **short-term prediction of urban water demand**, relying on Supervisory Control and Data Acquisition (SCADA) measurements, and mass balance calculations, in a set of real District Metered Areas (DMAs).

### Goal of the challenge, materials, and rules
The challenge proposed in the context of the BWDF is focused on the forecast of the water demands with reference to a case-study WDN located in the North-East of Italy, supplying a variety of areas that are considerably different as far as characteristics, size, and average water demand are concerned. Specifically, **forecasting is required for ten DMAs** of the WDN concerned with the **aim of defining optimal system operation for the near future (i.e., upcoming day and week), and optimizing the energy purchase**. The water demand of each DMA is assumed to be represented by the net inflow and thus it **includes all the types of water consumption and leakages of the DMA**.

### DMA characteristics

The water utility managing the DMAs concerned provided the hourly net-inflow time series $Q_{net}$ (L/s) for each DMA in relation to the period from 1 January 2021 to 31 March 2023. 

Net-inflow time series include water consumptions and leakages and are obtained through water balance:

$Q_{net}=\sum_{i=1}^{n_{in}} Q_{in,i} -\sum_{j=1}^{n_{out}} Q_{out,j}$


in which $Q_{in}$ is the flow rate entering the DMA concerned through the 𝑖-th inlet point (𝑖 = 1,2, ... $n_{𝑖𝑛}$) and acquired by the water utility SCADA system, whereas $Q_{out}$ is the flow rate outgoing from the DMA concerned through the 𝑗-th outlet point (𝑗 = 1,2, ... $n_{out}$). It is worth noting that no DMAs with storage facilities are included. Moreover, net-inflow data are not post-processed, so they can show some gaps related to SCADA system malfunctioning and other data collection/transmission issues.

## The challenge

First step in the challenge is to **forecast week 30 of 2022**, for all DMAs, that is, for the period **25/07/2022-31/07/2022**.

Solution should be stored in the file: SolutionTemplate_W1.xlsx

**Deadline for submission**: 31st January 2024

# Files that are needed to run this notebook:

* InflowData_1.xlsx

* WeatherData_1.xlsx

# Content of this notebook

* [1. Exploratory analysis of hourly net-inflow](#sec:eda_net_inflow)

* [2. Exploratory analysis of weather data](#sec:eda_weather)

* [3. Exploratory analysis of joint data](#sec:eda_joint)
