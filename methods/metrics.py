import numpy as np



# To compute Battle's metrics
# WARNING, for this to work, forecast range should be an entire week

def MeanError1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
    
    SumError = np.sum(Allerrors)
    AvgError = SumError / len(Allerrors)
    return AvgError

def MaxError1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
    
    MaxError = np.max(Allerrors)
    return MaxError

def MeanErrorExcluding1stday(y_pred, y_true):
    Allerrors = np.array([])
    for i in range(24, len(y_pred)):
        error = abs(y_pred[i]-y_true[i])
        Allerrors = np.append(Allerrors, error)
        
    SumError = np.sum(Allerrors)
    AvgError = SumError / len(Allerrors)
    return AvgError

def BattleMetrics_per_dma(y_pred, y_true):
    return MeanError1stday(y_pred, y_true)+MaxError1stday(y_pred, y_true)+ MeanErrorExcluding1stday(y_pred, y_true)
