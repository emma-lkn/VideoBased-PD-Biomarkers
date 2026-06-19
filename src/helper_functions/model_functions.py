import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

######################################################## Linear
def fit_linear_model(speed, amplitude):
    """
    amplitude = beta_0 + beta_1 * speed
    """
    X = speed.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, amplitude)
    
    return model


def predict_linear_model(model, speed):
    predict = model.predict(speed.reshape(-1, 1))
    return predict

######################################################## Exponential
def fit_exponential_model(speed, amplitude):
    """
    amplitude = a * exp(b * speed)
    However, we compute in log space to linearise it:
    log(amplitude) = log(a) + b * speed
    
    """
    amp_clip = np.clip(amplitude, 0.001, None)
    amplitude_log = np.log(amp_clip)
    X = speed.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, amplitude_log)
    
    return model


def predict_exponential_model(model, speed):
    log_pred = model.predict(speed.reshape(-1, 1))
    predict = np.exp(log_pred)
    return predict

######################################################## Logarithmic
def fit_logarithmic_model(speed, amplitude):
    """
    amplitude = a + b * log(speed)
    """
    speed_log = np.log(speed + 1e-8)  
    X = speed_log.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, amplitude)
    
    return model

def predict_logarithmic_model(model, speed):
    speed_log = np.log(speed + 1e-8)
    predict = model.predict(speed_log.reshape(-1, 1))
    predict = np.maximum(predict, 0) # added this in case for negative values)
    return predict

######################################################## Logistic
# while the logstic_function and fit_logistic_model could be done in a single function,
# i wanted the fit and predict to follow the same structure as the other models,
# such that the inputs for all fit and predict functions are consistent
# maybe use bounds: bounds=([0, 0, -10, np.min(speed)], [1000, 1000, 10, np.max(speed)])

def det_logistic_params(speed, amplitude):
    """
    estimates the parameters for the logistic function, using scipy.optimize curve_fit
    """
    initial_C = np.min(amplitude)
    initial_L = np.max(amplitude) - np.min(amplitude)
    initial_k = 1.0                                
    initial_x_0 = np.median(speed)  
    
    initial_estimation = [initial_C, initial_L, initial_k, initial_x_0]
    
    parameters, _ = curve_fit(
        logistic_function,
        speed,
        amplitude,
        p0 = initial_estimation,
        maxfev=10000,
        bounds=([0, 0, -10, np.min(speed)], [1000, 1000, 10, np.max(speed)])
    )
    
    return parameters

def logistic_function(speed, C, L, k, x_0):
    """
    C: lower asymptote
    L: difference between lupper and lower asymptote
    k: slope
    x_0: inflection point
    """
    exponent = -k * (speed - x_0)
    exponent = np.clip(exponent, -100, 100) # to prevent overflow
    logistic = C + (L / (1 + np.exp(exponent)))
    return logistic


def fit_logistic_model(speed, amplitude):
    """
    amplitude = C + (L / (1 + e^(-k * (speed - x_0))))
    """
    C, L, k, x_0 = det_logistic_params(speed, amplitude)
    model = {"C": C, "L": L, "k": k, "x_0": x_0}
    
    return model


def predict_logistic_model(model, speed):
    predict = logistic_function(speed, model["C"], model["L"], model["k"], model["x_0"])
    return predict


######################################################## Pareto Fronts
def pareto_front_true(speed, amplitude):
    #sort_idx = np.argsort(speed)[::-1]
    sort_idx = np.argsort(speed)
    speed_sorted = speed[sort_idx]
    amp_sorted = amplitude[sort_idx]
    
    pareto_speed = []
    pareto_amp = []
    best_current_amp = -np.inf
    
    for speed_, amp_ in zip(speed_sorted, amp_sorted):
        if amp_ > best_current_amp:
            pareto_speed.append(speed_)
            pareto_amp.append(amp_)
            best_current_amp = amp_

    speed_front = np.array(pareto_speed)
    amp_front =  np.array(pareto_amp)

    return speed_front, amp_front

