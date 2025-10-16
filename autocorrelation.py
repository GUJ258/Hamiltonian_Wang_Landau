import numpy as np
import matplotlib.pyplot as plt

def plot_autocorrelation(x_t, n_lags):
    mean_x = np.mean(x_t)
    acf_values = np.zeros(n_lags+1)
    
    for lag_index, lag in enumerate(range(n_lags + 1)):
        if lag == 0:
            acf_lag = np.sum((x_t - mean_x) ** 2)
        else:
            acf_lag = np.sum((x_t[lag:] - mean_x) * (x_t[:-lag] - mean_x))
        acf_values[lag_index] = acf_lag

    return acf_values / acf_values[0]

# x_t = rho * x_{t-1} + ε_t
np.random.seed(42)
n = 50000
rho = 0.9
eps = np.random.normal(0, 1, n)
x = np.zeros(n)
for t in range(1, n):
    x[t] = rho * x[t-1] + eps[t]

n_lags = 50
acf_custom = plot_autocorrelation(x, n_lags)
acf_theoretical = rho ** np.arange(n_lags+1)


plt.figure(figsize=(7,5))
plt.plot(acf_theoretical, 'k-', label="Theoretical ρ^k")
plt.plot(acf_custom, 's--', label="My ACF")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title(f"AR(1) process (ρ={rho}) — ACF comparison")
plt.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()
