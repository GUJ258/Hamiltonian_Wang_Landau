import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def f_normal(x):
    return np.exp(-0.5 * (x**2))

def U_normal(x):
    return 0.5 * (x**2)

def grad_U_normal(x):
    return x

def hmc_proposal(x, step_size, length):
    p = rng.normal()
    x_old, p_old = x, p

    g = grad_U_normal(x)
    p -= 0.5 * step_size * g

    for i in range(length-1):
        x += step_size * p
        g = grad_U_normal(x)
        p -= step_size * g

    x += step_size * p
    g = grad_U_normal(x)
    p -= 0.5 * step_size * g

    def H(x, p):
        U_tilde = U_normal(x)
        return U_tilde + 0.5 * np.dot(p, p)

    H_old = H(x_old, p_old)
    H_new = H(x, p)

    log_alpha = -(H_new - H_old)
    if np.log(rng.random()) < log_alpha:
        return x
    else:
        return x_old
    
def run_hmc(num_samples=1000, step_size=0.1, length=10):
    x = 0.0  # initial state
    samples = []

    for i in range(num_samples):
        x = hmc_proposal(x, step_size, length)
        samples.append(x)

    return samples


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



rng = np.random.default_rng(42)
samples = run_hmc(num_samples=10000, step_size=0.1, length=10)

n_lags = 100
acf_custom = plot_autocorrelation(samples, n_lags)
plt.figure(figsize=(7,5))
plt.plot(acf_custom, 's--', label="My ACF")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
plt.hist(samples, bins=40, density=True, alpha=0.6, color='steelblue', label="HMC samples")
plt.title("HMC Sampling from Standard Normal")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()