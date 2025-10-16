import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# U(x) = (1 - x1)^2 + 100*(x2 - x1^2)^2


def f_rosenbrock(x):
    x1, x2 = x[0], x[1]
    U = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    return np.exp(-U)

def U_rosenbrock(x):
    x1, x2 = x[0], x[1]
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

def grad_U_rosenbrock(x):
    x1, x2 = x[0], x[1]
    dU_dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    dU_dx2 = 200 * (x2 - x1**2)
    return np.array([dU_dx1, dU_dx2])


def hmc_proposal(x, step_size, length, rng):
    p = rng.normal(size=2)
    x_old, p_old = x.copy(), p.copy()

    g = grad_U_rosenbrock(x)
    p -= 0.5 * step_size * g

    for i in range(length-1):
        x += step_size * p
        g = grad_U_rosenbrock(x)
        p -= step_size * g

    x += step_size * p
    g = grad_U_rosenbrock(x)
    p -= 0.5 * step_size * g

    def H(x, p):
        U_tilde = U_rosenbrock(x)
        return U_tilde + 0.5 * np.dot(p, p)

    H_old = H(x_old, p_old)
    H_new = H(x, p)

    log_alpha = -(H_new - H_old)
    accepted = np.log(rng.random()) < log_alpha
    if accepted:
        return x.copy(), True
    else:
        return x_old.copy(), False
    
def run_hmc(num_samples=1000, step_size=0.1, length=10):
    rng = np.random.default_rng(42)
    x = np.array([0.0, 0.0])  # initial state
    samples = []
    accept_count = 0

    for i in range(num_samples):
        x, accepted = hmc_proposal(x, step_size, length, rng)
        samples.append(x.copy())
        if accepted:
            accept_count += 1

    accept_rate = accept_count / num_samples
    print(f"Acceptance rate = {accept_rate:.3f}")

    return np.array(samples)


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
samples = run_hmc(num_samples=100000, step_size=0.01, length=100)
plt.plot(samples[:,0], samples[:,1], '.', alpha=0.4)
plt.xlabel('x1'); plt.ylabel('x2')
plt.title('HMC Trajectory on Rosenbrock')
plt.show()

n_lags = 100
acf_custom = plot_autocorrelation(samples[:,0], n_lags)
plt.figure(figsize=(7,5))
plt.plot(acf_custom, 's--', label="ACF in Dimension1(x1)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()

acf_custom = plot_autocorrelation(samples[:,1], n_lags)
plt.figure(figsize=(7,5))
plt.plot(acf_custom, 's--', label="ACF in Dimension1(x2)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True, ls="--", alpha=0.6)
plt.legend()
plt.show()