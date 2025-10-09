import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import time

class WangLandau:
    def __init__(self, U, xmin=-2.0, xmax=2.0,
                 f0=np.exp(1.0), f_min=1e-6,
                 flatness=0.8, steps=200000):
        # potential related params
        self.U = U                                                          # potential

        # Theta_n related params
        self.xmin = xmin
        self.xmax =xmax
        self.xs = np.linspace(xmin, xmax, int((xmax-xmin)*100)+1)                    
        self.Es = np.array([U(x) for x in self.xs])
        self.Emin = int(self.Es.min())                                      # min_energy
        self.Emax = int(self.Es.max())                                      # max_energy
        self.M = int((xmax-xmin)*10)                                        # number of bins of theta
        self.edges = np.linspace(self.Emin, self.Emax, self.M+1)
        self.E_bins = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.f0 = f0                                                        # initial increasing factor
        self.log_f0 = np.log(self.f0)                                       # initial log increasing factor
        self.log_f = np.log(self.f0)                                        # current log increasing factor
        self.f_min = f_min                                                  # minimum increasing factor
        self.log_theta_n = np.zeros(self.M)                                 # initial guesses of log(theta_star(E))

        # histogram related params
        self.flatness = flatness                                            # flatness check
        self.steps = steps                                                  # number of sampling x (or E)
        self.H = np.zeros(self.M)                                           # initial histogram of E

        # random number generator
        self.rng = np.random.default_rng(42)

        # HMC related params
        self.grad_U = jax.jit(jax.grad(U))                                  # gradient of potential
        self.f_n = lambda x: 0.0
        self.f_prime_n = lambda x: 0.0
        self.log_f_list = []
        self.E_list = []
        self.kernel_width_0 = 0.0
        self.kernel_width_list = []
        self.kernel_decaying_factor = 0.0

    # original Wang Landau
    def random_walk_proposal(self, x):
        return x + self.rng.uniform(-2, 2)
    
    def run_rdw(self):
        x = (self.xmax + self.xmin) / 2
        E = self.U(x)
        idx = np.digitize(E, self.edges) - 1

        it = 0

        trajectory = []

        while np.exp(self.log_f) > self.f_min and it < self.steps:
            x_new = self.random_walk_proposal(x)
            E_new = self.U(x_new)

            idx_new = np.digitize(E_new, self.edges) - 1

            if 0 <= idx_new < self.M:
                log_p_acc = self.log_theta_n[idx] - self.log_theta_n[idx_new]
                if np.log(self.rng.random()) < log_p_acc:
                    x, E, idx = x_new, E_new, idx_new

            self.log_theta_n[idx] += self.log_f
            self.H[idx] += 1

            trajectory.append(x)

            if it % 100 == 0 and np.min(self.H) > self.flatness * np.mean(self.H):
                self.H[:] = 0
                self.log_f /= 2.0

            it += 1

        return np.array(trajectory)
    
    # HMC Based Wang_Landau
    def kernel(self, u):
        return jnp.exp(- u ** 2 / 2)

    def update_gradient(self, x, it):
        if it == 0:
            return x
        else: 
            curr_E = self.U(x)
            self.log_f_list.append(self.log_f)
            self.kernel_width_list.append(self.kernel_width_0 * (self.log_f / self.log_f0) ** self.kernel_decaying_factor)
            self.E_list.append(curr_E)

            def f_new(E):
                total = 0.0
                for i in range(len(self.E_list)):
                    E_i = self.E_list[i]
                    log_f_i = self.log_f_list[i]
                    kernel_width_i = self.kernel_width_list[i]
                    total += (log_f_i / kernel_width_i) * self.kernel((E - E_i) / kernel_width_i)
                return total

            self.f_n = f_new


            def f_prime_new(E):
                total = 0.0
                for i in range(len(self.E_list)):
                    E_i = self.E_list[i]
                    log_f_i = self.log_f_list[i]
                    kernel_width_i = self.kernel_width_list[i]
                    total += - (log_f_i / (kernel_width_i ** 2)) * (E - E_i) * self.kernel((E - E_i) / kernel_width_i)
                return total
                
            self.f_prime_n = f_prime_new


    def hmc_proposal(self, x, it, step_size, L):
        if it == 0:
            return x
        else: 
            p = np.random.normal()
            x_old, p_old = x, p

            g = self.f_prime_n(x) / self.f_n(x) * self.grad_U(x)
            p -= 0.5 * step_size * g

            for i in range(L-1):
                x += step_size * p
                g = self.f_prime_n(x) / self.f_n(x) * self.grad_U(x)
                p -= step_size * g

            x += step_size * p
            g = self.f_prime_n(x) / self.f_n(x) * self.grad_U(x)
            p -= 0.5 * step_size * g

            def H(x, p):
                U_tilde = -jnp.log(self.f_n(x))
                return U_tilde + 0.5 * np.dot(p, p)

            H_old = H(x_old, p_old)
            H_new = H(x, p)

            log_alpha = -(H_new - H_old)
            if np.log(self.rng.random()) < log_alpha:
                return x
            else:
                return x_old
        

    def run_hmc(self, kernel_width_0, kernel_decaying_factor, step_size, L):
        x = jnp.array((self.xmax + self.xmin) / 2.0)
        E = self.U(x)
        idx = np.digitize(E, self.edges) - 1

        it = 0

        self.kernel_width_0 = kernel_width_0
        self.kernel_decaying_factor = kernel_decaying_factor

        trajectory = []

        while np.exp(self.log_f) > self.f_min and it < self.steps:
            self.update_gradient(x, it)
            x_new = self.hmc_proposal(x, it, step_size, L)
            E_new = self.U(x_new)

            idx_new = np.digitize(E_new, self.edges) - 1

            if 0 <= idx_new < self.M:
                log_p_acc = self.log_theta_n[idx] - self.log_theta_n[idx_new]
                if np.log(self.rng.random()) < log_p_acc:
                    x, E, idx = x_new, E_new, idx_new

            self.log_theta_n[idx] += self.log_f
            self.H[idx] += 1

            trajectory.append(x)

            if it % 1000 == 0 and np.min(self.H) > self.flatness * np.mean(self.H):
                self.H[:] = 0
                self.log_f /= 2.0

            it += 1

        return np.array(trajectory)
    

    def compare_wl_and_real(self):

        g_wl = np.exp(self.log_theta_n)
        g_real, _ = np.histogram(self.Es, bins=self.edges)

        plt.plot(self.E_bins, g_wl / g_wl.max(), label="WL g(E) (norm.)")
        plt.plot(self.E_bins, g_real / g_real.max(), label="True g(E) (norm.)")
        plt.ylabel("g(E) normalized")

        plt.xlabel("E")
        plt.legend()
        plt.show()


def U(x):
    return (1 - x**2)**2 + 1 * x

def plot_autocorrelation(trajectory, max_lag=100, label="x"):
    x = np.ravel(trajectory) - np.mean(trajectory)
    n = len(x)
    f = np.fft.fft(x, n*2)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]

    lags = np.arange(max_lag+1)
    plt.figure(figsize=(6,4))
    plt.stem(lags, acf[:max_lag+1], basefmt="r-")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation of {label}")
    plt.grid(True, ls="--", alpha=0.6)
    plt.show()



start = time.time()
wl = WangLandau(U, xmin=-2.0, xmax=2.0, steps=300)
# traj = wl.run_rdw()
traj = wl.run_hmc(kernel_width_0=1, kernel_decaying_factor=0.5, step_size=0.1, L=10)
end = time.time()
print(f"runtime: {end - start:.2f} seconds")
wl.compare_wl_and_real()
plot_autocorrelation(traj, max_lag=100)
