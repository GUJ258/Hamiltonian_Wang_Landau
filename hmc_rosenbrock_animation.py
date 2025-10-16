import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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



rng = np.random.default_rng(42)
samples = run_hmc(num_samples=1000, step_size=0.01, length=100)

fig, ax = plt.subplots(figsize=(6,5))
x1 = np.linspace(-2, 4, 300)
x2 = np.linspace(-1, 15, 300)
X1, X2 = np.meshgrid(x1, x2)
Z = (1 - X1)**2 + 100 * (X2 - X1**2)**2
ax.contour(X1, X2, np.log(Z + 1), levels=30, cmap="gray", alpha=0.3)

point, = ax.plot([], [], 'ro', markersize=5)
path, = ax.plot([], [], 'b-', alpha=0.4)
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 15)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('HMC Sampling on Rosenbrock Function')

# -------------------------------
# Animation update function
# -------------------------------
def update(frame):
    path.set_data(samples[:frame, 0], samples[:frame, 1])
    point.set_data(samples[frame, 0], samples[frame, 1])
    return point, path

ani = animation.FuncAnimation(fig, update, frames=len(samples), interval=50, blit=True)

plt.show()

ani.save("hmc_rosenbrock.gif", writer="pillow", fps=20)
