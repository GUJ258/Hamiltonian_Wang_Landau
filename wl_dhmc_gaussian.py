import numpy as np
from scipy.stats import gamma as gamma_dist
import matplotlib.pyplot as plt
from numba import njit
import time


# =============================================================================
# Potential and gradient
# =============================================================================

@njit(cache=True)
def _U_flat(r_flat, N, D):
    sum_sq = 0.0
    sum_r  = np.zeros(D)
    for i in range(N):
        sq = 0.0
        for d in range(D):
            v        = r_flat[i * D + d]
            sq      += v * v
            sum_r[d] += v
        sum_sq += sq
    sum_r_sq = 0.0
    for d in range(D):
        sum_r_sq += sum_r[d] * sum_r[d]
    return sum_sq + (0.5 / N) * (N * sum_sq - sum_r_sq)


@njit(cache=True)
def _grad_U_particle(r_flat, N, D, i):
    mean_r = np.zeros(D)
    for k in range(N):
        for d in range(D):
            mean_r[d] += r_flat[k * D + d]
    for d in range(D):
        mean_r[d] /= N
    g = np.empty(D)
    for d in range(D):
        g[d] = 3.0 * r_flat[i * D + d] - mean_r[d]
    return g


def U(r):
    return _U_flat(r.ravel(), r.shape[0], r.shape[1])


# =============================================================================
# Bin helpers
# =============================================================================

@njit(cache=True, inline='always')
def _bin_index(E, E_min, dE, n_bins):
    if E < E_min or E > E_min + dE * n_bins:
        return -1
    k = int((E - E_min) / dE)
    return min(k, n_bins - 1)


# =============================================================================
# DHMC internals (Numba)
# =============================================================================

@njit(cache=True)
def _U_single_particle(r_i_new, sum_sq_others, sum_r_others, N, D):
    """
    O(D) update of U when only particle i changes.

    Precompute before bisection:
        sum_sq_others = sum_{k!=i} ||r_k||^2
        sum_r_others  = sum_{k!=i} r_k   (length D)

    Then for any new r_i:
        sum_sq = sum_sq_others + ||r_i_new||^2
        sum_r  = sum_r_others  + r_i_new
        U = sum_sq + (0.5/N)*(N*sum_sq - ||sum_r||^2)
    """
    sq_i = 0.0
    for d in range(D):
        sq_i += r_i_new[d] * r_i_new[d]

    sum_sq = sum_sq_others + sq_i

    sum_r_sq = 0.0
    for d in range(D):
        s = sum_r_others[d] + r_i_new[d]
        sum_r_sq += s * s

    return sum_sq + (0.5 / N) * (N * sum_sq - sum_r_sq)


@njit(cache=True)
def _precompute_others(r_flat, i, N, D):
    """Compute sum_sq and sum_r excluding particle i. O(ND) but called once."""
    sum_sq = 0.0
    sum_r  = np.zeros(D)
    for k in range(N):
        if k == i:
            continue
        for d in range(D):
            v = r_flat[k * D + d]
            sum_sq      += v * v
            sum_r[d]    += v
    return sum_sq, sum_r


@njit(cache=True)
def _find_crossing(r_flat, p_i, i, N, D, E_lo, E_hi, t_max,
                   sum_sq_others, sum_r_others,
                   tol=1e-10, max_iter=60):
    """
    Optimized _find_crossing: bisection only touches particle i's D coords.
    Uses precomputed sum_sq_others and sum_r_others so each U eval is O(D).
    """
    # --- check endpoint ---
    r_i_end = np.empty(D)
    for d in range(D):
        r_i_end[d] = r_flat[i * D + d] + t_max * p_i[d]
    U_end = _U_single_particle(r_i_end, sum_sq_others, sum_r_others, N, D)
    u_calls      = 1
    bisect_iters = 0

    crossed_lo = U_end < E_lo
    crossed_hi = U_end > E_hi
    if not crossed_lo and not crossed_hi:
        return -1.0, False, False, u_calls, bisect_iters

    t_lo, t_hi = 0.0, t_max
    r_i_base = np.empty(D)
    for d in range(D):
        r_i_base[d] = r_flat[i * D + d]

    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        r_i_mid = np.empty(D)
        for d in range(D):
            r_i_mid[d] = r_i_base[d] + t_mid * p_i[d]
        U_mid = _U_single_particle(r_i_mid, sum_sq_others, sum_r_others, N, D)
        u_calls      += 1
        bisect_iters += 1
        if crossed_hi:
            if U_mid > E_hi: t_hi = t_mid
            else:            t_lo = t_mid
        else:
            if U_mid < E_lo: t_hi = t_mid
            else:            t_lo = t_mid
        if t_hi - t_lo < tol:
            break
    return 0.5 * (t_lo + t_hi), crossed_lo, crossed_hi, u_calls, bisect_iters


@njit(cache=True)
def _reflect_refract(p_i, grad_i, delta_log_g, out_of_range):
    gnorm_sq = 0.0
    for d in range(len(grad_i)):
        gnorm_sq += grad_i[d] * grad_i[d]
    if gnorm_sq < 1e-28:
        return p_i.copy(), False
    gnorm = np.sqrt(gnorm_sq)
    p_n = 0.0
    for d in range(len(p_i)):
        p_n += p_i[d] * grad_i[d]
    p_n /= gnorm
    if out_of_range:
        p_new = p_i.copy()
        for d in range(len(p_i)):
            p_new[d] -= 2.0 * p_n * (grad_i[d] / gnorm)
        return p_new, False
    discriminant = p_n * p_n - 2.0 * delta_log_g
    p_new = p_i.copy()
    if discriminant >= 0.0:
        p_n_new = np.sign(p_n) * np.sqrt(discriminant)
        scale   = (p_n_new - p_n) / gnorm
        for d in range(len(p_i)):
            p_new[d] += scale * grad_i[d]
        return p_new, True
    else:
        scale = -2.0 * p_n / gnorm
        for d in range(len(p_i)):
            p_new[d] += scale * grad_i[d]
        return p_new, False


@njit(cache=True)
def _dhmc_single_particle(r_flat, p_i, particle_idx, N, D,
                           log_g, E_min, dE, n_bins, n_steps, step_size,
                           max_crossings=50, bisect_tol=1e-10):
    r     = r_flat.copy()
    p     = p_i.copy()
    i     = particle_idx
    E_max = E_min + dE * n_bins
    total_crossings  = 0
    total_u_calls    = 1   # initial U_curr
    total_bisect     = 0
    valid = True
    U_curr = _U_flat(r, N, D)

    # precompute sum_sq and sum_r excluding particle i (O(ND), once per trajectory)
    sum_sq_others, sum_r_others = _precompute_others(r, i, N, D)

    for _ in range(n_steps):
        if U_curr < E_min or U_curr > E_max:
            valid = False; break
        t_rem = step_size
        crossings_this_step = 0
        while t_rem > 1e-14:
            if crossings_this_step > max_crossings:
                valid = False; break
            k = _bin_index(U_curr, E_min, dE, n_bins)
            if k < 0:
                valid = False; break
            E_lo = E_min + k * dE
            E_hi = E_lo + dE
            E_lo_eff = E_lo if k > 0        else E_min
            E_hi_eff = E_hi if k < n_bins-1 else E_max

            # optimized: bisection only computes O(D) per iteration
            t_star, hit_lo, hit_hi, u_c, b_c = _find_crossing(
                r, p, i, N, D, E_lo_eff, E_hi_eff, t_rem,
                sum_sq_others, sum_r_others, bisect_tol)
            total_u_calls += u_c
            total_bisect  += b_c

            if t_star < 0.0:
                for d in range(D):
                    r[i * D + d] += t_rem * p[d]
                r_i_curr = np.empty(D)
                for d in range(D):
                    r_i_curr[d] = r[i * D + d]
                U_curr = _U_single_particle(r_i_curr, sum_sq_others, sum_r_others, N, D)
                total_u_calls += 1
                t_rem  = 0.0
                break

            for d in range(D):
                r[i * D + d] += t_star * p[d]
            r_i_curr = np.empty(D)
            for d in range(D):
                r_i_curr[d] = r[i * D + d]
            U_curr = _U_single_particle(r_i_curr, sum_sq_others, sum_r_others, N, D)
            total_u_calls += 1

            k_new        = k + 1 if hit_hi else k - 1
            out_of_range = (hit_hi and k == n_bins-1) or (hit_lo and k == 0)
            delta_lg     = 0.0 if out_of_range else (log_g[k_new] - log_g[k])
            grad_i = _grad_U_particle(r, N, D, i)
            p, refracted = _reflect_refract(p, grad_i, delta_lg, out_of_range)

            gnorm_sq = 0.0
            for d in range(D):
                gnorm_sq += grad_i[d] * grad_i[d]
            if gnorm_sq > 1e-28:
                gnorm  = np.sqrt(gnorm_sq)
                bsign  = 1.0 if hit_hi else -1.0
                nudge  = (1e-8 * bsign / gnorm) if (refracted and not out_of_range) \
                         else (-1e-8 * bsign / gnorm)
                for d in range(D):
                    r[i * D + d] += nudge * grad_i[d]
            r_i_curr = np.empty(D)
            for d in range(D):
                r_i_curr[d] = r[i * D + d]
            U_curr = _U_single_particle(r_i_curr, sum_sq_others, sum_r_others, N, D)
            total_u_calls += 1

            t_rem -= t_star
            total_crossings     += 1
            crossings_this_step += 1

        if not valid:
            break

    return r, p, total_crossings, valid, total_u_calls, total_bisect


def mh_step_single(r, i, log_g, e_min, dE, n_bins,
                   n_steps, step_size, momentum_sigma, rng):
    N, D   = r.shape
    r_flat = r.ravel().copy()
    p_i    = rng.normal(0.0, momentum_sigma, size=D)
    U_old  = _U_flat(r_flat, N, D)
    k_old  = _bin_index(U_old, e_min, dE, n_bins)
    H_old  = 0.5 * np.dot(p_i, p_i) + (log_g[k_old] if k_old >= 0 else np.inf)

    r_new_flat, p_new, _, valid, u_calls, bisect = _dhmc_single_particle(
        r_flat, p_i, i, N, D, log_g, e_min, dE, n_bins, n_steps, step_size)
    p_new  = -p_new
    U_new  = _U_flat(r_new_flat, N, D)
    k_new  = _bin_index(U_new, e_min, dE, n_bins)
    H_new  = 0.5 * np.dot(p_new, p_new) + (log_g[k_new] if k_new >= 0 else np.inf)
    # +2 for U_old and U_new in MH step
    u_calls += 2

    if not valid or np.isinf(H_new) or np.isnan(H_new):
        return r, False, u_calls, bisect
    if np.log(rng.uniform()) < H_old - H_new:
        return r_new_flat.reshape(N, D), True, u_calls, bisect
    return r, False, u_calls, bisect


# =============================================================================
# Wang-Landau outer loop
# =============================================================================

def wang_landau(r_init, e_min, e_max, n_bins,
                n_steps, step_size, momentum_sigma,
                log_f_init, log_f_final,
                flatness, check_interval, seed):
    rng  = np.random.default_rng(seed)
    r    = r_init.copy()
    N, D = r.shape
    dE   = (e_max - e_min) / n_bins

    log_g    = np.zeros(n_bins)
    hist_f   = np.zeros(n_bins, dtype=np.int64)
    hist_all = np.zeros(n_bins, dtype=np.int64)

    centers = np.array([e_min + (k + 0.5) * dE for k in range(n_bins)])
    dof     = N * D
    vmask   = centers > 0
    theory  = (dof / 2.0 - 1.0) * np.log(centers[vmask])
    theory -= theory.min()

    log_f        = float(log_f_init)
    stage        = 0
    total_steps  = 0
    accepted     = 0
    mse_history  = []
    log_g_history= []
    current_bin  = _bin_index(_U_flat(r.ravel(), N, D), e_min, dE, n_bins)
    steps_in_bin = 1
    stay_lengths = []

    t0 = time.time()
    print(f"WL-DHMC (single-particle)  N={N}  D={D}")
    print(f"  E=[{e_min:.3f}, {e_max:.3f}]  bins={n_bins}  dE={dE:.4f}")
    print(f"  L={n_steps}  step={step_size}  sigma={momentum_sigma}")
    print(f"  log_f: {log_f_init} -> {log_f_final}  "
          f"flatness={flatness}  check_interval={check_interval:,}")
    print("=" * 60)

    while log_f > log_f_final:
        print(f"Stage {stage}: log_f = {log_f:.6f}")
        hist_f[:] = 0
        steps_this_stage    = 0
        accepted_this_stage = 0
        stay_lengths        = []
        u_calls_stage       = 0
        bisect_stage        = 0

        while True:
            i = rng.integers(N)
            r, acc, u_c, b_c = mh_step_single(r, i, log_g, e_min, dE, n_bins,
                                    n_steps, step_size, momentum_sigma, rng)
            accepted            += int(acc)
            accepted_this_stage += int(acc)
            total_steps         += 1
            steps_this_stage    += 1
            u_calls_stage       += u_c
            bisect_stage        += b_c

            E_curr = _U_flat(r.ravel(), N, D)
            k = _bin_index(E_curr, e_min, dE, n_bins)
            if k >= 0:
                log_g[k]   += log_f
                hist_f[k]  += 1
                hist_all[k]+= 1

            lg = log_g[vmask] - log_g[vmask].min()
            mse_history.append(float(np.mean((lg - theory) ** 2)))

            if k >= 0 and k != current_bin:
                stay_lengths.append(steps_in_bin)
                current_bin  = k
                steps_in_bin = 1
            else:
                steps_in_bin += 1

            if steps_this_stage % check_interval == 0:
                visited  = hist_f[hist_f > 0]
                h_min    = int(visited.min()) if len(visited) else 0
                h_mean   = float(visited.mean()) if len(visited) else 0.0
                ratio    = h_min / h_mean if h_mean > 0 else 0.0
                avg_stay = float(np.mean(stay_lengths)) if stay_lengths else float(steps_in_bin)
                elapsed  = time.time() - t0
                avg_u    = u_calls_stage / steps_this_stage
                avg_b    = bisect_stage  / steps_this_stage
                print(f"  iter {total_steps:>10,} | Hmin {h_min} | "
                      f"ratio {ratio:.3f} | accept {accepted_this_stage/steps_this_stage:.3f} | "
                      f"avg_stay {avg_stay:.1f} | MSE {mse_history[-1]:.4f} | "
                      f"U/step {avg_u:.1f} | bisect/step {avg_b:.1f} | t {elapsed:.1f}s")
                if ratio >= flatness:
                    print(f"  >> Stage {stage} passed.")
                    log_g_history.append(log_g.copy())
                    break

        log_f /= 2.0
        stage  += 1

    total_time = time.time() - t0
    print("=" * 60)
    print(f"Done. stages={stage}  steps={total_steps:,}  time={total_time:.1f}s")

    return {
        "log_g"        : log_g,
        "log_g_history": log_g_history,
        "hist_all"     : hist_all,
        "mse_history"  : np.array(mse_history),
        "stay_lengths" : np.array(stay_lengths),
        "centers"      : centers,
        "total_steps"  : total_steps,
        "total_time"   : total_time,
        "stage"        : stage,
    }


# =============================================================================
# Helpers
# =============================================================================

def estimate_energy_range(N, D, low_q=0.005, high_q=0.995,
                           e_min_floor=0.5, n_fit=5000, seed=0):
    n, shape = N * D, N * D / 2.0
    rng   = np.random.default_rng(seed)
    pilot = np.array([U(rng.standard_normal((N, D))) for _ in range(n_fit)])
    theta = pilot.mean() / shape
    e_min = max(e_min_floor, float(gamma_dist.ppf(low_q,  shape, scale=theta)))
    e_max = float(gamma_dist.ppf(high_q, shape, scale=theta))
    print(f"Energy range: [{e_min:.3f}, {e_max:.3f}]  "
          f"(U ~ Gamma(shape={shape:.1f}, scale={theta:.4f}))")
    return e_min, e_max


def find_initial_config(N, D, e_min, e_max, n_tries=100_000, seed=1):
    rng = np.random.default_rng(seed)
    for trial in range(n_tries):
        scale = 0.05 + 3.0 * (trial / n_tries)
        r = rng.standard_normal((N, D)) * scale
        if e_min <= U(r) < e_max:
            print(f"Initial config: trial={trial+1}  U={U(r):.4f}")
            return r
    raise RuntimeError("Could not find initial config.")


# =============================================================================
# Plot
# =============================================================================

def plot_results(res, N, D, n_steps, step_size, momentum_sigma):
    centers = res["centers"]
    log_g   = res["log_g"].copy();  log_g -= log_g.min()
    dof     = N * D
    ref     = (dof / 2.0 - 1.0) * np.log(np.maximum(centers, 1e-300))
    ref    -= ref.min()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"WL-DHMC (single-particle)  N={N}  D={D}  "
        f"L={n_steps}  step={step_size}  sigma={momentum_sigma}  "
        f"stages={res['stage']}  steps={res['total_steps']:,}  "
        f"time={res['total_time']:.1f}s",
        fontsize=9)

    axes[0].plot(centers, log_g, lw=2, label="WL")
    axes[0].plot(centers, ref, 'r--', lw=1.5, label="Theory")
    axes[0].set(xlabel="E", ylabel="ln g(E) [norm]", title="Density of States")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(centers, res["hist_all"],
                width=(centers[1]-centers[0])*0.9, color="C3", alpha=0.7)
    axes[1].set(xlabel="E", ylabel="Visits", title="Overall Histogram")
    axes[1].grid(True, alpha=0.3)

    mse = res["mse_history"]
    if len(mse):
        w  = max(1, len(mse) // 500)
        sm = np.convolve(mse, np.ones(w)/w, mode='valid')
        axes[2].semilogy(sm, color='purple', lw=0.8)
    axes[2].set(xlabel="Iteration", ylabel="MSE", title="Convergence")
    axes[2].grid(True, alpha=0.3)

    half_n     = N * D / 2.0
    log_g_norm = res["log_g"] - res["log_g"].min()
    log_g_an   = (half_n - 1.0) * np.log(centers)
    log_g_an  -= log_g_an.min()
    T_lo   = 2 * centers[0]  / (N * D)
    T_hi   = 2 * centers[-1] / (N * D)
    T_vals = np.linspace(T_lo, T_hi, 300)

    def compute_S(lg, ctrs, T_arr):
        out = []
        for T in T_arr:
            lw      = lg - ctrs / T
            shift   = lw.max()
            log_Z   = shift + np.log(np.sum(np.exp(lw - shift)))
            weights = np.exp(lw - shift)
            E_mean  = np.dot(ctrs, weights) / weights.sum()
            out.append(log_Z + E_mean / T)
        return np.array(out)

    S_wl = compute_S(log_g_norm, centers, T_vals)
    S_an = compute_S(log_g_an,   centers, T_vals)
    axes[3].plot(T_vals, S_wl, color='C1', lw=1.5, label="WL")
    axes[3].plot(T_vals, S_an, color='r',  lw=1.5, ls='--', label="Analytic")
    axes[3].legend(fontsize=8)
    axes[3].set(xlabel="Temperature T", ylabel="S(T)",
                title=f"Entropy vs Temperature  [T∈{T_lo:.1f},{T_hi:.1f}]")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = (f"wl_dhmc_single_N{N}_D{D}"
             f"_L{n_steps}_step{step_size}_sigma{momentum_sigma}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Compiling Numba kernels...")
    _r = np.ones(6)
    _U_flat(_r, 2, 3)
    _grad_U_particle(_r, 2, 3, 0)
    print("Done.\n")

    # --- parameters ---
    N, D           = 100, 5
    N_BINS         = 100
    N_STEPS        = 10
    STEP_SIZE      = 0.1
    MOMENTUM_SIGMA = 1.0
    LOG_F_INIT     = 1.0
    LOG_F_FINAL    = 1e-4
    FLATNESS       = 0.90
    CHECK_INTERVAL = 10_000
    SEED           = 42

    E_MIN, E_MAX = estimate_energy_range(N, D, low_q=0.005, high_q=0.995, seed=0)
    r0 = find_initial_config(N, D, E_MIN, E_MAX, seed=1)

    res = wang_landau(
        r0,
        e_min          = E_MIN,
        e_max          = E_MAX,
        n_bins         = N_BINS,
        n_steps        = N_STEPS,
        step_size      = STEP_SIZE,
        momentum_sigma = MOMENTUM_SIGMA,
        log_f_init     = LOG_F_INIT,
        log_f_final    = LOG_F_FINAL,
        flatness       = FLATNESS,
        check_interval = CHECK_INTERVAL,
        seed           = SEED,
    )

    plot_results(res, N, D, N_STEPS, STEP_SIZE, MOMENTUM_SIGMA)