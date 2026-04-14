import numpy as np
from scipy.stats import gamma as gamma_dist
import matplotlib.pyplot as plt
from numba import njit
import time


# =============================================================================
# Potential
# =============================================================================

@njit(cache=True)
def _U_flat(r_flat, N, D):
    """O(ND) pairwise quadratic via identity."""
    sum_sq = 0.0
    sum_r  = np.zeros(D)
    for i in range(N):
        sq = 0.0
        for d in range(D):
            v       = r_flat[i * D + d]
            sq     += v * v
            sum_r[d] += v
        sum_sq += sq
    sum_r_sq = 0.0
    for d in range(D):
        sum_r_sq += sum_r[d] * sum_r[d]
    return sum_sq + (0.5 / N) * (N * sum_sq - sum_r_sq)


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


@njit(cache=True)
def _U_single_particle(r_i_new, sum_sq_others, sum_r_others, N, D):
    """O(D) U evaluation when only particle i changes."""
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
def _init_sums(r_flat, N, D):
    """Compute global sum_sq = sum_k ||r_k||^2 and sum_r = sum_k r_k. O(ND)."""
    sum_sq = 0.0
    sum_r  = np.zeros(D)
    for k in range(N):
        for d in range(D):
            v = r_flat[k * D + d]
            sum_sq    += v * v
            sum_r[d]  += v
    return sum_sq, sum_r


@njit(cache=True)
def _rdw_step(r_flat, e, log_g, e_min, dE, n_bins,
              N, D, step_size,
              rand_i, rand_delta, rand_u,
              sum_sq, sum_r):
    """
    Single RDW step (Numba), O(D) U evaluation:
    - subtract particle i's old contribution from sum_sq, sum_r
    - compute proposed r_i_new
    - evaluate U_new in O(D) via _U_single_particle
    - WL accept/reject
    - if accepted: update r_flat, sum_sq, sum_r in O(D)
    Returns (r_flat, e, accepted, u_calls=1, sum_sq, sum_r)
    """
    i = rand_i

    # old coordinates of particle i
    sq_i_old = 0.0
    r_i_old  = np.empty(D)
    for d in range(D):
        v = r_flat[i * D + d]
        r_i_old[d]  = v
        sq_i_old   += v * v

    # others = global - particle i
    sum_sq_others = sum_sq - sq_i_old
    sum_r_others  = np.empty(D)
    for d in range(D):
        sum_r_others[d] = sum_r[d] - r_i_old[d]

    # proposed r_i
    r_i_new = np.empty(D)
    for d in range(D):
        r_i_new[d] = r_i_old[d] + rand_delta[d] * step_size

    # O(D) U evaluation
    e_new   = _U_single_particle(r_i_new, sum_sq_others, sum_r_others, N, D)
    u_calls = 1

    k_old = _bin_index(e,     e_min, dE, n_bins)
    k_new = _bin_index(e_new, e_min, dE, n_bins)

    accepted = False
    if k_new >= 0:
        log_alpha = log_g[k_old] - log_g[k_new]
        if np.log(rand_u) < log_alpha:
            # update r_flat
            for d in range(D):
                r_flat[i * D + d] = r_i_new[d]
            e = e_new
            accepted = True
            # O(D) update of sum_sq and sum_r
            sq_i_new = 0.0
            for d in range(D):
                sq_i_new += r_i_new[d] * r_i_new[d]
            sum_sq = sum_sq_others + sq_i_new
            for d in range(D):
                sum_r[d] = sum_r_others[d] + r_i_new[d]

    return r_flat, e, accepted, u_calls, sum_sq, sum_r


# =============================================================================
# Wang-Landau — single-particle move
# =============================================================================

def wang_landau(
    r_init,
    e_min, e_max, n_bins,
    step_size      = 0.5,
    log_f_init     = 1.0,
    log_f_final    = 1e-4,
    flatness       = 0.90,
    check_interval = 10_000,
    seed           = 42,
):
    """
    Wang-Landau with single-particle random displacement.

    At each step:
      1. Pick particle i ~ Uniform{0, ..., N-1}
      2. Propose r[i] -> r[i] + delta,  delta ~ N(0, step_size^2) in D dims
      3. Accept with probability min(1, g(E_old)/g(E_new))
      4. Update log_g[bin(E_current)] += log_f  (WL rule, every step)
    """
    rng  = np.random.default_rng(seed)
    r    = r_init.copy()
    N, D = r.shape
    e    = U(r)

    assert e_min <= e < e_max, f"Initial energy {e:.4f} outside [{e_min}, {e_max})"

    log_g      = np.zeros(n_bins)
    hist_f     = np.zeros(n_bins, dtype=np.int64)
    hist_all   = np.zeros(n_bins, dtype=np.int64)

    dE   = (e_max - e_min) / n_bins

    log_f = float(log_f_init)
    stage = 0
    total_steps   = 0
    accepted      = 0
    mse_history   = []
    log_g_history = []

    current_bin  = _bin_index(e, e_min, dE, n_bins)
    steps_in_bin = 1

    centers = np.array([e_min + (k + 0.5) * dE for k in range(n_bins)])
    dof   = N * D
    vmask = centers > 0
    theory = (dof / 2.0 - 1.0) * np.log(centers[vmask])
    theory -= theory.min()

    t0 = time.time()
    print(f"WL-RDW (single-particle)  N={N}  D={D}")
    print(f"  E=[{e_min:.3f}, {e_max:.3f}]  bins={n_bins}  dE={dE:.4f}  step={step_size}")
    print(f"  log_f: {log_f_init} -> {log_f_final}  "
          f"flatness={flatness}  check_interval={check_interval:,}")
    print("=" * 60)

    r_flat = r.ravel().copy()
    # initialize global sum_sq and sum_r once (O(ND))
    sum_sq, sum_r = _init_sums(r_flat, N, D)

    while log_f > log_f_final:
        print(f"Stage {stage}: log_f = {log_f:.6f}")
        hist_f[:] = 0
        steps_this_stage    = 0
        accepted_this_stage = 0
        stay_lengths        = []
        u_calls_stage       = 0

        while True:
            # --- sample randomness upfront (outside Numba) ---
            i          = int(rng.integers(N))
            rand_delta = rng.standard_normal(D)
            rand_u     = rng.random()

            # --- Numba step: O(D) U evaluation ---
            r_flat, e, acc, u_c, sum_sq, sum_r = _rdw_step(
                r_flat, e, log_g, e_min, dE, n_bins,
                N, D, step_size, i, rand_delta, rand_u,
                sum_sq, sum_r)

            if acc:
                accepted            += 1
                accepted_this_stage += 1

            # --- WL update ---
            k = _bin_index(e, e_min, dE, n_bins)
            if k >= 0:
                log_g[k]   += log_f
                hist_f[k]  += 1
                hist_all[k]+= 1

            total_steps      += 1
            steps_this_stage += 1
            u_calls_stage    += u_c

            # --- avg stay in bin ---
            if k >= 0 and k != current_bin:
                stay_lengths.append(steps_in_bin)
                current_bin  = k
                steps_in_bin = 1
            else:
                steps_in_bin += 1

            # --- MSE tracking ---
            lg = log_g[vmask] - log_g[vmask].min()
            mse_history.append(float(np.mean((lg - theory) ** 2)))

            # --- flatness check ---
            if steps_this_stage % check_interval == 0:
                visited  = hist_f[hist_f > 0]
                h_min    = int(visited.min()) if len(visited) else 0
                h_mean   = float(visited.mean()) if len(visited) else 0.0
                ratio    = h_min / h_mean if h_mean > 0 else 0.0
                elapsed  = time.time() - t0
                mse_now  = mse_history[-1]
                avg_stay = float(np.mean(stay_lengths)) if stay_lengths else float(steps_in_bin)
                avg_u    = u_calls_stage / steps_this_stage
                print(f"  iter {total_steps:>10,} | Hmin {h_min} | "
                      f"ratio {ratio:.3f} | accept {accepted_this_stage/steps_this_stage:.3f} | "
                      f"avg_stay {avg_stay:.1f} | MSE {mse_now:.4f} | "
                      f"U/step {avg_u:.1f} | t {elapsed:.1f}s")

                if ratio >= flatness:
                    print(f"  >> Stage {stage} passed.")
                    log_g_history.append(log_g.copy())
                    break

        r = r_flat.reshape(N, D)
        log_f /= 2.0
        stage += 1

    total_time = time.time() - t0
    print("=" * 60)
    print(f"Done. stages={stage}  steps={total_steps:,}  time={total_time:.1f}s")

    return {
        "log_g"          : log_g,
        "log_g_history"  : log_g_history,
        "hist_all"       : hist_all,
        "mse_history"    : np.array(mse_history),
        "stay_lengths"   : np.array(stay_lengths),
        "centers"        : centers,
        "total_steps"    : total_steps,
        "total_time"     : total_time,
        "stage"          : stage,
    }


# =============================================================================
# Energy range estimation
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

def plot_results(res, N, D, step_size):
    centers   = res["centers"]
    log_g     = res["log_g"].copy();  log_g -= log_g.min()
    n         = N * D
    ref       = (n / 2.0 - 1.0) * np.log(np.maximum(centers, 1e-300))
    ref      -= ref.min()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"WL-RDW (single-particle)  N={N}  D={D}  step={step_size}  "
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

    # S(T) = ln Z(T) + <E>(T) / T
    # Z(T) = sum_k g(E_k) exp(-E_k/T) = sum_k exp(log_g_norm[k] - E_k/T)
    # Use log-sum-exp: ln Z = shift + ln(sum exp(lw - shift))
    half_n     = N * D / 2.0
    log_g_norm = res["log_g"] - res["log_g"].min()
    log_g_an   = (half_n - 1.0) * np.log(centers)
    log_g_an  -= log_g_an.min()

    e_min_plot = centers[0]
    e_max_plot = centers[-1]
    T_lo = 2 * e_min_plot / (N * D)
    T_hi = 2 * e_max_plot / (N * D)
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
    fname = f"wl_rdw_single_N{N}_D{D}_step{step_size}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # warm up Numba
    _U_flat(np.ones(4), 2, 2)

    # --- parameters ---
    N, D           = 100, 3
    N_BINS         = 100
    STEP_SIZE      = 1.0
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
        step_size      = STEP_SIZE,
        log_f_init     = LOG_F_INIT,
        log_f_final    = LOG_F_FINAL,
        flatness       = FLATNESS,
        check_interval = CHECK_INTERVAL,
        seed           = SEED,
    )

    plot_results(res, N, D, step_size=STEP_SIZE)