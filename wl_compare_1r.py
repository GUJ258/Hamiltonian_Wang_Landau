"""
WL-RDW vs WL-DHMC Comparison  (harmonic trap + 1/r pairwise)
=============================================================
4 plots:
  1. Density of States  (final log g, both methods)
  2. Convergence: L2 residual of log g per stage vs final log g
  3. f-Reduction Timeline: steps per stage
  4. Per-step wall-clock cost (us/step per stage)
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time


# =============================================================================
# Shared: potential  O(N^2 D)
# =============================================================================

@njit(cache=True, parallel=True)
def _U_flat(r_flat, N, D):
    trap = 0.0
    for i in range(N):
        for d in range(D):
            v = r_flat[i * D + d]
            trap += v * v
    interact = 0.0
    for i in prange(N):
        local = 0.0
        for j in range(i + 1, N):
            dist2 = 0.0
            for d in range(D):
                diff   = r_flat[i * D + d] - r_flat[j * D + d]
                dist2 += diff * diff
            local += 1.0 / np.sqrt(dist2)
        interact += local
    return trap + interact / (2.0 * N)


def U(r):
    return _U_flat(r.ravel(), r.shape[0], r.shape[1])


# =============================================================================
# Shared: gradient of U w.r.t. particle i  O(N D)
# =============================================================================

@njit(cache=True)
def _grad_U_particle(r_flat, N, D, i):
    g = np.empty(D)
    for d in range(D):
        g[d] = 2.0 * r_flat[i * D + d]
    for j in range(N):
        if j == i:
            continue
        dist2 = 0.0
        for d in range(D):
            diff   = r_flat[i * D + d] - r_flat[j * D + d]
            dist2 += diff * diff
        dist3 = dist2 * np.sqrt(dist2)
        inv_dist3_2N = 1.0 / (dist3 * 2.0 * N)
        for d in range(D):
            g[d] -= (r_flat[i * D + d] - r_flat[j * D + d]) * inv_dist3_2N
    return g


# =============================================================================
# Shared: energy difference when particle i moves  O(N D)
# =============================================================================

@njit(cache=True)
def _delta_U(r_flat, i_move, r_i_new, N, D):
    sq_old = 0.0
    sq_new = 0.0
    for d in range(D):
        v_old   = r_flat[i_move * D + d]
        v_new   = r_i_new[d]
        sq_old += v_old * v_old
        sq_new += v_new * v_new
    d_trap = sq_new - sq_old

    d_interact = 0.0
    inv_2N = 0.5 / N
    for j in range(N):
        if j == i_move:
            continue
        dist2_old = 0.0
        dist2_new = 0.0
        for d in range(D):
            vj        = r_flat[j * D + d]
            diff_old  = r_flat[i_move * D + d] - vj
            diff_new  = r_i_new[d]              - vj
            dist2_old += diff_old * diff_old
            dist2_new += diff_new * diff_new
        d_interact += 1.0 / np.sqrt(dist2_new) - 1.0 / np.sqrt(dist2_old)

    return d_trap + d_interact * inv_2N


# =============================================================================
# Shared: bin helper
# =============================================================================

@njit(cache=True, inline='always')
def _bin_index(E, E_min, dE, n_bins):
    if E < E_min or E >= E_min + dE * n_bins:
        return -1
    k = int((E - E_min) / dE)
    return min(k, n_bins - 1)


# =============================================================================
# Shared: energy range + initial config
# =============================================================================

def estimate_energy_range(N, D, low_q=0.005, high_q=0.995,
                           n_pilot=5000, seed=0):
    rng    = np.random.default_rng(seed)
    pilots = np.array([U(rng.standard_normal((N, D))) for _ in range(n_pilot)])
    e_min  = float(np.quantile(pilots, low_q))
    e_max  = float(np.quantile(pilots, high_q))
    print(f"Pilot energy range: [{e_min:.3f}, {e_max:.3f}]")
    return e_min, e_max


def find_initial_config(N, D, e_min, e_max, n_tries=100_000, seed=1):
    rng = np.random.default_rng(seed)
    for trial in range(n_tries):
        scale = 0.1 + 2.0 * (trial / n_tries)
        r     = rng.standard_normal((N, D)) * scale
        if e_min <= U(r) < e_max:
            print(f"Initial config found: trial={trial+1}  U={U(r):.4f}")
            return r
    raise RuntimeError("Could not find initial config in range.")


# =============================================================================
# WL-RDW
# =============================================================================

def wang_landau_rdw(r_init, e_min, e_max, n_bins,
                    step_size, log_f_init, log_f_final,
                    flatness, check_interval, seed):
    rng  = np.random.default_rng(seed)
    r    = r_init.copy().astype(np.float64)
    N, D = r.shape
    e    = U(r)
    dE   = (e_max - e_min) / n_bins

    log_g    = np.zeros(n_bins)
    hist_f   = np.zeros(n_bins, dtype=np.int64)
    hist_all = np.zeros(n_bins, dtype=np.int64)
    centers  = e_min + (np.arange(n_bins) + 0.5) * dE

    log_f        = float(log_f_init)
    stage        = 0
    total_steps  = 0
    log_g_history= []
    check_steps  = []   # cumulative step count at each check
    stage_steps  = []
    stage_times  = []

    r_flat = r.ravel().copy()
    t0 = time.time()
    print(f"\n{'='*60}\nWL-RDW  N={N}  D={D}  step={step_size}")
    print(f"  E=[{e_min:.3f},{e_max:.3f}]  bins={n_bins}")
    print(f"  log_f: {log_f_init} -> {log_f_final}  flatness={flatness}")
    print(f"{'='*60}")

    while log_f > log_f_final:
        print(f"Stage {stage}  log_f={log_f:.6f}")
        hist_f[:] = 0
        steps_stage = accepted_stage = 0
        t_stage = time.time()

        while True:
            i_move  = int(rng.integers(N))
            delta   = rng.standard_normal(D) * step_size
            r_i_old = r_flat[i_move * D : i_move * D + D].copy()
            r_i_new = (r_i_old + delta).astype(np.float64)

            dU    = _delta_U(r_flat, i_move, r_i_new, N, D)
            e_new = e + dU

            k_old = _bin_index(e,     e_min, dE, n_bins)
            k_new = _bin_index(e_new, e_min, dE, n_bins)

            if k_new >= 0:
                if np.log(rng.random()) < log_g[k_old] - log_g[k_new]:
                    r_flat[i_move * D : i_move * D + D] = r_i_new
                    e               = e_new
                    accepted_stage += 1

            k = _bin_index(e, e_min, dE, n_bins)
            if k >= 0:
                log_g[k]   += log_f
                hist_f[k]  += 1
                hist_all[k]+= 1

            total_steps  += 1
            steps_stage  += 1

            if steps_stage % check_interval == 0:
                visited = hist_f[hist_f > 0]
                h_min   = int(visited.min()) if len(visited) else 0
                h_mean  = float(visited.mean()) if len(visited) else 1.0
                ratio   = h_min / h_mean if h_mean > 0 else 0.0
                print(f"  iter {total_steps:>9,} | ratio {ratio:.3f} | "
                      f"accept {accepted_stage/steps_stage:.3f} | "
                      f"t {time.time()-t0:.1f}s")
                log_g_history.append(log_g.copy())
                check_steps.append(total_steps)
                if ratio >= flatness:
                    print(f"  >> Stage {stage} done.")
                    break

        stage_steps.append(steps_stage)
        stage_times.append(time.time() - t_stage)
        log_f /= 2.0
        stage += 1

    total_time = time.time() - t0
    print(f"Done.  stages={stage}  steps={total_steps:,}  time={total_time:.1f}s")
    return {
        "log_g"        : log_g,
        "log_g_history": log_g_history,
        "check_steps"  : check_steps,
        "hist_all"     : hist_all,
        "centers"      : centers,
        "total_steps"  : total_steps,
        "total_time"   : total_time,
        "stage"        : stage,
        "stage_steps"  : stage_steps,
        "stage_times"  : stage_times,
        "u_per_step"   : 1.0,
        "label"        : f"RDW  step={step_size}",
    }


# =============================================================================
# DHMC v3 internals
# Key optimisation: inv_dist_cache stores 1/|r_i - r_j| for current particle i.
# _delta_U_inv reads old side from cache (O(1) per j) and fills out_buf with
# new inverse distances. After a committed move, swap out_buf -> cache in O(N).
# Bisection uses a throwaway inv_bscratch so the cache is never polluted.
# =============================================================================

@njit(cache=True)
def _build_inv_dist_cache(r_flat, i, N, D):
    cache  = np.empty(N)
    i_base = i * D
    for j in range(N):
        if j == i:
            cache[j] = 0.0
            continue
        d2     = 0.0
        j_base = j * D
        for d in range(D):
            diff = r_flat[i_base + d] - r_flat[j_base + d]
            d2  += diff * diff
        cache[j] = 1.0 / np.sqrt(d2)
    return cache


@njit(cache=True)
def _delta_U_inv(r_flat, i_move, r_i_new, N, D, inv_dist_cache, out_buf):
    sq_old = 0.0
    sq_new = 0.0
    i_base = i_move * D
    for d in range(D):
        v_old   = r_flat[i_base + d]
        v_new   = r_i_new[d]
        sq_old += v_old * v_old
        sq_new += v_new * v_new
    d_trap = sq_new - sq_old

    d_interact = 0.0
    inv_2N     = 0.5 / N
    for j in range(N):
        if j == i_move:
            out_buf[j] = 0.0
            continue
        inv_old = inv_dist_cache[j]
        d2_new  = 0.0
        j_base  = j * D
        for d in range(D):
            diff    = r_i_new[d] - r_flat[j_base + d]
            d2_new += diff * diff
        inv_new    = 1.0 / np.sqrt(d2_new)
        out_buf[j] = inv_new
        d_interact += inv_new - inv_old

    return d_trap + d_interact * inv_2N


@njit(cache=True, inline='always')
def _reflect_refract(p_i, grad_i, delta_log_g, out_of_range, D):
    gnorm_sq = 0.0
    for d in range(D):
        gnorm_sq += grad_i[d] * grad_i[d]
    if gnorm_sq < 1e-28:
        return p_i.copy(), False
    gnorm     = np.sqrt(gnorm_sq)
    inv_gnorm = 1.0 / gnorm
    p_n = 0.0
    for d in range(D):
        p_n += p_i[d] * grad_i[d]
    p_n *= inv_gnorm

    p_new = p_i.copy()
    if out_of_range:
        scale = -2.0 * p_n * inv_gnorm
        for d in range(D):
            p_new[d] += scale * grad_i[d]
        return p_new, False

    discriminant = p_n * p_n - 2.0 * delta_log_g
    if discriminant >= 0.0:
        p_n_new = np.sign(p_n) * np.sqrt(discriminant)
        scale   = (p_n_new - p_n) * inv_gnorm
        for d in range(D):
            p_new[d] += scale * grad_i[d]
        return p_new, True
    else:
        scale = -2.0 * p_n * inv_gnorm
        for d in range(D):
            p_new[d] += scale * grad_i[d]
        return p_new, False


@njit(cache=True)
def _dhmc_mh_step(r_flat, particle_idx, N, D,
                   log_g, E_min, dE, n_bins, n_steps, step_size,
                   U_in, p_i, log_u,
                   max_crossings=50, bisect_tol=1e-10, max_bisect=40):
    i      = particle_idx
    i_base = i * D
    E_max  = E_min + dE * n_bins

    inv_cache    = _build_inv_dist_cache(r_flat, i, N, D)
    inv_new_buf  = np.empty(N)
    inv_bscratch = np.empty(N)

    r_traj       = r_flat.copy()
    p            = p_i.copy()
    U_curr       = U_in
    r_i_bscratch = np.empty(D)
    r_i_new      = np.empty(D)
    r_i_nudged   = np.empty(D)

    total_u_calls = 0
    total_bisect  = 0
    valid         = True

    KE_old = 0.0
    for d in range(D):
        KE_old += p_i[d] * p_i[d]
    KE_old *= 0.5
    k_old   = _bin_index(U_curr, E_min, dE, n_bins)
    H_old   = KE_old + (log_g[k_old] if k_old >= 0 else 1e300)

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
            E_lo     = E_min + k * dE
            E_hi     = E_lo  + dE
            E_lo_eff = E_lo if k > 0         else E_min
            E_hi_eff = E_hi if k < n_bins-1  else E_max

            # Probe endpoint
            for d in range(D):
                r_i_bscratch[d] = r_traj[i_base + d] + t_rem * p[d]
            U_end = U_curr + _delta_U_inv(
                r_traj, i, r_i_bscratch, N, D, inv_cache, inv_bscratch)
            total_u_calls += 1

            crossed_lo = U_end < E_lo_eff
            crossed_hi = U_end > E_hi_eff
            t_star = -1.0
            hit_lo = False; hit_hi = False

            if crossed_lo or crossed_hi:
                hit_lo = crossed_lo; hit_hi = crossed_hi
                t_lo_b, t_hi_b = 0.0, t_rem
                for _bi in range(max_bisect):
                    t_mid = 0.5 * (t_lo_b + t_hi_b)
                    for d in range(D):
                        r_i_bscratch[d] = r_traj[i_base + d] + t_mid * p[d]
                    U_mid = U_curr + _delta_U_inv(
                        r_traj, i, r_i_bscratch, N, D, inv_cache, inv_bscratch)
                    total_u_calls += 1
                    total_bisect  += 1
                    if crossed_hi:
                        if U_mid > E_hi_eff: t_hi_b = t_mid
                        else:                t_lo_b = t_mid
                    else:
                        if U_mid < E_lo_eff: t_hi_b = t_mid
                        else:                t_lo_b = t_mid
                    if t_hi_b - t_lo_b < bisect_tol:
                        break
                t_star = 0.5 * (t_lo_b + t_hi_b)

            if t_star < 0.0:
                for d in range(D):
                    r_i_new[d] = r_traj[i_base + d] + t_rem * p[d]
                U_curr += _delta_U_inv(r_traj, i, r_i_new, N, D, inv_cache, inv_new_buf)
                total_u_calls += 1
                for j in range(N):
                    inv_cache[j] = inv_new_buf[j]
                for d in range(D):
                    r_traj[i_base + d] = r_i_new[d]
                t_rem = 0.0
                break

            for d in range(D):
                r_i_new[d] = r_traj[i_base + d] + t_star * p[d]
            U_curr += _delta_U_inv(r_traj, i, r_i_new, N, D, inv_cache, inv_new_buf)
            total_u_calls += 1
            for j in range(N):
                inv_cache[j] = inv_new_buf[j]
            for d in range(D):
                r_traj[i_base + d] = r_i_new[d]

            k_new        = k + 1 if hit_hi else k - 1
            out_of_range = (hit_hi and k == n_bins-1) or (hit_lo and k == 0)
            delta_lg     = 0.0 if out_of_range else (log_g[k_new] - log_g[k])
            grad_i       = _grad_U_particle(r_traj, N, D, i)
            p, refracted = _reflect_refract(p, grad_i, delta_lg, out_of_range, D)

            gnorm_sq = 0.0
            for d in range(D):
                gnorm_sq += grad_i[d] * grad_i[d]
            if gnorm_sq > 1e-28:
                gnorm = np.sqrt(gnorm_sq)
                bsign = 1.0 if hit_hi else -1.0
                nudge = (1e-8 * bsign / gnorm) if (refracted and not out_of_range) \
                        else (-1e-8 * bsign / gnorm)
                for d in range(D):
                    r_i_nudged[d] = r_traj[i_base + d] + nudge * grad_i[d]
                U_curr += _delta_U_inv(r_traj, i, r_i_nudged, N, D, inv_cache, inv_new_buf)
                total_u_calls += 1
                for j in range(N):
                    inv_cache[j] = inv_new_buf[j]
                for d in range(D):
                    r_traj[i_base + d] = r_i_nudged[d]

            t_rem -= t_star
            crossings_this_step += 1

        if not valid:
            break

    KE_new = 0.0
    for d in range(D):
        KE_new += p[d] * p[d]
    KE_new *= 0.5
    k_new_bin = _bin_index(U_curr, E_min, dE, n_bins)
    H_new     = KE_new + (log_g[k_new_bin] if k_new_bin >= 0 else 1e300)

    if not valid or H_new >= 1e299:
        return r_flat, False, U_in, total_u_calls, total_bisect
    if log_u < H_old - H_new:
        return r_traj, True, U_curr, total_u_calls, total_bisect
    return r_flat, False, U_in, total_u_calls, total_bisect


@njit(cache=True)
def _wl_inner_chunk(r_flat, N, D, log_g, hist_f, hist_all,
                    E_min, dE, n_bins, n_steps, step_size,
                    U_curr, particle_ids, momenta, log_uniforms, log_f,
                    max_crossings=50):
    chunk_size = particle_ids.shape[0]
    accepted   = 0
    u_calls    = 0
    bisect_tot = 0

    for s in range(chunk_size):
        i   = particle_ids[s]
        p_i = momenta[s]
        lu  = log_uniforms[s]

        r_flat, acc, U_curr, uc, bc = _dhmc_mh_step(
            r_flat, i, N, D, log_g, E_min, dE, n_bins, n_steps, step_size,
            U_curr, p_i, lu, max_crossings)

        accepted   += int(acc)
        u_calls    += uc
        bisect_tot += bc

        k = _bin_index(U_curr, E_min, dE, n_bins)
        if k >= 0:
            log_g[k]    += log_f
            hist_f[k]   += 1
            hist_all[k] += 1

    return r_flat, U_curr, accepted, u_calls, bisect_tot


# =============================================================================
# WL-DHMC
# =============================================================================

def wang_landau_dhmc(r_init, e_min, e_max, n_bins,
                     n_steps, step_size, momentum_sigma,
                     log_f_init, log_f_final,
                     flatness, check_interval, seed):
    rng  = np.random.default_rng(seed)
    r    = r_init.copy().astype(np.float64)
    N, D = r.shape
    dE   = (e_max - e_min) / n_bins

    log_g    = np.zeros(n_bins)
    hist_f   = np.zeros(n_bins, dtype=np.int64)
    hist_all = np.zeros(n_bins, dtype=np.int64)
    centers  = e_min + (np.arange(n_bins) + 0.5) * dE

    log_f         = float(log_f_init)
    stage         = 0
    total_steps   = 0
    log_g_history = []
    check_steps   = []   # cumulative step count at each check
    stage_steps   = []
    stage_times   = []
    u_per_step_all= []

    r_flat = r.ravel().copy()
    U_curr = _U_flat(r_flat, N, D)

    t0 = time.time()
    print(f"\n{'='*60}\nWL-DHMC v3  N={N}  D={D}  L={n_steps}  "
          f"step={step_size}  sigma={momentum_sigma}")
    print(f"  E=[{e_min:.3f},{e_max:.3f}]  bins={n_bins}")
    print(f"  log_f: {log_f_init} -> {log_f_final}  flatness={flatness}")
    print(f"{'='*60}")

    while log_f > log_f_final:
        print(f"Stage {stage}  log_f={log_f:.6f}")
        hist_f[:] = 0
        steps_stage = accepted_stage = u_calls_stage = bisect_stage = 0
        t_stage = time.time()

        while True:
            particle_ids = rng.integers(0, N, size=check_interval).astype(np.int64)
            momenta      = rng.normal(0.0, momentum_sigma,
                                      size=(check_interval, D))
            log_uniforms = np.log(rng.uniform(size=check_interval))

            r_flat, U_curr, acc, uc, bc = _wl_inner_chunk(
                r_flat, N, D, log_g, hist_f, hist_all,
                e_min, dE, n_bins, n_steps, step_size,
                U_curr, particle_ids, momenta, log_uniforms, log_f)

            accepted_stage  += acc
            steps_stage     += check_interval
            total_steps     += check_interval
            u_calls_stage   += uc
            bisect_stage    += bc

            visited = hist_f[hist_f > 0]
            h_min   = int(visited.min())    if len(visited) else 0
            h_mean  = float(visited.mean()) if len(visited) else 1.0
            ratio   = h_min / h_mean if h_mean > 0 else 0.0
            print(f"  iter {total_steps:>9,} | ratio {ratio:.3f} | "
                  f"accept {accepted_stage/steps_stage:.3f} | "
                  f"U/step {u_calls_stage/steps_stage:.1f} | "
                  f"t {time.time()-t0:.1f}s")
            log_g_history.append(log_g.copy())
            check_steps.append(total_steps)
            if ratio >= flatness:
                print(f"  >> Stage {stage} done.")
                break

        stage_steps.append(steps_stage)
        stage_times.append(time.time() - t_stage)
        u_per_step_all.append(u_calls_stage / steps_stage)
        log_f /= 2.0
        stage += 1

    total_time = time.time() - t0
    print(f"Done.  stages={stage}  steps={total_steps:,}  time={total_time:.1f}s")
    return {
        "log_g"        : log_g,
        "log_g_history": log_g_history,
        "check_steps"  : check_steps,
        "hist_all"     : hist_all,
        "centers"      : centers,
        "total_steps"  : total_steps,
        "total_time"   : total_time,
        "stage"        : stage,
        "stage_steps"  : stage_steps,
        "stage_times"  : stage_times,
        "u_per_step"   : float(np.mean(u_per_step_all)),
        "label"        : f"DHMC v3  L={n_steps}  step={step_size}",
    }


# =============================================================================
# Convergence metric: L2 residual vs final log g  (option B)
# =============================================================================

def stage_residuals(log_g_history, log_g_final):
    """
    For each stage k, compute
      residual_k = || (log_g_k - min) - (log_g_final - min) ||_2
    normalised by number of bins so it's scale-independent.
    """
    ref = log_g_final - log_g_final.min()
    residuals = []
    for lg in log_g_history:
        lg_norm = lg - lg.min()
        residuals.append(float(np.sqrt(np.mean((lg_norm - ref) ** 2))))
    return residuals


# =============================================================================
# Comparison plot
# =============================================================================

def plot_comparison(res_rdw, res_dhmc, N, D):
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"WL Comparison: RDW vs DHMC  |  N={N}  D={D}", fontsize=11)

    centers = res_rdw["centers"]
    dE      = centers[1] - centers[0]

    # 1. DOS
    ax = axes[0]
    for res, color in [(res_rdw, "#378ADD"), (res_dhmc, "#D85A30")]:
        lg = res["log_g"].copy(); lg -= lg.min()
        ax.plot(res["centers"], lg, lw=1.5, color=color, label=res["label"])
    ax.set(xlabel="E", ylabel="log g(E)  [normalised]", title="Density of states")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Convergence: L2 residual vs cumulative steps (one point per check_interval)
    ax = axes[1]
    for res, color, marker in [(res_rdw, "#378ADD", "o"), (res_dhmc, "#D85A30", "s")]:
        resid = stage_residuals(res["log_g_history"], res["log_g"])
        xs    = res["check_steps"]
        label = (f"{res['label']}  "
                 f"({res['total_steps']:,} steps, {res['total_time']:.0f}s)")
        ax.semilogy(xs, resid, color=color, lw=1.2, label=label)
    ax.set(xlabel="Cumulative steps", ylabel="L2 residual vs final log g",
           title="Convergence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Steps per stage
    ax = axes[2]
    for res, color, marker in [(res_rdw, "#378ADD", "o"), (res_dhmc, "#D85A30", "s")]:
        ss = res["stage_steps"]
        ax.plot(range(1, len(ss) + 1), ss, color=color, marker=marker,
                lw=1.5, ms=5, label=f"{res['label']}  ({len(ss)} stages)")
    ax.set(xlabel="Stage", ylabel="Steps per stage",
           title="f-Reduction timeline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Wall-clock cost per step  (y-axis starts at 0)
    ax = axes[3]
    all_t = []
    for res, color, marker in [(res_rdw, "#378ADD", "o"), (res_dhmc, "#D85A30", "s")]:
        ss = res["stage_steps"]
        st = res["stage_times"]
        t_per_step = [st[j] / ss[j] * 1e6 for j in range(len(ss))]
        all_t.extend(t_per_step)
        ax.plot(range(1, len(ss) + 1), t_per_step, color=color, marker=marker,
                lw=1.5, ms=5,
                label=f"{res['label']}  U/step={res['u_per_step']:.1f}")
    ax.set_ylim(bottom=0, top=max(all_t) * 1.1)
    ax.set(xlabel="Stage", ylabel="Time per step (us)",
           title="Per-step wall-clock cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"wl_compare_1r_N{N}_D{D}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Compiling Numba kernels...")
    _rw  = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5], dtype=np.float64)
    _pw  = np.array([0.1, 0.2], dtype=np.float64)
    _lg  = np.zeros(10)
    _U_flat(_rw, 3, 2)
    _grad_U_particle(_rw, 3, 2, 0)
    _ic  = _build_inv_dist_cache(_rw, 0, 3, 2)
    _ob  = np.empty(3)
    _rn  = np.array([0.05, 0.05])
    _delta_U_inv(_rw, 0, _rn, 3, 2, _ic, _ob)
    _dhmc_mh_step(_rw, 0, 3, 2, _lg, 0.0, 1.0, 10, 5, 0.1, 0.5, _pw, -1.0)
    _pids = np.array([0, 1], dtype=np.int64)
    _mom  = np.random.randn(2, 2)
    _lus  = np.log(np.random.uniform(size=2))
    _hf   = np.zeros(10, dtype=np.int64)
    _ha   = np.zeros(10, dtype=np.int64)
    _wl_inner_chunk(_rw, 3, 2, _lg, _hf, _ha, 0.0, 1.0, 10, 5, 0.1,
                    0.5, _pids, _mom, _lus, 1.0)
    print("Done.\n")

    N, D           = 100, 3
    N_BINS         = 100
    LOG_F_INIT     = 1.0
    LOG_F_FINAL    = 1e-4
    FLATNESS       = 0.90
    CHECK_INTERVAL = 10_000
    SEED           = 42

    RDW_STEP       = 0.5

    DHMC_N_STEPS   = 10
    DHMC_STEP      = 0.1
    DHMC_MOM_SIGMA = 1.0

    E_MIN, E_MAX = estimate_energy_range(N, D, seed=0)
    r0           = find_initial_config(N, D, E_MIN, E_MAX, seed=1)

    res_rdw = wang_landau_rdw(
        r0,
        e_min          = E_MIN,
        e_max          = E_MAX,
        n_bins         = N_BINS,
        step_size      = RDW_STEP,
        log_f_init     = LOG_F_INIT,
        log_f_final    = LOG_F_FINAL,
        flatness       = FLATNESS,
        check_interval = CHECK_INTERVAL,
        seed           = SEED,
    )

    res_dhmc = wang_landau_dhmc(
        r0,
        e_min          = E_MIN,
        e_max          = E_MAX,
        n_bins         = N_BINS,
        n_steps        = DHMC_N_STEPS,
        step_size      = DHMC_STEP,
        momentum_sigma = DHMC_MOM_SIGMA,
        log_f_init     = LOG_F_INIT,
        log_f_final    = LOG_F_FINAL,
        flatness       = FLATNESS,
        check_interval = CHECK_INTERVAL,
        seed           = SEED,
    )

    plot_comparison(res_rdw, res_dhmc, N, D)