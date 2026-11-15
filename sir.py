# SIR epidemic model: vectorised RK4 integration and a Poisson likelihood surface.
import numpy as np


def sir_incidence(beta, gamma, N=100_000, I0=10, days=120, steps_per_day=20):
    """Daily new infections. beta/gamma can be arrays (evaluated in parallel)."""
    beta, gamma = np.broadcast_arrays(np.asarray(beta, float), np.asarray(gamma, float))
    S = np.full(beta.shape, N - I0, dtype=float)
    I = np.full(beta.shape, float(I0))
    h = 1.0 / steps_per_day

    def deriv(S, I):
        inf = beta * S * I / N
        return -inf, inf - gamma * I

    out = np.empty((days,) + beta.shape)
    for d in range(days):
        s_start = S.copy()
        for _ in range(steps_per_day):
            k1 = deriv(S, I)
            k2 = deriv(S + h / 2 * k1[0], I + h / 2 * k1[1])
            k3 = deriv(S + h / 2 * k2[0], I + h / 2 * k2[1])
            k4 = deriv(S + h * k3[0], I + h * k3[1])
            S = S + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            I = I + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out[d] = s_start - S
    return out


def poisson_loglik(cases, expected):
    expected = np.maximum(expected, 1e-9)
    cases = np.asarray(cases, float).reshape((-1,) + (1,) * (expected.ndim - 1))
    return np.sum(cases * np.log(expected) - expected, axis=0)


def likelihood_surface(cases, betas, gammas, reporting=0.3, **kw):
    B, G = np.meshgrid(betas, gammas, indexing="ij")
    inc = sir_incidence(B, G, days=len(cases), **kw)
    return B, G, poisson_loglik(cases, reporting * inc)


def r0_interval(B, G, ll, drop=1.92):
    # Approximate 95% region: grid points within 1.92 log-likelihood units of the best.
    r0 = B / G
    keep = ll >= ll.max() - drop
    best = np.unravel_index(np.argmax(ll), ll.shape)
    return float(r0[best]), float(r0[keep].min()), float(r0[keep].max())
