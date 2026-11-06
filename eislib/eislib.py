"""Shared library for the E_a-portability study.
KK consistency screening, constrained ECM fitting, Arrhenius regression, Butler-Volmer
large-signal model. All functions are exactly those used to produce the numbers in the
manuscript.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress

# ---------------- Lin-KK (Voigt chain + series R, L, C) ----------------
def linkk(fr, z, per_decade=4, ret_spec=False):
    """Linear Kramers-Kronig consistency test.
    Voigt elements with fixed log-spaced time constants spanning the measured
    band +/- 0.5 decade, plus series R, series C (1/jwC) and series L (jwL).
    Returns max residual in % of |Z| (or the residual spectra)."""
    w = 2*np.pi*np.asarray(fr, float)
    z = np.asarray(z, complex)
    M = max(int(per_decade*np.log10(fr.max()/fr.min())), 8)
    taus = 1/np.logspace(np.log10(w.min())-0.5, np.log10(w.max())+0.5, M)
    cols = [np.ones_like(w, dtype=complex)] + [1/(1+1j*w*t) for t in taus] + [1/(1j*w), 1j*w]
    A = np.array(cols).T
    Ari = np.vstack([A.real, A.imag]); b = np.concatenate([z.real, z.imag])
    x, *_ = np.linalg.lstsq(Ari, b, rcond=None)
    zf = A @ x
    rr = (z.real-zf.real)/np.abs(z); ri = (z.imag-zf.imag)/np.abs(z)
    if ret_spec: return rr*100, ri*100
    return max(np.max(np.abs(rr)), np.max(np.abs(ri)))*100

# ---------------- ECM: Rohm [+L] + Rsei||CPE1 + Rct||CPE2 + FSW ----------------
def _fsw(Rw0, u):
    """Finite-space (reflective / blocking) Warburg: Z_W = R_W0 * coth(u)/u,
    u = sqrt(j*A_W0*w), written exactly as R_W0/tanh(u)/u.

    Z_W -> R_W0*(j*A_W0*w)^-1/2 (45 deg) at high frequency and diverges as
    R_W0/(j*A_W0*w) + R_W0/3 (capacitive, blocking) as w -> 0, which is the correct
    boundary condition for intercalation into particles with no through-flux.

    At high frequency the intermediate exponentials inside tanh overflow; numpy still
    returns the correct limit tanh -> 1, so the suppressed warning is purely cosmetic
    and the arithmetic is bit-for-bit the expression used for the published tables."""
    with np.errstate(over='ignore', invalid='ignore'):
        return Rw0/np.tanh(u)/u


def zmodel(p, w, with_L=False, warburg='coth'):
    """Z = R_ohm [+ jwL] + R_SEI||CPE1 + R_ct||CPE2 + Z_W.
    warburg='coth' is the finite-space reflective element used throughout (eq. 4);
    warburg='semi' is the semi-infinite element R_W0 (j A_W0 w)^(-1/2), used only by the
    diffusion-element sensitivity variant (script 12)."""
    if with_L:
        Rohm,Rsei,A1,P1,Rct,A2,P2,Rw0,Aw0,L = p
        zL = 1j*w*L
    else:
        Rohm,Rsei,A1,P1,Rct,A2,P2,Rw0,Aw0 = p
        zL = 0
    jw = 1j*w
    u = np.sqrt(1j*Aw0*w)
    zw = _fsw(Rw0, u) if warburg == 'coth' else Rw0/u   # coth: blocking as w->0
    return Rohm + zL + Rsei/(1+Rsei*A1*jw**P1) + Rct/(1+Rct*A2*jw**P2) + zw

BOUNDS9 = ([1e-4,1e-6,1e-3,0.5,1e-6,1e-3,0.5,1e-6,1e-2],
           [1.0, 1.0, 1e5, 1.0, 1.0, 1e5, 1.0, 2.0, 1e4])

def branch_tau(p):
    """Characteristic time constants of the two R-CPE branches, tau=(R*A)^(1/P)."""
    return (p[1]*p[2])**(1.0/p[3]), (p[4]*p[5])**(1.0/p[6])

def canonicalize(p):
    """Label the branches by time constant: branch 1 (SEI) is always the faster one.
    Removes the label-swap degeneracy of the two R-CPE branches, which is otherwise
    the dominant source of non-reproducibility between fitting pipelines."""
    p = np.array(p, float)
    t1, t2 = branch_tau(p)
    if t1 > t2:
        p[[1, 2, 3, 4, 5, 6]] = p[[4, 5, 6, 1, 2, 3]]
    return p

def zero_crossing_real(fr, z):
    """Re(Z) at the zero-phase crossing, Im(Z) = 0, nearest the high-frequency end.

    The crossing is located between the two adjacent frequency points at which Im(Z)
    changes sign and Re(Z) is linearly interpolated there. Spectra that never cross
    (for example spectra truncated before the inductive branch) return None, and
    fit_ecm then uses the high-frequency minimum of Re(Z) instead."""
    fr = np.asarray(fr, float); z = np.asarray(z, complex)
    o = np.argsort(fr); zz = z[o]; im = zz.imag
    idx = [i for i in range(len(zz)-1) if im[i] == 0 or im[i]*im[i+1] < 0]
    if not idx:
        return None
    i = idx[-1]
    if im[i] == 0:
        return float(zz.real[i])
    f = im[i]/(im[i]-im[i+1])
    return float(zz.real[i] + f*(zz.real[i+1]-zz.real[i]))


def _dz_dp(p, w, with_L, warburg='coth'):
    """Analytic derivatives dZ/dp of zmodel, shape (n_params, n_freq).

    Used as the Jacobian of the fit. It agrees with central finite differences to
    ~1e-8 relative and makes each local solve about twice as fast."""
    if with_L:
        Rohm, R1, A1, P1, R2, A2, P2, Rw0, Aw0, L = p
    else:
        Rohm, R1, A1, P1, R2, A2, P2, Rw0, Aw0 = p
    jw = 1j*w; ljw = np.log(w) + 0.5j*np.pi
    out = [np.ones_like(w, dtype=complex)]                       # d/dRohm
    for R, A, P in ((R1, A1, P1), (R2, A2, P2)):
        s = jw**P; d2 = (1 + R*A*s)**2
        out += [1/d2, -R*R*s/d2, -R*R*A*s*ljw/d2]                # d/dR, d/dA, d/dP
    u = np.sqrt(1j*Aw0*w)
    if warburg == 'coth':
        big = np.real(u) > 20.0                                  # coth -> 1, csch^2 -> 0
        us = np.where(big, 1.0, u)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            cth = np.where(big, 1.0+0j, 1/np.tanh(us))
            csch2 = np.where(big, 0j, 1/np.sinh(us)**2)
        out.append(cth/u)                                        # d/dRw0
        out.append(Rw0*(-csch2/u - cth/(u*u))*u/(2*Aw0))         # d/dAw0
    else:
        out.append(1/u)                                          # semi-infinite element
        out.append(-Rw0/(2*Aw0*u))
    if with_L:
        out.append(jw)                                           # d/dL
    return np.array(out)


def initial_guesses(fr, z, p0=None, with_L=False, n_restart=4, n_broad=160, seed=0):
    """Deterministic set of starting points for fit_ecm (no run-to-run randomness).

      * p0, if given (the neighbouring solution of the visiting order);
      * a data-derived guess whose R_ohm is Re(Z) at the zero-phase crossing, plus
        `n_restart` seeded perturbations of its two branch resistances and CPE
        coefficients;
      * the same guess with R_ohm = 0.97 x min Re(Z) (the high-frequency intercept used
        by earlier releases), plus the same perturbations;
      * `n_broad` seeded starts drawn log-uniformly across the whole admissible
        region, so that every basin of the two-branch objective is visited.
    The narrow perturbations alone (n_broad = 0, the procedure of earlier releases) miss the
    lowest-cost basin in 57 of the 116 reference-corpus spectra (script 17). With
    n_broad = 160, 240 broad starts move no accepted minimum by more than 0.003 % in R_ct, and
    another seed moves R_ct by at most 0.25 % in the spectra of the state-resolved table (3.1 %
    in one 100 % SoC spectrum, where that seed stops at a higher cost); neither variant reaches
    a lower cost anywhere (out/search_convergence.csv, script 17)."""
    z = np.asarray(z, complex)
    span = max(z.real.max()-z.real.min(), 1e-3)
    r_min = z.real.min()*0.97
    r_zc = zero_crossing_real(fr, z)
    bases = ([r_zc] if r_zc is not None else []) + [r_min]
    inits = [list(p0)] if p0 is not None else []
    for r0 in bases:
        base = [r0, span*0.25, 20, 0.8, span*0.5, 200, 0.85, span*0.8, 30]
        if with_L: base = base+[1e-7]
        inits.append(base)
        rng = np.random.default_rng(seed)
        for _ in range(n_restart):
            q = np.array(base, float)
            q[1] *= 10**rng.uniform(-0.7, 0.7); q[4] *= 10**rng.uniform(-0.7, 0.7)
            q[2] *= 10**rng.uniform(-1.5, 1.5); q[5] *= 10**rng.uniform(-1.5, 1.5)
            inits.append(list(q))
    rng = np.random.default_rng(seed + 1)
    r_ref = bases[0]
    for _ in range(n_broad):
        q = [r_ref*10**rng.uniform(-0.05, 0.02),
             span*10**rng.uniform(-2, 0.3), 10**rng.uniform(-1, 4), rng.uniform(0.55, 1.0),
             span*10**rng.uniform(-2, 0.3), 10**rng.uniform(-1, 4), rng.uniform(0.55, 1.0),
             span*10**rng.uniform(-1.5, 0.5), 10**rng.uniform(-1, 3.5)]
        if with_L: q = q+[10**rng.uniform(-8, -5.5)]
        inits.append(q)
    return inits


def fit_ecm(fr, z, p0=None, with_L=False, max_nfev=8000, n_restart=4, n_broad=160, seed=0,
            weighting='modulus', warburg='coth'):
    """Constrained complex NLLS, residuals weighted by 1/|Z| (or unit-weighted).

    Deterministic multi-start with an explicit acceptance rule, so that the result does
    not depend on the order in which spectra are visited:
      1. candidate initializations from initial_guesses() (neighbouring solution,
         zero-crossing and high-frequency-intercept guesses with seeded perturbations,
         and seeded broad starts);
      2. every candidate is optimized under the physical bounds (trust-region reflective
         least squares, analytic Jacobian, Jacobian-based variable scaling);
      3. the lowest-cost solution is accepted;
      4. branch labels are canonicalized by time constant (tau_1 < tau_2).
    weighting: 'modulus' (residuals divided by |Z_i|, eq. 5) or 'unit' (unweighted);
    warburg: 'coth' (finite-space reflective, eq. 4) or 'semi' (semi-infinite).
    Returns (parameters, rms residual per normalized real/imaginary component in %, always
    modulus-weighted so that fit quality is comparable across variants).
    """
    w = 2*np.pi*np.asarray(fr, float); z = np.asarray(z, complex)
    lb, ub = [list(BOUNDS9[0]), list(BOUNDS9[1])]
    if with_L: lb, ub = lb+[0.0], ub+[1e-5]
    wt = np.abs(z) if weighting == 'modulus' else np.full(len(z), np.mean(np.abs(z)))
    inits = initial_guesses(fr, z, p0, with_L, n_restart, n_broad, seed)
    def resid(p):
        e = (zmodel(p, w, with_L, warburg)-z)/wt
        return np.concatenate([e.real, e.imag])
    def jac(p):
        D = _dz_dp(p, w, with_L, warburg)/wt
        return np.hstack([D.real, D.imag]).T
    best = None
    for pp in inits:
        try:
            r = least_squares(resid, np.clip(pp, lb, ub), jac=jac, bounds=(lb, ub),
                              max_nfev=max_nfev, x_scale='jac')
            if best is None or r.cost < best.cost: best = r
        except Exception:
            pass
    p = canonicalize(best.x)
    # reported residual is always the modulus-weighted rms, so fit quality is comparable
    # across weighting variants: 100*sqrt(cost/N) with cost = 0.5*sum(r^2), r = (Z-Zi)/|Zi|
    em = (zmodel(p, w, with_L, warburg)-z)/np.abs(z)
    rms = np.sqrt(0.5*np.sum(em.real**2+em.imag**2)/len(fr))*100
    return p, rms

# ---------------- Arrhenius ----------------
from scipy.stats import t as student_t

def arrhenius(T_kelvin, R_values):
    """Arrhenius regression of ln(1/R) on 1/T.

    Returns (Ea in kJ/mol, half-width of the 95 % confidence interval, R^2).

    The regression is ln(1/R) = -(Ea/Rg)(1/T) + ln A0, so Ea = -slope * Rg.
    The confidence half-width is CI_95 = t(0.975, k-2) * SE(slope) * Rg, where SE(slope)
    is the standard error of the least-squares slope and the multiplier is the two-sided
    95 % Student-t quantile on k-2 degrees of freedom (slope and intercept are estimated).
    It is computed exactly (scipy.stats.t.ppf), e.g. 12.706 for k = 3, 2.447 for k = 8,
    2.306 for k = 10 and 2.262 for k = 11. Fewer than three setpoints leave no residual
    degrees of freedom, so NaN is returned for the interval.
    """
    T_kelvin = np.asarray(T_kelvin, float)
    R_values = np.asarray(R_values, float)
    reg = linregress(1/T_kelvin, np.log(1/R_values))
    n = len(R_values)
    Ea = -reg.slope*RG/1000
    if n < 3:
        return Ea, float('nan'), reg.rvalue**2
    ci = reg.stderr*RG/1000*float(student_t.ppf(0.975, n-2))
    return Ea, ci, reg.rvalue**2

# ---------------- Butler-Volmer large-signal ----------------
F_CONST, RG = 96485.0, 8.314
def bv_ratio(I_peak, I0, T):
    """R_ct(measured, fundamental)/R_ct(true) for a sinusoidal galvanostatic drive
    I(t) = I_peak sin(wt), symmetric one-electron Butler-Volmer kinetics, kinetics only.

    I_peak and I0 are cell-level currents in amperes (I0 = i0*A_e is the lumped exchange
    current, eqs. 9-12 of the manuscript); only their ratio enters, so the electroactive
    area cancels. eta(t) = (2RgT/F) asinh(I/(2 I0)); the measured resistance is the first
    Fourier coefficient of eta divided by I_peak; R_ct(true) = RgT/(F I0)."""
    t = np.linspace(0, 1, 2001)[:-1]
    i = I_peak*np.sin(2*np.pi*t)
    eta = (2*RG*T/F_CONST)*np.arcsinh(i/(2*I0))
    a1 = 2*np.mean(eta*np.sin(2*np.pi*t))
    return (a1/I_peak)/(RG*T/(F_CONST*I0))


# ---------------- Reference corpus: retained spectra ----------------
def unisa_groups(specs):
    """Triplicate groups of the parsed reference corpus (three repeats per state)."""
    return [specs[i:i+3] for i in range(0, len(specs), 3)]


def load_selected_spectra(out_dir, ridx=None):
    """Triplicate-median spectra retained by script 02 for the state-resolved regression.

    Reads out/unisa_arrhenius_input.csv (one row per state and temperature, written by
    02_fit_unisa.py) and out/unisa_parsed.pkl, and returns a list of dicts with keys
    T, gi, ridx, ocv, soc, fr, z sorted by state and temperature. `ridx` restricts the
    list to one state (5 = the mid state used throughout the manuscript)."""
    import os, pickle
    import pandas as pd
    sel = pd.read_csv(os.path.join(out_dir, 'unisa_arrhenius_input.csv'))
    alldata = pickle.load(open(os.path.join(out_dir, 'unisa_parsed.pkl'), 'rb'))
    if ridx is not None:
        sel = sel[sel['ridx'] == ridx]
    out = []
    for _, r in sel.sort_values(['ridx', 'T_C']).iterrows():
        T = int(round(r['T_C'])); g = unisa_groups(alldata[T])[int(r['gi'])]
        out.append(dict(T=T, gi=int(r['gi']), ridx=int(r['ridx']), ocv=float(r['OCV_V']),
                        soc=float(r['SOC_pct']) if 'SOC_pct' in r else float('nan'),
                        fr=np.asarray(g[0][1], float),
                        z=np.median(np.array([s[2] for s in g]), axis=0)))
    return out
