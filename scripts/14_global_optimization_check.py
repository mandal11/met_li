"""Does a global optimizer find better fits than the released deterministic search? (script 15)

Motivation
----------
Script 02 fits each spectrum with a deterministic multi-start (eislib.fit_ecm): the
neighbouring solution of the visiting order, zero-crossing and high-frequency-intercept
guesses with seeded narrow perturbations, and seeded broad starts across the admissible
region; the lowest cost is accepted. A fair objection is: why not simply run a global
optimizer, then polish locally?

This script answers it empirically on the OCV-matched mid state of the reference corpus
(the Table 3 row), replacing the whole fitting stage with differential evolution followed
by a local least-squares polish. Every spectrum is fitted INDEPENDENTLY: no chaining, no
data-derived first guess, no information shared between temperatures. Two
parameterizations are compared, because the answer depends on this choice:

  raw   : differential evolution over the published bounds as written. Several parameters
          span up to eight decades and are sampled LINEARLY, so the region where the
          physics lives is barely explored.
  log   : the same search with the resistive and capacitive parameters searched in log10
          space, exponents and inductance linearly.

For every method the per-spectrum residual is compared with the released search, so the
table states how many spectra each method fits better or worse than the reference.
"""
import numpy as np, pandas as pd, pickle, os, sys, time
from scipy.optimize import differential_evolution, least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import zmodel, BOUNDS9, canonicalize, arrhenius, fit_ecm, load_selected_spectra

STATE = 5                      # OCV-matched mid state, as used by scripts 12 and 14
SEEDS = [0, 7]
LB = np.array(list(BOUNDS9[0]) + [1e-12])
UB = np.array(list(BOUNDS9[1]) + [1e-5])
LOG = np.array([True, True, True, False, True, True, False, True, True, True])


def series_state(alldata, state=STATE):
    def groups(s): return [s[i:i + 3] for i in range(0, len(s), 3)]
    ref = [np.median([x[3] for x in g]) for g in groups(alldata[25])]
    out = {}
    for T in sorted(alldata):
        for g in groups(alldata[T]):
            o = np.median([x[3] for x in g])
            if int(np.argmin([abs(o - r) for r in ref])) == state:
                out[T] = (g[0][1], np.median(np.array([x[2] for x in g]), axis=0))
    return out


def _polish(p0, fr, z):
    w = 2 * np.pi * fr
    def resid(p):
        e = (zmodel(p, w, with_L=True) - z) / np.abs(z)
        return np.concatenate([e.real, e.imag])
    r = least_squares(resid, np.clip(p0, LB, UB), bounds=(LB, UB), max_nfev=8000)
    return canonicalize(r.x), np.sqrt(r.cost / len(fr)) * 100


def global_fit(fr, z, seed, log_space):
    w = 2 * np.pi * fr
    if log_space:
        lo = np.where(LOG, np.log10(LB), LB); hi = np.where(LOG, np.log10(UB), UB)
        def to_lin(q):
            p = np.array(q, float); p[LOG] = 10 ** p[LOG]; return p
        def cost(q):
            e = (zmodel(to_lin(q), w, with_L=True) - z) / np.abs(z)
            return 0.5 * np.sum(e.real ** 2 + e.imag ** 2)
        r = differential_evolution(cost, list(zip(lo, hi)), seed=seed, maxiter=250,
                                   popsize=20, tol=1e-10, polish=False, init='sobol')
        return _polish(to_lin(r.x), fr, z)
    def cost_lin(p):
        e = (zmodel(p, w, with_L=True) - z) / np.abs(z)
        return 0.5 * np.sum(e.real ** 2 + e.imag ** 2)
    r = differential_evolution(cost_lin, list(zip(LB, UB)), seed=seed, maxiter=180,
                               popsize=18, tol=1e-8, polish=False, init='sobol')
    return _polish(r.x, fr, z)


def ea_of(series, fits):
    Ts = sorted(series); TK = [T + 273.15 for T in Ts]
    return {k: arrhenius(TK, [fits[T][0][i] for T in Ts])[0]
            for k, i in [('Rohm', 0), ('Rsei', 1), ('Rct', 4)]}


if __name__ == '__main__':
    need = [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_arrhenius_input.csv')]
    if not all(os.path.exists(f) for f in need):
        print('[abort] run 01_parse_unisa.py and 02_fit_unisa.py first -> '
              'global_optimization_check.csv NOT written.')
        sys.exit(0)
    series = {s['T']: (s['fr'], s['z']) for s in load_selected_spectra(OUT, ridx=STATE)}
    Ts = sorted(series)

    prev = None; ref_fits = {}
    for T in sorted(series, reverse=True):                      # warm -> cold, as script 02
        fr, z = series[T]
        p, rms = fit_ecm(fr, z, p0=prev, with_L=True); prev = p.copy()
        ref_fits[T] = (p, rms)
    ref_ea = ea_of(series, ref_fits)
    ref_rms = float(np.median([ref_fits[T][1] for T in Ts]))
    rows = [dict(method='reference (released deterministic multi-start)', seed='-',
                 median_rms_pct=round(ref_rms, 3),
                 Ea_Rohm=round(ref_ea['Rohm'], 2), Ea_Rsei=round(ref_ea['Rsei'], 2),
                 Ea_Rct=round(ref_ea['Rct'], 2), dEa_Rct_vs_reference=0.0,
                 n_spectra_lower_cost=0, n_spectra_higher_cost=0)]
    print(f"reference (released multi-start): Ea_Rct {ref_ea['Rct']:6.2f}  median rms {ref_rms:.3f} %")

    for log_space, label in [(False, 'global DE, raw bounds'), (True, 'global DE, log-parameterized')]:
        for seed in ([0] if not log_space else SEEDS):
            t0 = time.time()
            fits = {T: global_fit(*series[T], seed, log_space) for T in Ts}
            ea = ea_of(series, fits)
            rms = float(np.median([fits[T][1] for T in Ts]))
            lower = sum(fits[T][1] < ref_fits[T][1] - 1e-6 for T in Ts)
            higher = sum(fits[T][1] > ref_fits[T][1] + 1e-6 for T in Ts)
            rows.append(dict(method=label, seed=seed, median_rms_pct=round(rms, 3),
                             Ea_Rohm=round(ea['Rohm'], 2), Ea_Rsei=round(ea['Rsei'], 2),
                             Ea_Rct=round(ea['Rct'], 2),
                             dEa_Rct_vs_reference=round(abs(ea['Rct'] - ref_ea['Rct']), 2),
                             n_spectra_lower_cost=int(lower), n_spectra_higher_cost=int(higher)))
            print(f"{label:32s} seed {seed}: Ea_Rct {ea['Rct']:7.2f}  median rms {rms:.3f} %"
                  f"  |dEa_Rct| {abs(ea['Rct'] - ref_ea['Rct']):6.2f}  better/worse than reference in "
                  f"{lower}/{higher} of {len(Ts)} spectra  [{time.time() - t0:.0f}s]",
                  flush=True)

    df = pd.DataFrame(rows)
    lg = df[df.method.str.contains('log-parameterized')]
    if len(lg) > 1:
        spread = float(lg.Ea_Rct.max() - lg.Ea_Rct.min())
        print(f"\nseed-to-seed spread of the LOG-parameterized global optimizer: "
              f"{spread:.2f} kJ/mol at fit qualities {lg.median_rms_pct.min():.3f}-"
              f"{lg.median_rms_pct.max():.3f} %.")
    safe_write(df, 'global_optimization_check.csv', True,
               'the parsed reference corpus (script 01)')
