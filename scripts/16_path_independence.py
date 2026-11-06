"""Initialization-path independence of the extraction (manuscript section 2.4.1).

The OCV-matched mid state of the reference corpus (state 5, OCV ~3.67 V; the Table 3
row) is fitted under three visiting orders that differ only in how the optimizer is
seeded from neighbouring solutions:

  * warm -> cold chaining: each temperature starts from the next-warmer solution;
  * cold -> warm chaining: each temperature starts from the next-colder solution;
  * independent fits: no information is passed between temperatures.

Each order is run twice:

  * with the released procedure, eislib.fit_ecm (zero-crossing and high-frequency-
    intercept guesses with seeded narrow restarts, seeded broad restarts, lowest cost
    accepted, tau-canonicalization), and
  * with a deliberately weak procedure for contrast: one local solve from the chained
    solution (or, for the first temperature, from the data-derived guess), no restarts
    and no canonicalization.

The Arrhenius regression always uses the same eleven (1/T, ln 1/R) pairs; ordinary least
squares depends only on that set, not on the order in which the fits were produced, so
any difference between orders comes from the local minima reached by the fits.

Writes out/path_independence.csv. Needs out/unisa_parsed.pkl (script 01) and
out/unisa_arrhenius_input.csv (script 02).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import (fit_ecm, zmodel, BOUNDS9, arrhenius, load_selected_spectra,
                           zero_crossing_real)

MID = 5


def weak_fit(fr, z, p0):
    """Single local solve, no restarts, no canonicalization (the contrast case)."""
    w = 2*np.pi*np.asarray(fr, float)
    lb, ub = list(BOUNDS9[0])+[0.0], list(BOUNDS9[1])+[1e-5]
    if p0 is None:
        span = max(z.real.max()-z.real.min(), 1e-3)
        r0 = zero_crossing_real(fr, z)
        r0 = z.real.min()*0.97 if r0 is None else r0
        p0 = [r0, span*0.25, 20, 0.8, span*0.5, 200, 0.85, span*0.8, 30, 1e-7]
    def resid(p):
        e = (zmodel(p, w, True)-z)/np.abs(z)
        return np.concatenate([e.real, e.imag])
    r = least_squares(resid, np.clip(p0, lb, ub), bounds=(lb, ub), max_nfev=8000)
    return r.x, np.sqrt(r.cost/len(fr))*100


def run_order(series, order, procedure):
    temps = sorted(series)
    if order == 'cold -> warm chaining':
        seq = temps
    else:
        seq = temps[::-1]
    prev, out = None, {}
    for T in seq:
        fr, z = series[T]
        p0 = prev if order != 'independent fits' else None
        if procedure == 'released':
            p, rms = fit_ecm(fr, z, p0=p0, with_L=True)
        else:
            p, rms = weak_fit(fr, z, p0)
        prev = p.copy(); out[T] = (p, rms)
    TK = np.array([T+273.15 for T in temps])
    row = dict(procedure=procedure, order=order)
    for name, i in [('Rohm', 0), ('Rsei', 1), ('Rct', 4)]:
        row['Ea_'+name] = round(arrhenius(TK, [out[T][0][i] for T in temps])[0], 2)
    row['median_rms_pct'] = round(float(np.median([out[T][1] for T in temps])), 3)
    Tc = min(temps)
    row['R_fast_coldest_mOhm'] = round(out[Tc][0][1]*1e3, 2)
    row['R_slow_coldest_mOhm'] = round(out[Tc][0][4]*1e3, 2)
    return row


if __name__ == '__main__':
    need = [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_arrhenius_input.csv')]
    if not all(os.path.exists(f) for f in need):
        print("[abort] run 01_parse_unisa.py and 02_fit_unisa.py first -> "
              "path_independence.csv NOT written."); sys.exit(0)
    series = {s['T']: (s['fr'], s['z']) for s in load_selected_spectra(OUT, ridx=MID)}
    print(f"state {MID}: {len(series)} setpoints {sorted(series)}")
    rows = []
    for procedure in ['released', 'single start, no canonicalization']:
        for order in ['warm -> cold chaining', 'cold -> warm chaining', 'independent fits']:
            r = run_order(series, order, procedure); rows.append(r)
            print(f"  {procedure:34s} {order:22s} Ea(Rohm/Rsei/Rct) = "
                  f"{r['Ea_Rohm']:.2f} / {r['Ea_Rsei']:.2f} / {r['Ea_Rct']:.2f}   "
                  f"median rms {r['median_rms_pct']:.3f} %   coldest R_fast/R_slow "
                  f"{r['R_fast_coldest_mOhm']:.2f}/{r['R_slow_coldest_mOhm']:.2f} mOhm",
                  flush=True)
    df = pd.DataFrame(rows)
    safe_write(df, 'path_independence.csv', len(df) == 6,
               "out/unisa_parsed.pkl and out/unisa_arrhenius_input.csv")
    rel = df[df.procedure == 'released']
    print(f"\nreleased procedure: spread across orders  Rohm {np.ptp(rel.Ea_Rohm):.2f}  "
          f"Rsei {np.ptp(rel.Ea_Rsei):.2f}  Rct {np.ptp(rel.Ea_Rct):.2f} kJ/mol")
