"""Exact enumeration of the branch-label degeneracy (Table 2, label term).

Rationale
---------
An earlier version measured the label term by running an uncanonicalized pipeline with
random restarts and recording how far its Arrhenius slope fell from the canonicalized one.
That estimator is a SAMPLE of a discrete quantity - which subset of setpoints happens to
come out branch-swapped - and the subset depends on which of several near-equal-cost minima
the optimizer reaches. It therefore differs between machines: an independent run of
byte-identical code on a different BLAS moved one dataset's raw estimate by 30.7 kJ/mol.

The degeneracy itself, however, is exactly enumerable and needs no optimizer. Take the
canonicalized fit at each of the k setpoints (reproducible to <= 0.06 kJ/mol across
platforms). The branch labels are exchangeable, so any of the 2^k label assignments is an
admissible output of a pipeline that does not canonicalize. Evaluating the Arrhenius
regression on all 2^k assignments gives:

    label_term_max   worst-case |dE_a| relative to the canonical assignment
    label_term_span  full range of E_a over all assignments
    n_assignments    2^k

These are deterministic, seed-free and platform-independent to the precision of the
underlying fits, and they BOUND rather than sample the effect. The complementary
near-degeneracy term (distinct parameter sets of equal fit quality) is still measured
empirically by script 11.
"""
import numpy as np, pandas as pd, os, sys, itertools
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, arrhenius

SHEETS = ['50', '40', '25', '15', '0', '-10']

def load(f, sheet, scale=1.0, flip=False):
    d = pd.read_excel(f, sheet_name=sheet).dropna()
    fr = d.iloc[:, 0].astype(float).values
    z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
    i = np.argsort(fr); fr, z = fr[i], z[i]*scale
    if flip: z = z.real - 1j*np.abs(z.imag)
    return fr, z

def enumerate_labels(TK, R_fast, R_slow):
    """E_a over all 2^k label assignments; returns (canonical, min, max, worst |delta|)."""
    R_fast, R_slow = np.asarray(R_fast, float), np.asarray(R_slow, float)
    k = len(TK)
    canon = arrhenius(TK, R_slow)[0]                      # canonical: slow branch = R_ct
    vals = []
    for mask in itertools.product([0, 1], repeat=k):
        pick = np.where(np.array(mask) == 1, R_fast, R_slow)
        vals.append(arrhenius(TK, pick)[0])
    vals = np.array(vals)
    return canon, vals.min(), vals.max(), float(np.max(np.abs(vals - canon)))

if __name__ == '__main__':
    datasets = [('file A', SIXT_FILE_A, 1e-3, True),
                ('file B', SIXT_FILE_B, 1e-3, True),
                ('in-house HE2', INHOUSE_HE2, 1.0, False),
                ('in-house VTC5A', INHOUSE_VTC5A, 1.0, False)]
    rows = []
    for name, f, sc, fl in datasets:
        if f is None:
            print(f"[skip] {name}: data not present"); continue
        prev = None; rec = []
        for t in SHEETS:                                   # warm -> cold, as in script 11
            fr, z = load(f, t, sc, fl)
            p, _ = fit_ecm(fr, z, p0=prev); prev = p.copy()
            rec.append((float(t), p[1], p[4]))             # canonicalized: p1 = fast, p4 = slow
        TK = [t+273.15 for t, _, _ in rec]
        canon, lo, hi, worst = enumerate_labels(TK, [a for _, a, _ in rec], [b for _, _, b in rec])
        rows.append(dict(source=name, n_assignments=2**len(TK),
                         Ea_Rct_canonical=round(canon, 2),
                         Ea_min_over_labels=round(lo, 2), Ea_max_over_labels=round(hi, 2),
                         label_term_worst_case=round(worst, 2),
                         label_term_span=round(hi-lo, 2)))
        print(f"  {name:16s} canonical {canon:6.2f} | over {2**len(TK)} assignments "
              f"E_a in [{lo:.2f}, {hi:.2f}] | worst |delta| {worst:.2f} | span {hi-lo:.2f}", flush=True)
    df = pd.DataFrame(rows)
    complete = (len(rows) == len(datasets))
    safe_write(df, 'label_enumeration.csv', complete,
               "all four six-setpoint datasets (S1/S2, DOI [A])")
