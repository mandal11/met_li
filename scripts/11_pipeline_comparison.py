"""Reference-vs-alternative pipeline comparison (Table 2) and per-temperature fit quality.

Reference pipeline: eislib.fit_ecm chained warm->cold (deterministic, released search).
Alternative pipeline of equal effort: per-spectrum best-of-12 random restarts under the
same bounds, no chaining, no canonicalization at fit time. Because the alternative's raw
branch assignment depends on the restart seed, results are computed for three seeds
(0, 7, 13) and the released row is the seed-median; per-seed values are retained in the
CSV so the spread is inspectable. Canonicalization is applied post hoc to split the
disagreement into a label-convention term and a near-degeneracy term.
"""
import numpy as np, pandas as pd, os, sys, time
from scipy.optimize import least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, zmodel, BOUNDS9, canonicalize, branch_tau, arrhenius

SHEETS = ['50', '40', '25', '15', '0', '-10']
SEEDS = [0, 7, 13]

def load(f, sheet, scale=1.0, flip=False):
    d = pd.read_excel(f, sheet_name=sheet).dropna()
    fr = d.iloc[:, 0].astype(float).values
    z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
    i = np.argsort(fr); fr, z = fr[i], z[i]*scale
    if flip: z = z.real - 1j*np.abs(z.imag)
    return fr, z

def fit_alt_raw(fr, z, rng, n=12, max_nfev=4000):
    """Best-of-n random restarts; returns the RAW (uncanonicalized) solution and rms."""
    w = 2*np.pi*np.asarray(fr, float)
    lb, ub = list(BOUNDS9[0]), list(BOUNDS9[1])
    span = max(z.real.max()-z.real.min(), 1e-3)
    def resid(p):
        e = (zmodel(p, w)-z)/np.abs(z)
        return np.concatenate([e.real, e.imag])
    best = None
    for _ in range(n):
        p0 = [z.real.min()*rng.uniform(.9, 1.), span*rng.uniform(.05, .6), 10**rng.uniform(-1, 3),
              rng.uniform(.6, 1), span*rng.uniform(.05, .6), 10**rng.uniform(-1, 3),
              rng.uniform(.6, 1), span*rng.uniform(.3, 1.5), 10**rng.uniform(0, 2.5)]
        try:
            r = least_squares(resid, np.clip(p0, lb, ub), bounds=(lb, ub), max_nfev=max_nfev)
            if best is None or r.cost < best.cost: best = r
        except Exception:
            pass
    return best.x, np.sqrt(best.cost/len(fr))*100

def ea(TK, R): return arrhenius(TK, R)[0]

if __name__ == '__main__':
    datasets = [('file A', SIXT_FILE_A, 1e-3, True),
                ('file B', SIXT_FILE_B, 1e-3, True),
                ('in-house HE2', INHOUSE_HE2, 1.0, False),
                ('in-house VTC5A', INHOUSE_VTC5A, 1.0, False)]
    cache_dir = os.path.join(OUT, '.cache_pipeline'); os.makedirs(cache_dir, exist_ok=True)
    # the cache is keyed to the exact eislib source, so fits made by an earlier version of the
    # fitting procedure are never reused after eislib changes
    import hashlib
    _lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eislib', 'eislib.py')
    _tag = hashlib.sha1(open(_lib, 'rb').read() + open(__file__, 'rb').read()).hexdigest()[:10]
    rows, fq = [], []
    for name, f, sc, fl in datasets:
        if f is None:
            print(f"[skip] {name}: data not present"); continue
        t0 = time.time()
        cache = os.path.join(cache_dir, name.replace(' ', '_')+f'_{_tag}.pkl')
        if os.path.exists(cache):
            import pickle as _pk
            ref, per_seed, TK = _pk.load(open(cache, 'rb'))
            print(f"  {name}: loaded from cache", flush=True)
        else:
            spectra = {t: load(f, t, sc, fl) for t in SHEETS}
            # reference pipeline
            prev = None; ref = []
            for t in SHEETS:
                fr, z = spectra[t]
                p, rms = fit_ecm(fr, z, p0=prev); prev = p.copy()
                t1, t2 = branch_tau(p)
                ref.append((float(t), p, rms, t2/t1))
            TK = [t+273.15 for t, _, _, _ in ref]
            # alternative pipeline, three seeds
            per_seed = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                raws, rmss = [], []
                for t in SHEETS:
                    fr, z = spectra[t]
                    x, rms = fit_alt_raw(fr, z, rng)
                    raws.append(x); rmss.append(rms)
                swaps = sum(1 for x in raws if branch_tau(x)[0] > branch_tau(x)[1])
                e_raw = dict(Rohm=ea(TK, [x[0] for x in raws]), Rsei=ea(TK, [x[1] for x in raws]),
                             Rct=ea(TK, [x[4] for x in raws]))
                canon = [canonicalize(x) for x in raws]
                e_can = dict(Rsei=ea(TK, [x[1] for x in canon]), Rct=ea(TK, [x[4] for x in canon]))
                per_seed.append(dict(seed=seed, swaps=swaps, rmss=rmss, e_raw=e_raw, e_can=e_can))
            import pickle as _pk
            _pk.dump((ref, per_seed, TK), open(cache, 'wb'))
        ref_ea = dict(Rohm=ea(TK, [p[0] for _, p, _, _ in ref]),
                      Rsei=ea(TK, [p[1] for _, p, _, _ in ref]),
                      Rct=ea(TK, [p[4] for _, p, _, _ in ref]))
        med = lambda key, sub: float(np.median([s_[key][sub] for s_ in per_seed]))
        lt = [abs(s_['e_raw']['Rct']-s_['e_can']['Rct']) for s_ in per_seed]
        rt = [abs(s_['e_can']['Rct']-ref_ea['Rct']) for s_ in per_seed]
        rows.append(dict(
            source=name,
            swaps='/'.join(str(s_['swaps']) for s_ in per_seed)+f" of 6 (seeds {SEEDS})",
            Ea_Rct_reference=round(ref_ea['Rct'], 2),
            Ea_Rct_naive_raw=round(med('e_raw', 'Rct'), 2),
            Ea_Rct_naive_canon=round(med('e_can', 'Rct'), 2),
            label_term=round(float(np.median(lt)), 2),
            label_term_min=round(min(lt), 2), label_term_max=round(max(lt), 2),
            residual_term=round(float(np.median(rt)), 2),
            residual_term_min=round(min(rt), 2), residual_term_max=round(max(rt), 2),
            Ea_Rsei_reference=round(ref_ea['Rsei'], 2),
            Ea_Rsei_naive_raw=round(med('e_raw', 'Rsei'), 2),
            Ea_Rsei_naive_canon=round(med('e_can', 'Rsei'), 2),
            Ea_Rohm_reference=round(ref_ea['Rohm'], 2),
            Ea_Rohm_naive_raw=round(med('e_raw', 'Rohm'), 2)))
        for (t, p, rms, sep), s7 in zip(ref, per_seed[1]['rmss']):
            fq.append(dict(source=name, T=int(t), rms_ref=round(rms, 3),
                           rms_alt=round(float(np.median([ps['rmss'][SHEETS.index(str(int(t)))] for ps in per_seed])), 3),
                           sep=round(sep, 1)))
        print(f"  {name}: done in {time.time()-t0:.0f}s  label_term(med) {rows[-1]['label_term']}"
              f"  residual_term(med) {rows[-1]['residual_term']}", flush=True)
    df = pd.DataFrame(rows)
    complete = (len(rows) == len(datasets))
    needs = ("all four six-setpoint datasets: data/inhouse/{Samsung,Panasonic,LGHE2,"
             "all_temp_sony}.xlsx (S1/S2, DOI [A])")
    if safe_write(df, 'pipeline_comparison.csv', complete, needs) is None:
        sys.exit(0)
    safe_write(pd.DataFrame(fq), 'fit_quality_separation.csv', complete, needs)
    cols = ['source', 'swaps', 'Ea_Rct_reference', 'Ea_Rct_naive_raw',
            'Ea_Rct_naive_canon', 'label_term', 'label_term_min', 'label_term_max',
            'residual_term', 'Ea_Rohm_reference', 'Ea_Rohm_naive_raw']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
