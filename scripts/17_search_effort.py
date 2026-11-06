"""Search-effort sensitivity of the state-resolved activation energies (sections 2.4, 3.3).

The two-branch objective of the reference corpus has several basins of nearly equal cost.
Earlier releases of eislib.fit_ecm used a narrow deterministic multi-start: the chained
neighbouring solution, one data-derived guess with R_ohm = 0.97 x min Re(Z), and four
perturbations of its branch resistances and CPE coefficients. This script refits every
corpus spectrum with exactly that legacy procedure (same visiting order and chaining as
script 02) and compares it with the released broad search (out/unisa_fits.csv).

Writes
  out/search_effort_fits.csv       legacy fit of every spectrum next to the released fit
  out/search_effort_by_state.csv   per state: E_a with the legacy and the released search,
                                   and the shift |dE_a| (used by script 08)
  out/search_effort_inhouse.csv    six-setpoint datasets: E_a(R_ct) of the reference pipeline of
                                   script 11 with the legacy search, and the resulting residual term
  out/search_effort_inhouse_fits.csv  the same, per setpoint, next to the released fit quality
  out/search_convergence.csv       convergence of the released search: every corpus spectrum
                                   refitted with 240 instead of 160 broad starts, and with
                                   another seed (section 2.4, Appendix B)
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import zmodel, BOUNDS9, canonicalize, arrhenius, unisa_groups


def fit_ecm_legacy(fr, z, p0=None, max_nfev=8000, n_restart=4, seed=0, with_L=True):
    """eislib.fit_ecm as released before October 2026, verbatim."""
    w = 2*np.pi*np.asarray(fr, float); z = np.asarray(z, complex)
    lb, ub = list(BOUNDS9[0]), list(BOUNDS9[1])
    if with_L: lb, ub = lb+[0.0], ub+[1e-5]
    span = max(z.real.max()-z.real.min(), 1e-3)
    base = [z.real.min()*0.97, span*0.25, 20, 0.8, span*0.5, 200, 0.85, span*0.8, 30]
    if with_L: base = base+[1e-7]
    inits = ([list(p0)] if p0 is not None else []) + [base]
    rng = np.random.default_rng(seed)
    for _ in range(n_restart):
        q = np.array(base, float)
        q[1] *= 10**rng.uniform(-0.7, 0.7); q[4] *= 10**rng.uniform(-0.7, 0.7)
        q[2] *= 10**rng.uniform(-1.5, 1.5); q[5] *= 10**rng.uniform(-1.5, 1.5)
        inits.append(list(q))
    def resid(p):
        e = (zmodel(p, w, with_L)-z)/np.abs(z)
        return np.concatenate([e.real, e.imag])
    best = None
    for pp in inits:
        try:
            r = least_squares(resid, np.clip(pp, lb, ub), bounds=(lb, ub), max_nfev=max_nfev)
            if best is None or r.cost < best.cost: best = r
        except Exception:
            pass
    return canonicalize(best.x), np.sqrt(best.cost/len(fr))*100


def _fit_T(args):
    """Worker of the convergence check: all groups of one temperature, chained as in script 02."""
    from eislib.eislib import fit_ecm
    T, groups, n_broad, seed = args
    prev, out = None, []
    for gi, (fr, z) in enumerate(groups):
        p, rms = fit_ecm(fr, z, p0=prev, with_L=True, n_broad=n_broad, seed=seed); prev = p.copy()
        out.append(dict(T=T, gi=gi, Rsei=p[1], Rct=p[4], rms=rms))
    return n_broad, seed, out


if __name__ == '__main__':
    need = [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_fits.csv'),
            os.path.join(OUT, 'unisa_arrhenius_input.csv')]
    if not all(os.path.exists(f) for f in need):
        print("[abort] run 01 and 02 first -> search_effort_*.csv NOT written."); sys.exit(0)
    alldata = pickle.load(open(need[0], 'rb'))
    ref = pd.read_csv(need[1])
    sel = pd.read_csv(need[2])[['ridx', 'T_C', 'gi']].rename(columns={'T_C': 'T'})
    rows = []
    for T in sorted(alldata, reverse=True):              # visiting order of script 02
        prev = None
        for gi, g in enumerate(unisa_groups(alldata[T])):
            fr = g[0][1]; z = np.median(np.array([s[2] for s in g]), axis=0)
            p, rms = fit_ecm_legacy(fr, z, p0=prev); prev = p.copy()
            r = ref[(ref['T'] == T) & (ref['gi'] == gi)].iloc[0]
            rows.append(dict(T=T, gi=gi, ridx=int(r.ridx), Rohm_legacy=p[0], Rsei_legacy=p[1],
                             Rct_legacy=p[4], rms_legacy=rms, Rohm=r.Rohm, Rsei=r.Rsei, Rct=r.Rct,
                             rms=r.rms))
        print(f"  T = {T:>3} C done", flush=True)
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(OUT, 'search_effort_fits.csv'), index=False)
    lower = int((f.rms < f.rms_legacy - 1e-6).sum()); higher = int((f.rms > f.rms_legacy + 1e-6).sum())
    big = int((np.abs(f.Rct - f.Rct_legacy)/f.Rct_legacy > 0.10).sum())
    print(f"\nreleased search reaches a lower cost than the legacy search in {lower} of {len(f)} "
          f"spectra (higher in {higher}); R_ct moves by more than 10 % in {big}")
    print(f"median rms: legacy {f.rms_legacy.median():.4f} %, released {f.rms.median():.4f} %")

    # per-state regression on the spectra retained by script 02, both searches
    m = sel.merge(f, on=['ridx', 'T', 'gi'])
    out = []
    for r, gg in m.groupby('ridx'):
        gg = gg.sort_values('T')
        if len(gg) < 8:
            continue
        TK = gg['T'].values + 273.15
        row = dict(ridx=int(r), nT=len(gg))
        gg = gg.assign(Rsum=gg.Rsei + gg.Rct, Rsum_legacy=gg.Rsei_legacy + gg.Rct_legacy)
        for p in ['Rohm', 'Rsei', 'Rct', 'Rsum']:
            a = arrhenius(TK, gg[p+'_legacy'].values)[0]; b = arrhenius(TK, gg[p].values)[0]
            row[f'Ea_{p}_legacy'] = round(a, 2); row[f'Ea_{p}'] = round(b, 2)
            row[f'd_{p}'] = round(abs(b-a), 2)
        out.append(row)
    out = pd.DataFrame(out)
    safe_write(out, 'search_effort_by_state.csv', True, "out/unisa_fits.csv")
    print(out.to_string(index=False))

    # six-setpoint datasets: the reference pipeline of script 11 with the legacy search, compared
    # with the released reference and with the canonicalized alternative pipeline (seed median)
    pc = os.path.join(OUT, 'pipeline_comparison.csv')
    if os.path.exists(pc):
        pc = pd.read_csv(pc).set_index('source')
        SHEETS = ['50', '40', '25', '15', '0', '-10']
        def load(f, sheet, scale=1.0, flip=False):
            d = pd.read_excel(f, sheet_name=sheet).dropna()
            fr = d.iloc[:, 0].astype(float).values
            z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
            i = np.argsort(fr); fr, z = fr[i], z[i]*scale
            if flip: z = z.real - 1j*np.abs(z.imag)
            return fr, z
        rows, per_T = [], []
        for name, f, sc, fl in [('file A', SIXT_FILE_A, 1e-3, True), ('file B', SIXT_FILE_B, 1e-3, True),
                                ('in-house HE2', INHOUSE_HE2, 1.0, False),
                                ('in-house VTC5A', INHOUSE_VTC5A, 1.0, False)]:
            if f is None or name not in pc.index:
                continue
            prev = None; rec = []
            for t in SHEETS:                                   # warm -> cold, as script 11
                fr, z = load(f, t, sc, fl)
                p, rms = fit_ecm_legacy(fr, z, p0=prev, with_L=False); prev = p.copy()
                rec.append((float(t)+273.15, p[4], rms))
                per_T.append(dict(source=name, T=int(t), rms_legacy=round(rms, 4),
                                  Rsei_legacy_mOhm=round(p[1]*1e3, 4), Rct_legacy_mOhm=round(p[4]*1e3, 4)))
            ea = arrhenius([r[0] for r in rec], [r[1] for r in rec])[0]
            rows.append(dict(source=name, Ea_Rct_reference_legacy_search=round(ea, 2),
                             Ea_Rct_reference_released=pc.loc[name, 'Ea_Rct_reference'],
                             Ea_Rct_alternative_canonical=pc.loc[name, 'Ea_Rct_naive_canon'],
                             residual_term_with_legacy_reference=round(abs(ea-pc.loc[name, 'Ea_Rct_naive_canon']), 2),
                             residual_term_with_released_reference=pc.loc[name, 'residual_term']))
        ih = pd.DataFrame(rows)
        safe_write(ih, 'search_effort_inhouse.csv', len(rows) == 4,
                   "data/inhouse workbooks and out/pipeline_comparison.csv (script 11)")
        print(ih.to_string(index=False))
        # per setpoint: legacy fit next to the released reference fit of script 11
        pt = pd.DataFrame(per_T)
        fq = os.path.join(OUT, 'fit_quality_separation.csv')
        if os.path.exists(fq) and len(pt):
            fq = pd.read_csv(fq)[['source', 'T', 'rms_ref']].rename(columns={'rms_ref': 'rms_released'})
            pt = pt.merge(fq, on=['source', 'T'], how='left')
            pt['released_lower_cost'] = pt.rms_released < pt.rms_legacy - 1e-3   # rms_ref has 3 decimals
        safe_write(pt, 'search_effort_inhouse_fits.csv', len(rows) == 4,
                   "data/inhouse workbooks and out/fit_quality_separation.csv (script 11)")
        print(pt.to_string(index=False))

    # convergence of the released search (160 broad starts, seed 0): refit every corpus spectrum
    # with 240 broad starts and with another seed, same visiting order and chaining as script 02
    tasks = []
    for nb, sd in [(240, 0), (160, 1)]:
        for T in sorted(alldata, reverse=True):
            groups = [(g[0][1], np.median(np.array([x[2] for x in g]), axis=0))
                      for g in unisa_groups(alldata[T])]
            tasks.append((T, groups, nb, sd))
    n = int(os.environ.get('EIS_WORKERS', '0')) or os.cpu_count() or 1
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(min(n, len(tasks))) as pool:
        res = pool.map(_fit_T, tasks, chunksize=1)
    conv = ref[['T', 'gi', 'ridx', 'Rct', 'rms']].copy()
    for nb, sd, out in res:
        tag = 'nb240' if nb == 240 else 'seed1'
        d = pd.DataFrame(out).rename(columns={'Rct': 'Rct_' + tag, 'rms': 'rms_' + tag, 'Rsei': 'Rsei_' + tag})
        if 'Rct_' + tag in conv.columns:
            conv = conv.set_index(['T', 'gi']); conv.update(d.set_index(['T', 'gi'])); conv = conv.reset_index()
        else:
            conv = conv.merge(d, on=['T', 'gi'], how='left')
    arr = pd.read_csv(os.path.join(OUT, 'unisa_arrhenius.csv'))
    t3 = set(arr.loc[arr.nT >= 8, 'ridx'])
    keep = set(zip(sel['T'], sel['gi'], sel['ridx']))
    conv['in_table3'] = [(r.T, r.gi, r.ridx) in keep and r.ridx in t3 for r in conv.itertuples()]
    for tag in ['nb240', 'seed1']:
        conv['dRct_' + tag + '_pct'] = (100*np.abs(conv['Rct_' + tag] - conv.Rct)/conv.Rct).round(4)
        conv['released_lower_cost_' + tag] = conv['rms_' + tag] > conv.rms + 1e-6
    safe_write(conv, 'search_convergence.csv', conv[['Rct_nb240', 'Rct_seed1']].notna().all().all(),
               "out/unisa_fits.csv (script 02)")
    for tag, lab in [('nb240', '240 broad starts'), ('seed1', 'seed 1')]:
        d = conv['dRct_' + tag + '_pct']
        print(f"{lab}: max |dR_ct| {d.max():.3f} % over all {len(conv)} spectra, "
              f"{d[conv.in_table3].max():.3f} % over the {int(conv.in_table3.sum())} Table-3 spectra; "
              f"lower cost than the released search in "
              f"{int((conv['rms_' + tag] < conv.rms - 1e-6).sum())} spectra")
