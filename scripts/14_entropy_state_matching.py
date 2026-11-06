"""State matching across temperature: entropy coefficient and matching rule (section 2.8, 3.3).

Why this exists
---------------
States are matched across temperature by MEASURED open-circuit voltage against the ladder
observed at the reference temperature (25 C, script 02). Two things can make that matching
ambiguous, and this script quantifies both from the same fits (the ECM fits do not depend
on the state label: script 02 fits every triplicate-median spectrum and only afterwards
assigns it to a state, so the assignment can be recomputed without refitting):

1. Entropy coefficient. OCV at fixed state of charge is temperature-dependent through
   dU/dT = (dU_ocv/dT)|_SoC, so the ladder at temperature T is displaced by
   (T - T_ref) dU/dT. For each assumed coefficient every measured OCV is mapped back to
   the reference temperature, U_ref = OCV - (T - T_ref) dU/dT (eq. 20 of the manuscript),
   re-assigned to the nearest reference rung, and the regression is repeated. The sweep
   spans -0.5 ... +0.5 mV/K, which brackets reported full-cell coefficients of
   graphite-based cells (a few tenths of a mV/K, sign-changing with graphite staging).

2. Matching rule. The corpus also labels every spectrum with a nominal state of charge
   (100 ... 0 % in 10 % steps), set by charge counting relative to the capacity measured
   at each temperature. Matching by that label instead of by OCV is the obvious alternative
   rule; the two rules agree except at the low-SoC end, where the cold ladders sit up to
   ~70 mV (10 % label) above the 25 C ladder.

In every variant, duplicates (two spectra of one temperature assigned to the same state)
are resolved exactly as in script 02: the spectrum whose (corrected) OCV is closest to the
reference rung is kept, ties broken by the lower fit residual, then by acquisition order.

Writes
  out/entropy_state_matching.csv    one row per dU/dT: fits reassigned, max |dE_a| per channel
  out/state_matching_by_state.csv   per state: E_a under OCV and label matching, the label-rule
                                    shift and the largest entropy-sweep shift (used by script 08)
"""
import os, sys
import numpy as np, pandas as pd
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import arrhenius

T_REF = 25.0
DUDT_MV_PER_K = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
PROCS = ['Rohm', 'Rsei', 'Rct', 'Rsum']   # Rsum = R_SEI + R_ct, the swap-invariant sum


def select_one_per_state(d, key, ref):
    """Script-02 duplicate rule on a frame with columns ridx_new, T, rms, order and `key`."""
    d = d.copy()
    d['dist'] = np.abs(d[key].values - np.asarray(ref)[d['ridx_new'].values])
    d = d.sort_values(['ridx_new', 'T', 'dist', 'rms', 'order'])
    return d.drop_duplicates(['ridx_new', 'T'], keep='first')


def regress_states(dm):
    rows = []
    for r, gg in dm.groupby('ridx_new'):
        gg = gg.sort_values('T')
        if len(gg) < 8:
            continue
        TK = gg['T'].values + 273.15
        row = {'ridx': int(r), 'nT': len(gg)}
        for proc in PROCS:
            row['Ea_' + proc] = arrhenius(TK, gg[proc].values)[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index('ridx')


def ocv_matching(df, ref, dudt_V_per_K):
    """Re-assign states using entropy-corrected OCV, then regress. Returns (table, n_moved)."""
    u_ref = df.ocv.values - (df['T'].values - T_REF) * dudt_V_per_K
    ridx = np.array([int(np.argmin(np.abs(u - np.asarray(ref)))) for u in u_ref])
    moved = int((ridx != df.ridx.values).sum())
    d = df.copy(); d['ridx_new'] = ridx; d['u_ref'] = u_ref
    return regress_states(select_one_per_state(d, 'u_ref', ref)), moved


def label_matching(df, ref_soc):
    """Match by the corpus's nominal SoC label: state = the 25 C rung with the same label."""
    lab2r = {round(s, 6): i for i, s in enumerate(ref_soc)}
    d = df[df.soc.notna()].copy()
    d['ridx_new'] = [lab2r.get(round(s, 6), -1) for s in d.soc]
    d = d[d.ridx_new >= 0]
    d['order'] = d['order']
    return regress_states(d.sort_values(['ridx_new', 'T', 'order']).drop_duplicates(['ridx_new', 'T']))


if __name__ == '__main__':
    fits = os.path.join(OUT, 'unisa_fits.csv')
    if not os.path.exists(fits):
        print("[abort] out/unisa_fits.csv not found -> run 02_fit_unisa.py first.")
        print("        entropy_state_matching.csv NOT written."); sys.exit(0)
    df = pd.read_csv(fits).reset_index().rename(columns={'index': 'order'})
    df['Rsum'] = df['Rsei'] + df['Rct']

    # reference ladder used by script 02: the OCVs (and labels) at T_REF
    r25 = df[df['T'] == T_REF].sort_values('ridx')
    ref = r25.groupby('ridx').ocv.median().sort_index().values
    span = df['T'].max() - df['T'].min()
    print(f"reference ladder at {T_REF:.0f} C: {len(ref)} states, {ref.min():.3f}-{ref.max():.3f} V, "
          f"median spacing {np.median(np.abs(np.diff(np.sort(ref))))*1e3:.0f} mV; span {span:.0f} K\n")

    base, _ = ocv_matching(df, ref, 0.0)
    rows, per_state = [], {r: [] for r in base.index}
    for mv in DUDT_MV_PER_K:
        tab, moved = ocv_matching(df, ref, mv*1e-3)
        common = base.index.intersection(tab.index)
        d = {p: float(np.max(np.abs(tab.loc[common, 'Ea_'+p] - base.loc[common, 'Ea_'+p]))) for p in PROCS}
        for r in common:
            per_state[r].append(abs(tab.loc[r, 'Ea_Rct'] - base.loc[r, 'Ea_Rct']))
        rows.append(dict(dUdT_mV_per_K=mv, ocv_shift_over_span_mV=round(abs(mv)*span, 1),
                         states_reassigned=moved, n_states=len(tab),
                         max_dEa_Rohm=round(d['Rohm'], 2), max_dEa_Rsei=round(d['Rsei'], 2),
                         max_dEa_Rct=round(d['Rct'], 2), max_dEa_Rsum=round(d['Rsum'], 2)))
        print(f"  dU/dT = {mv:+.1f} mV/K  shift over span {abs(mv)*span:5.1f} mV  {moved:3d} fits "
              f"reassigned  max |dEa| (Rohm/Rsei/Rct) = {d['Rohm']:.2f} / {d['Rsei']:.2f} / {d['Rct']:.2f}")
    out = pd.DataFrame(rows)
    safe_write(out, 'entropy_state_matching.csv', True, "out/unisa_fits.csv from script 02")

    # alternative matching rule: nominal SoC label
    by_state = []
    if 'soc' in df.columns and df.soc.notna().all():
        ref_soc = r25.groupby('ridx').soc.median().sort_index().values
        lab = label_matching(df, ref_soc)
        n_disagree = int(sum(1 for _, x in df.iterrows()
                             if abs(ref_soc[int(x.ridx)] - x.soc) > 1e-6))
        print(f"\nnominal-SoC-label matching: {n_disagree} of {len(df)} spectra are assigned to a "
              f"different state than by OCV")
        for r in sorted(set(base.index) | set(lab.index)):
            row = dict(ridx=int(r), ocv_ref=round(float(ref[r]), 3), soc_label_pct=float(ref_soc[r]))
            for name, t in [('ocv', base), ('label', lab)]:
                row[f'nT_{name}'] = int(t.loc[r, 'nT']) if r in t.index else 0
                for p in PROCS:
                    row[f'Ea_{p}_{name}'] = round(float(t.loc[r, 'Ea_'+p]), 2) if r in t.index else np.nan
            for p in ['Rct', 'Rsum']:
                row[f'd_rule_{p}'] = (round(abs(row[f'Ea_{p}_label'] - row[f'Ea_{p}_ocv']), 2)
                                      if (r in base.index and r in lab.index) else np.nan)
            row['max_d_entropy_Rct'] = round(max(per_state.get(r, [0.0])), 2) if r in base.index else np.nan
            by_state.append(row)
        bs = pd.DataFrame(by_state)
        bs.to_csv(os.path.join(OUT, 'state_matching_by_state.csv'), index=False)
        print(bs.to_string(index=False))
    else:
        print("\n[note] no SoC labels in out/unisa_fits.csv (re-run 01 and 02 with the corpus "
              "workbooks); label-rule comparison skipped")
    print(f"\nworst-case |dEa(Rct)| over the swept dU/dT envelope: {out.max_dEa_Rct.max():.2f} kJ/mol")
