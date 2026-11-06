"""Monte-Carlo propagation of the methodological sensitivity inventory (Table 5 -> Table 6).

This is a SCENARIO analysis, not a GUM uncertainty budget: the term magnitudes are the
measured shifts from the rest of the study, but their probability distributions and mutual
independence are assumed, not calibrated. The variance shares are therefore conditional on
these assumptions and must not be read as a formal uncertainty decomposition. The 5th-95th
percentile interval is reported as an overlap with, not a reproduction of, the literature range.

Every magnitude is read from the result tables written by the earlier scripts (nothing is
typed in by hand); the values actually used are written to out/budget_inputs.csv:

  term                      source table (script)                      model
  measurement (amplitude)   bv_amplitude_bias.csv (05)                 one-signed: -U(0,b) - 50 % Exp(b)
  label convention          label_enumeration.csv (13)                 +-U(0, worst case over datasets)
  near-degeneracy           pipeline_comparison.csv (11)               incidence x +-U(0, max residual term)
  search effort             search_effort_by_state.csv (17)            per state: +-U(0, |dEa| of that state)
  weighting                 model_weighting_sensitivity.csv (12)       +-U(0, |dEa|)
  diffusion element         model_weighting_sensitivity.csv (12)       +-U(0, |dEa|)
  grid + window             unisa_subsampling.csv (03), k = 6          N(0, CI_k6/2) + U(min-full, max-full)
  state                     unisa_arrhenius.csv (02)                   U(min-E_true, max-E_true)
  state matching            state_matching_by_state.csv (14)           per state: +-U(0, max(rule, dU/dT))

"Per state" terms draw one of the resolved states at random for every synthetic campaign
and apply the shift measured for that state, so states that are insensitive contribute
nothing. A second run applies the largest per-state shifts unconditionally (worst case).
"""
import numpy as np, pandas as pd, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

K_DESIGN = 6          # six-setpoint protocol assumed for the grid term
MID = 5               # reference (mid) state


def _load_budget_inputs():
    """Load sensitivity bounds from the result tables produced by earlier scripts."""
    def read(name):
        path = os.path.join(OUT, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found; run the producing script first')
        return pd.read_csv(path)

    arr = read('unisa_arrhenius.csv')
    mid = arr.loc[arr['ridx'] == MID, 'Ea_Rct']
    if mid.empty:
        raise ValueError(f'unisa_arrhenius.csv has no ridx={MID} reference state')
    bv = read('bv_amplitude_bias.csv')
    amp = bv[(bv.Ea_true_kJmol == 50) & (bv.I_ac_A == 0.5)]['Ea_bias_kJmol'].iloc[0]
    weighting = read('model_weighting_sensitivity.csv')
    reference = weighting.iloc[0]['Ea_Rct']
    unit = weighting.loc[weighting['variant'] == 'unit weighting', 'Ea_Rct'].iloc[0]
    semi = weighting.loc[weighting['variant'] == 'semi-infinite Warburg', 'Ea_Rct'].iloc[0]
    labels = read('label_enumeration.csv')
    pipeline = read('pipeline_comparison.csv')
    sub = read('unisa_subsampling.csv')
    k6 = sub[sub.k == K_DESIGN]; full = float(sub[sub.k == sub.k.max()].Ea.iloc[0])
    sm = read('state_matching_by_state.csv')
    sm = sm[sm.nT_ocv >= 8]
    sm_shift = np.maximum(sm.d_rule_Rct.fillna(0).values, sm.max_d_entropy_Rct.fillna(0).values)
    se = read('search_effort_by_state.csv')
    return dict(
        E_true=float(mid.iloc[0]),
        amplitude_bias=abs(float(amp)),
        state_min=float(arr['Ea_Rct'].min()),
        state_max=float(arr['Ea_Rct'].max()),
        label_max=float(labels['label_term_worst_case'].max()),
        near_max=float(pipeline['residual_term_max'].max()),
        near_incidence=float((pipeline['residual_term_max'] >= 1.0).mean()),
        weighting_shift=abs(float(reference) - float(unit)),
        diffusion_shift=abs(float(reference) - float(semi)),
        grid_ci_k6=float(k6.CI.median()),
        grid_win_lo=float(k6.Ea.min()) - full,
        grid_win_hi=float(k6.Ea.max()) - full,
        state_match_shifts=[float(x) for x in sm_shift],
        search_shifts=[float(x) for x in se['d_Rct'].values],
    )


def run(seed=11, N=20000, E_true=None, return_E=False, worst_case=False, drop=()):
    inputs = _load_budget_inputs()
    if E_true is None:
        E_true = inputs['E_true']
    rng = np.random.default_rng(seed)
    terms = {}
    b = inputs['amplitude_bias']
    terms['measurement (amplitude)'] = (-np.abs(rng.uniform(0, b, N))
                                        - np.where(rng.random(N) < 0.5, np.abs(rng.exponential(b, N)), 0))
    terms['label convention'] = rng.choice([-1, 1], N) * rng.uniform(0, inputs['label_max'], N)
    terms['near-degeneracy'] = np.where(rng.random(N) < inputs['near_incidence'],
                                        rng.choice([-1, 1], N) * rng.uniform(0, inputs['near_max'], N), 0)
    terms['weighting'] = rng.choice([-1, 1], N) * rng.uniform(0, inputs['weighting_shift'], N)
    terms['diffusion element'] = rng.choice([-1, 1], N) * rng.uniform(0, inputs['diffusion_shift'], N)
    terms['grid + window'] = (rng.normal(0, inputs['grid_ci_k6']/2, N)
                              + rng.uniform(inputs['grid_win_lo'], inputs['grid_win_hi'], N))
    terms['state'] = rng.uniform(inputs['state_min'] - E_true, inputs['state_max'] - E_true, N)
    for name, key in [('state matching', 'state_match_shifts'), ('search effort', 'search_shifts')]:
        s = np.asarray(inputs[key], float)
        bound = np.full(N, s.max()) if worst_case else s[rng.integers(0, len(s), N)]
        terms[name] = rng.choice([-1, 1], N) * rng.uniform(0, 1, N) * bound
    for name in drop:
        terms.pop(name)
    tot = sum(terms.values()); E = E_true + tot; v = np.var(tot)
    rows = []
    for k, x in terms.items():
        rows.append(dict(term=k, sd=round(float(np.std(x)), 2),
                         var_share_pct=round(float(100*np.var(x)/v), 1),
                         p5=round(float(np.percentile(E_true+x, 5)), 1),
                         p95=round(float(np.percentile(E_true+x, 95)), 1)))
    rows.append(dict(term='ALL TERMS COMBINED', sd=round(float(np.std(tot)), 2), var_share_pct=100.0,
                     p5=round(float(np.percentile(E, 5)), 1), p95=round(float(np.percentile(E, 95)), 1)))
    frac = float(np.mean((E >= 30) & (E <= 70)) * 100)
    if return_E: return pd.DataFrame(rows), frac, E
    return pd.DataFrame(rows), frac


if __name__ == '__main__':
    inputs = _load_budget_inputs()
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([dict(input=k, value=(';'.join(f'{x:.2f}' for x in v) if isinstance(v, list) else round(v, 4)))
                  for k, v in inputs.items()]).to_csv(os.path.join(OUT, 'budget_inputs.csv'), index=False)
    df, frac, E = run(return_E=True)
    df['fraction_30_70_pct'] = np.nan; df['median_abs_dev'] = np.nan
    i = df.index[df.term == 'ALL TERMS COMBINED'][0]
    df.loc[i, 'fraction_30_70_pct'] = round(frac, 1)
    df.loc[i, 'median_abs_dev'] = round(float(np.median(np.abs(E-inputs['E_true']))), 1)
    df.to_csv(os.path.join(OUT, 'budget_propagation.csv'), index=False)
    print(df.to_string(index=False))
    print(f"\nE_true (mid state) = {inputs['E_true']:.2f} kJ/mol")
    print(f"fraction inside 30-70 kJ/mol: {frac:.0f} %")
    print(f"median |E-E_true|: {np.median(np.abs(E-inputs['E_true'])):.1f} kJ/mol")
    wc, frac_wc, E_wc = run(return_E=True, worst_case=True)
    wc.to_csv(os.path.join(OUT, 'budget_propagation_worstcase.csv'), index=False)
    r = wc[wc.term == 'ALL TERMS COMBINED'].iloc[0]
    print(f"worst-case variant (largest per-state shifts applied to every campaign): "
          f"5-95 % {r.p5} to {r.p95} kJ/mol, sd {r.sd}")
    # the 'state' and 'state matching' terms both act through the state dependence of E_a and are
    # therefore correlated; dropping the smaller of the two shows how much that double counting adds
    sh = df.set_index('term')['var_share_pct']
    smaller = 'state' if sh['state'] <= sh['state matching'] else 'state matching'
    dr, _ = run(drop=(smaller,))
    dr.to_csv(os.path.join(OUT, 'budget_propagation_drop_correlated.csv'), index=False)
    r2 = dr[dr.term == 'ALL TERMS COMBINED'].iloc[0]
    print(f"without the smaller correlated state term ('{smaller}'): 5-95 % {r2.p5} to {r2.p95} kJ/mol")
    print("wrote budget_propagation.csv, budget_propagation_worstcase.csv, "
          "budget_propagation_drop_correlated.csv, budget_inputs.csv")
