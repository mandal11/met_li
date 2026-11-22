"""Model/weighting sensitivity of the mid-state activation energies (Table 5).

Fits the OCV-matched mid state (state 5, the Table 3 row) of the reference corpus under
four variants and reports the resulting activation-energy shifts:
  - reference: modulus (1/|Z|) weighting, finite-space reflective (coth) Warburg;
  - unit weighting: identical model, unweighted real/imaginary residuals;
  - semi-infinite Warburg: modulus weighting, Z_W = R_W0 (j A_W0 w)^(-1/2);
  - no series inductance: the reference model without L (section 2.3).
All four use the released fitting procedure (eislib.fit_ecm: identical starting set, seeds,
bounds, warm -> cold chaining and tau-canonicalization) with only the residual weighting or
the diffusion element changed, so the deltas isolate the stated choice and nothing else.
The reported residual is always the modulus-weighted rms, so fit quality is comparable.
"""
import numpy as np, pandas as pd, os, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, arrhenius, load_selected_spectra

MID = 5

if __name__ == '__main__':
    need = [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_arrhenius_input.csv')]
    if not all(os.path.exists(f) for f in need):
        print("[abort] run 01_parse_unisa.py and 02_fit_unisa.py first -> "
              "model_weighting_sensitivity.csv NOT written.")
        print("        The released CSV has been left untouched."); sys.exit(0)
    series = {s['T']: (s['fr'], s['z']) for s in load_selected_spectra(OUT, ridx=MID)}
    rows = []
    for name, weight, warburg, with_L in [
            ('reference (modulus weighting, finite-space reflective Warburg)', 'modulus', 'coth', True),
            ('unit weighting', 'unit', 'coth', True),
            ('semi-infinite Warburg', 'modulus', 'semi', True),
            ('no series inductance', 'modulus', 'coth', False)]:
        prev = None; rec = []
        for T in sorted(series, reverse=True):        # warm -> cold chaining
            fr, z = series[T]
            p, rms = fit_ecm(fr, z, p0=prev, with_L=with_L, weighting=weight, warburg=warburg)
            prev = p.copy(); rec.append((T, p, rms))
        TK = [T+273.15 for T, _, _ in rec]
        ea = {k: arrhenius(TK, [p[i] for _, p, _ in rec])[0] for k, i in
              [('Ea_Rohm', 0), ('Ea_Rsei', 1), ('Ea_Rct', 4)]}
        rows.append(dict(variant=name, Ea_Rohm=round(ea['Ea_Rohm'], 2),
                         Ea_Rsei=round(ea['Ea_Rsei'], 2), Ea_Rct=round(ea['Ea_Rct'], 2),
                         median_rms_pct=round(float(np.median([r for _, _, r in rec])), 3)))
        print(f"  {name}: done", flush=True)
    df = pd.DataFrame(rows)
    safe_write(df, 'model_weighting_sensitivity.csv', len(df) == 4,
               "the reference corpus parsed to out/unisa_parsed.pkl")
    print(df.to_string(index=False))
    b = df.iloc[0]
    for v in df.variant[1:]:
        r = df[df.variant == v].iloc[0]
        print(f"  delta vs reference [{v}]: Rohm {abs(b.Ea_Rohm-r.Ea_Rohm):.2f}  "
              f"Rsei {abs(b.Ea_Rsei-r.Ea_Rsei):.2f}  Rct {abs(b.Ea_Rct-r.Ea_Rct):.2f}")
