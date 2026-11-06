"""Lin-KK audit of every locally available spectrum. Missing sources are skipped."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import linkk
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

def load(f, s, scale=1.0, imag_flip=False):
    d = pd.read_excel(f, sheet_name=s).dropna()
    fr = d.iloc[:, 0].astype(float).values
    z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
    i = np.argsort(fr); fr, z = fr[i], z[i]*scale
    if imag_flip: z = z.real - 1j*np.abs(z.imag)
    return fr, z

TEMPS = ['-10', '0', '15', '25', '40', '50']
rows = []
print("== in-house 0.5 A campaign ==")
for f, lab in [(INHOUSE_HE2, 'in-house HE2'), (INHOUSE_VTC5A, 'in-house VTC5A')]:
    if f is None:
        print(f"  [skip] {lab}: file not found (set its path in 00_paths.py)"); continue
    for t in TEMPS:
        try:
            fr, z = load(f, t)
        except Exception as e:
            print(f"  [skip] {lab} {t}C: {e}"); continue
        kk = linkk(fr, z); vp = 0.5*np.abs(z).max()*1e3
        print(f"  {lab:16s} {t:>4s}C  KKmax={kk:5.2f}%  Vpeak={vp:6.1f} mV")
        rows.append(dict(source=lab, T=t, KK=kk, Vpeak_mV=vp))

print("== 6-temperature files (stored in mOhm, -Im positive) ==")
for f, lab in [(SIXT_FILE_A, '6-T file A'), (SIXT_FILE_B, '6-T file B')]:
    if f is None:
        print(f"  [skip] {lab}: file not found"); continue
    for t in TEMPS:
        try:
            fr, z = load(f, t, 1e-3, True)
        except Exception as e:
            print(f"  [skip] {lab} {t}C: {e}"); continue
        kk = linkk(fr, z)
        print(f"  {lab:16s} {t:>4s}C  KKmax={kk:5.2f}%")
        rows.append(dict(source=lab, T=t, KK=kk, Vpeak_mV=float('nan')))

print("== reference corpus (mid-state per temperature) ==")
pkl = os.path.join(OUT, 'unisa_parsed.pkl')
if os.path.exists(pkl):
    alldata = pickle.load(open(pkl, 'rb'))
    for T, specs in sorted(alldata.items()):
        s = specs[len(specs)//2]
        kk = linkk(s[1], s[2]); vp = 0.15*np.abs(s[2]).max()*1e3
        print(f"  UNISA {T:>4d}C  KKmax={kk:5.2f}%  Vpeak={vp:5.1f} mV")
        rows.append(dict(source='UNISA', T=T, KK=kk, Vpeak_mV=vp))
    # triplicate repeatability of the corpus (quoted in section 3.3): coefficient of variation of
    # the three repeats of Z'(0.1 Hz) for every state and temperature
    rep = []
    for T, specs in sorted(alldata.items()):
        for gi in range(0, len(specs), 3):
            g = specs[gi:gi+3]; j = int(np.argmin(np.abs(g[0][1]-0.1)))
            zr = np.array([x[2].real[j] for x in g])
            rep.append(dict(T=T, group=gi//3, cv_Zreal_0p1Hz_pct=100*zr.std(ddof=1)/zr.mean()))
    rep = pd.DataFrame(rep); rep.to_csv(os.path.join(OUT, 'unisa_repeatability.csv'), index=False)
    print(f"  triplicate repeatability of Z'(0.1 Hz): median CV {rep.cv_Zreal_0p1Hz_pct.median():.2f} % "
          f"over {len(rep)} state/temperature groups")
else:
    print("  [skip] out/unisa_parsed.pkl not found - run 01_parse_unisa.py first")

if True:
    _complete = (len(rows) == 35)      # 12 in-house + 12 six-setpoint + 11 corpus
    safe_write(pd.DataFrame(rows), 'kk_audit.csv', _complete,
               "all in-house and six-setpoint workbooks plus the reference corpus "
               "(35 spectra: 12 + 12 + 11)")
else:
    print("nothing audited - no data sources available")
