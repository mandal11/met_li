"""KK model-order and structure robustness (kk_robustness.csv).
Recomputes the consistency residual at 3/4/6/8 Voigt elements per decade and with the
series C or series L term removed, for representative passing and failing spectra."""
import numpy as np, pandas as pd, pickle, os
import numpy.linalg as la
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

def linkk_var(fr, z, per_decade=4, use_C=True, use_L=True):
    w = 2*np.pi*np.asarray(fr, float); z = np.asarray(z, complex)
    M = max(int(per_decade*np.log10(fr.max()/fr.min())), 8)
    taus = 1/np.logspace(np.log10(w.min())-0.5, np.log10(w.max())+0.5, M)
    cols = [np.ones_like(w, dtype=complex)] + [1/(1+1j*w*t) for t in taus]
    if use_C: cols.append(1/(1j*w))
    if use_L: cols.append(1j*w)
    A = np.array(cols).T
    Ari = np.vstack([A.real, A.imag]); b = np.concatenate([z.real, z.imag])
    x, *_ = la.lstsq(Ari, b, rcond=None); zf = A @ x
    return max(np.max(np.abs(z.real-zf.real)/np.abs(z)),
               np.max(np.abs(z.imag-zf.imag)/np.abs(z)))*100

def load_xlsx(f, sheet, scale=1.0, flip=False):
    d = pd.read_excel(f, sheet_name=sheet).dropna()
    fr = d.iloc[:, 0].astype(float).values
    z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
    i = np.argsort(fr); fr, z = fr[i], z[i]*scale
    if flip: z = z.real - 1j*np.abs(z.imag)
    return fr, z

if __name__ == '__main__':
    alldata = pickle.load(open(os.path.join(OUT, 'unisa_parsed.pkl'), 'rb'))
    cases = []
    he2 = INHOUSE_HE2
    vtc = INHOUSE_VTC5A
    if he2:
        cases += [('in-house HE2 25C (pass)', *load_xlsx(he2, '25')),
                  ('in-house HE2 -10C (fail)', *load_xlsx(he2, '-10'))]
    if vtc:
        cases += [('in-house VTC5A 15C (fail)', *load_xlsx(vtc, '15'))]
    cases += [('corpus -10C (pass)', alldata[-10][len(alldata[-10])//2][1], alldata[-10][len(alldata[-10])//2][2]),
              ('corpus 25C (pass)', alldata[25][len(alldata[25])//2][1], alldata[25][len(alldata[25])//2][2])]
    rows = []
    for name, fr, z in cases:
        v = [linkk_var(fr, z, p) for p in [3, 4, 6, 8]] + \
            [linkk_var(fr, z, 4, use_C=False), linkk_var(fr, z, 4, use_L=False)]
        cls = 'PASS' if all(x < 2 for x in v) else ('FAIL' if all(x >= 2 for x in v) else 'INCONSISTENT')
        rows.append(dict(spectrum=name, m3=round(v[0], 2), m4=round(v[1], 2), m6=round(v[2], 2),
                         m8=round(v[3], 2), no_series_C=round(v[4], 2), no_series_L=round(v[5], 2),
                         classification=cls))
    df = pd.DataFrame(rows)
    complete = (he2 is not None and vtc is not None and len(rows) == 5)
    safe_write(df, 'kk_robustness.csv', complete,
               "data/inhouse/LGHE2.xlsx and data/inhouse/all_temp_sony.xlsx (S1, DOI [A]) "
               "for the three in-house rows")
    print(df.to_string(index=False))
