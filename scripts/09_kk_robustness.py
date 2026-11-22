"""KK model-order and structure robustness (table, residual plots, and Excel details).
Recomputes the consistency residual at 3/4/6/8 Voigt elements per decade and with the
series C or series L term removed, for representative passing and failing spectra.
Also plots and exports the full real and imaginary residual spectra for every loaded input."""
import numpy as np, pandas as pd, pickle, os
import numpy.linalg as la
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

def linkk_var(fr, z, per_decade=4, use_C=True, use_L=True, ret_spec=False):
    w = 2*np.pi*np.asarray(fr, float); z = np.asarray(z, complex)
    M = max(int(per_decade*np.log10(fr.max()/fr.min())), 8)
    taus = 1/np.logspace(np.log10(w.min())-0.5, np.log10(w.max())+0.5, M)
    cols = [np.ones_like(w, dtype=complex)] + [1/(1+1j*w*t) for t in taus]
    if use_C: cols.append(1/(1j*w))
    if use_L: cols.append(1j*w)
    A = np.array(cols).T
    Ari = np.vstack([A.real, A.imag]); b = np.concatenate([z.real, z.imag])
    x, *_ = la.lstsq(Ari, b, rcond=None); zf = A @ x
    rr = (z.real-zf.real)/np.abs(z)*100
    ri = (z.imag-zf.imag)/np.abs(z)*100
    if ret_spec:
        return rr, ri
    return max(np.max(np.abs(rr)), np.max(np.abs(ri)))

def plot_residuals(cases, complete):
    if not cases:
        print("[skip] no spectra available for KK residual plot")
        return
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = int(np.ceil(len(cases)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.3*nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax, (name, fr, z) in zip(axes.flat, cases):
        rr, ri = linkk_var(fr, z, per_decade=4, ret_spec=True)
        ax.semilogx(fr, rr, '.-', ms=3, lw=0.8, label='Re')
        ax.semilogx(fr, ri, '.-', ms=3, lw=0.8, label='Im')
        ax.axhline(0, color='black', lw=0.6)
        ax.axhspan(-2, 2, color='gray', alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('KK residual (% of |Z|)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cases):]:
        ax.set_visible(False)
    fig.suptitle('KK residual spectra (4 Voigt elements per decade)', y=1.01)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    suffix = '' if complete else '.partial'
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT, f'kk_robustness_residuals{suffix}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path)
    plt.close(fig)

def write_residual_excel(cases, complete):
    if not cases:
        print("[skip] no spectra available for KK residual workbook")
        return
    os.makedirs(OUT, exist_ok=True)
    suffix = '' if complete else '.partial'
    path = os.path.join(OUT, f'kk_robustness_residuals{suffix}.xlsx')
    summaries = []
    with pd.ExcelWriter(path) as writer:
        for i, (name, fr, z) in enumerate(cases, start=1):
            rr, ri = linkk_var(fr, z, per_decade=4, ret_spec=True)
            abs_z = np.abs(z)
            fit_real = z.real - rr*abs_z/100
            fit_imag = z.imag - ri*abs_z/100
            detail = pd.DataFrame({
                'frequency_Hz': fr,
                'Z_measured_real_ohm': z.real,
                'Z_measured_imag_ohm': z.imag,
                'Z_fit_real_ohm': fit_real,
                'Z_fit_imag_ohm': fit_imag,
                'Re_residual_pct_of_absZ': rr,
                'Im_residual_pct_of_absZ': ri,
                'abs_Re_residual_pct': np.abs(rr),
                'abs_Im_residual_pct': np.abs(ri),
                'max_component_abs_residual_pct': np.maximum(np.abs(rr), np.abs(ri)),
            })
            detail.to_excel(writer, sheet_name=f'Spectrum {i}', index=False)
            re_idx = int(np.argmax(np.abs(rr)))
            im_idx = int(np.argmax(np.abs(ri)))
            max_idx = int(np.argmax(np.maximum(np.abs(rr), np.abs(ri))))
            summaries.append({
                'spectrum': name,
                'frequency_points': len(fr),
                'frequency_min_Hz': fr.min(),
                'frequency_max_Hz': fr.max(),
                'max_abs_Re_residual_pct': abs(rr[re_idx]),
                'frequency_at_max_abs_Re_Hz': fr[re_idx],
                'max_abs_Im_residual_pct': abs(ri[im_idx]),
                'frequency_at_max_abs_Im_Hz': fr[im_idx],
                'max_component_abs_residual_pct': max(detail['max_component_abs_residual_pct']),
                'frequency_at_max_component_Hz': fr[max_idx],
            })
        pd.DataFrame(summaries).to_excel(writer, sheet_name='Summary', index=False)
    print('wrote', path)

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
    plot_residuals(cases, complete)
    write_residual_excel(cases, complete)
    print(df.to_string(index=False))
