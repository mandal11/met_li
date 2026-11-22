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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, arrhenius, zmodel

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
    kk_audit_path = os.path.join(OUT, 'kk_audit.csv')
    if not os.path.exists(kk_audit_path):
        raise SystemExit(f"{kk_audit_path} not found - run 04_kk_audit.py first.")
    kk_audit = pd.read_csv(kk_audit_path)
    required = {'source', 'T', 'KK_status'}
    if not required.issubset(kk_audit.columns):
        raise SystemExit(f"{kk_audit_path} must contain columns {sorted(required)}.")
    kk_sources = {'file A': '6-T file A', 'file B': '6-T file B',
                  'in-house HE2': 'in-house HE2', 'in-house VTC5A': 'in-house VTC5A'}
    expected_temps = {float(t) for t in SHEETS}
    rows = []
    complete = True
    for name, f, sc, fl in datasets:
        if f is None:
            print(f"[skip] {name}: data not present")
            complete = False
            continue
        audited = kk_audit[kk_audit['source'].astype(str) == kk_sources[name]]
        if audited.empty:
            print(f"[skip] {name}: no matching results in kk_audit.csv")
            complete = False
            continue
        status = audited['KK_status'].fillna('').astype(str).str.strip().str.lower()
        audited_temps = set(pd.to_numeric(audited['T'], errors='coerce').dropna().astype(float))
        missing_temps = expected_temps - audited_temps
        failed = status == 'fail'
        if not status.isin(['pass', 'fail']).all() or missing_temps:
            complete = False
        if failed.any():
            failed_temps = audited.loc[failed, 'T'].astype(str).tolist()
            print(f"[skip] {name}: KK test failed at {len(failed_temps)} spectrum(s), "
                  f"T={', '.join(failed_temps)}")
            continue
        if not status.eq('pass').all():
            print(f"[skip] {name}: KK results are incomplete or not all pass"
                  + (f"; missing T={sorted(missing_temps)}" if missing_temps else ""))
            continue
        fit_dir = os.path.join(OUT, 'ecm_fit_results', os.path.splitext(os.path.basename(f))[0])
        os.makedirs(fit_dir, exist_ok=True)
        prev = None; rec = []; fit_rows = []
        for t in SHEETS:                                   # warm -> cold, as in script 11
            t_c = float(t)
            fr, z = load(f, t, sc, fl)
            p, rms_pct = fit_ecm(fr, z, p0=prev); prev = p.copy()
            zfit = zmodel(p, 2*np.pi*fr)
            error = zfit - z
            rmse_ohm = float(np.sqrt(np.mean(np.abs(error)**2)))
            fit_row = dict(source=name, input_file=os.path.basename(f), T_C=t_c,
                           Rohm_ohm=p[0], Rsei_ohm=p[1], A1=p[2], P1=p[3],
                           Rct_ohm=p[4], A2=p[5], P2=p[6], Rw0_ohm=p[7],
                           Aw0=p[8], rms_normalized_pct=rms_pct, rmse_complex_ohm=rmse_ohm)
            fit_rows.append(fit_row)
            rec.append((t_c, p[1], p[4]))

            pd.DataFrame({
                'frequency_Hz': fr,
                'Z_real_ohm': z.real,
                'Z_imag_ohm': z.imag,
                'Zfit_real_ohm': zfit.real,
                'Zfit_imag_ohm': zfit.imag,
                'absolute_error_ohm': np.abs(error)
            }).to_csv(os.path.join(fit_dir, f'fit_data_T{t_c:g}C.csv'), index=False)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(z.real*1e3, -z.imag*1e3, 'o', markersize=3, label='data')
            ax.plot(zfit.real*1e3, -zfit.imag*1e3, '-', lw=1.8, label='ECM fit')
            ax.set_xlabel("Z' (mΩ)")
            ax.set_ylabel("-Z'' (mΩ)")
            ax.set_title(f'{name} — T={t_c:g} °C\n'
                         f'normalized RMS={rms_pct:.3f}%, complex RMSE={rmse_ohm*1e3:.4g} mΩ')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(fit_dir, f'ecm_fit_T{t_c:g}C.png'), dpi=200)
            plt.close(fig)

        pd.DataFrame(fit_rows).to_csv(os.path.join(fit_dir, 'ecm_fit_summary.csv'), index=False)
        print(f"  ECM fit details and plots saved in: {fit_dir}", flush=True)
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
    safe_write(df, 'label_enumeration.csv', complete,
               "all four six-setpoint datasets checked against kk_audit.csv "
               "(S1/S2, DOI [A])")
