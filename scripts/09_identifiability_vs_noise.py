"""Noise injection on the MEASURED reference-corpus spectra (manuscript section 3.2).

What it does
------------
Earlier releases studied the noise dependence of the identifiability limit on synthetic
two-arc spectra. This script asks the same question of the real data instead. For every
spectrum retained for the state-resolved regression (script 02: OCV-matched states, one
triplicate-median spectrum per state and temperature) proportional Gaussian noise is added
to the measured real and imaginary parts,

    Z_noisy(w) = Z_meas(w) + sigma * |Z_meas(w)| * (n_re + j n_im),   n ~ N(0, 1),

at sigma = 0.2, 0.5, 1 and 2 % (the measured triplicate repeatability of the corpus is
about 0.3 %), and every noisy spectrum is refitted with the released procedure
(eislib.fit_ecm, finite-space Warburg and series inductance, no warm start). The reference
for each spectrum is the same procedure applied to the unperturbed measured spectrum, so
the test measures how far added noise moves the extraction from where it lands on the
real data. Noise is generated with a fixed seed per (spectrum, level, realization), so the
study is deterministic and independent of the number of worker processes.

Outputs
-------
  out/noise_injection_reference.csv  reference fit of every retained spectrum (tau2/tau1)
  out/noise_injection_spectra.csv    one row per (spectrum, noise level, realization)
  out/identifiability_vs_noise.csv   gross-failure rate (|dR_ct|/R_ct > 50 %) and median
                                     |dR_ct|/R_ct by noise level, overall and by the
                                     measured separation tau2/tau1 of the reference fit;
                                     the same error measures for R_SEI + R_ct
  out/noise_injection_campaigns.csv  E_a of the mid state (state 5) from complete noisy
                                     campaigns (all eleven setpoints re-noised and refitted),
                                     for R_ohm, R_SEI, R_ct and the swap-invariant sum
  figs/FigS1_noise_injection.png/.pdf

Run time: about 1 400 fits (~2 CPU-hours; noisy spectra converge more slowly). The work is spread over all CPU cores; set the
environment variable EIS_WORKERS to limit the number of processes, and EIS_NOISE_QUICK=1 for
a short smoke test (2 realizations per spectrum). Every finished fit is appended to a
checkpoint file in out/.cache_noise/, so an interrupted run resumes where it stopped.
"""
import os, sys, time
import numpy as np, pandas as pd
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, branch_tau, arrhenius, load_selected_spectra

NOISES = [0.2, 0.5, 1.0, 2.0]                 # % of |Z|, per component
MID = 5                                       # mid state (OCV ~3.67 V)
QUICK = os.environ.get('EIS_NOISE_QUICK', '0') == '1'
N_REP_MID = 2 if QUICK else 20                # realizations per mid-state spectrum (= campaigns)
N_REP_OTHER = 1                               # realizations per spectrum of the other states
SEP_BINS = [0, 10, 20, 50, 200, np.inf]
SEP_LABELS = ['<10', '10-20', '20-50', '50-200', '>200']


def _fit_task(args):
    """Worker: fit one (possibly noisy) spectrum. Returns a flat dict."""
    key, fr, z, sigma, seed = args
    if sigma > 0:
        rng = np.random.default_rng(seed)
        z = z + sigma/100*np.abs(z)*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))
    p, rms = fit_ecm(fr, z, with_L=True)
    t1, t2 = branch_tau(p)
    return dict(key=key, Rohm=p[0], Rsei=p[1], Rct=p[4], rms=rms, sep=t2/t1)


def _seed(ridx, T, level, rep):
    return 1_000_000*ridx + 10_000*(T+20) + 100*level + rep


def _code_digest(path):
    """Digest of a Python source that ignores comments, docstrings and formatting."""
    import ast
    tree = ast.parse(open(path, encoding='utf-8').read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], 'value', None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree).encode()


def _cache_file():
    """Checkpoint file for finished fits. Its name is a hash of everything a fit result
    depends on (fit library code, noise generator, seeds, noise levels, input data), so a
    stale checkpoint is never reused; an interrupted run resumes where it stopped. Comments
    and docstrings of the fit library do not enter the hash."""
    import hashlib, inspect
    h = hashlib.sha1()
    h.update(_code_digest(os.path.join(REPO_ROOT, 'eislib', 'eislib.py')))
    for f in [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_arrhenius_input.csv')]:
        h.update(open(f, 'rb').read())
    for fn in (_fit_task, _seed):
        h.update(inspect.getsource(fn).encode())
    h.update(repr(NOISES).encode())
    d = os.path.join(OUT, '.cache_noise'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'fits_{h.hexdigest()[:12]}.jsonl')


def _run_cached(tasks, cache):
    """Fit all tasks (in parallel), appending every finished fit to the checkpoint file."""
    import json
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            try:
                r = json.loads(line); done[tuple(r['key'])] = r
            except ValueError:
                pass                                   # line cut off by an interrupted run
    todo = [t for t in tasks if tuple(t[0]) not in done]
    if todo:
        n = int(os.environ.get('EIS_WORKERS', '0')) or os.cpu_count() or 1
        t0 = time.time()
        with open(cache, 'a') as fh:
            def keep(i, r):
                r = dict(r, key=list(r['key'])); done[tuple(r['key'])] = r
                fh.write(json.dumps(r) + '\n'); fh.flush()
                if (i+1) % 50 == 0 or i+1 == len(todo):
                    el = time.time()-t0
                    print(f"    {i+1}/{len(todo)} fits, {el/60:.1f} min, "
                          f"about {el/(i+1)*(len(todo)-i-1)/60:.0f} min left", flush=True)
            if n <= 1:
                for i, t in enumerate(todo):
                    keep(i, _fit_task(t))
            else:
                import multiprocessing as mp
                with mp.get_context('spawn').Pool(n) as pool:
                    for i, r in enumerate(pool.imap_unordered(_fit_task, todo, chunksize=1)):
                        keep(i, r)
    return [dict(done[tuple(t[0])], key=tuple(t[0])) for t in tasks]


def _figure(spec, ref, rows, camp, fig_dir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    # (a) what the added noise looks like on a measured spectrum (mid state, coldest T)
    s = min((x for x in spec if x['ridx'] == MID), key=lambda x: x['T'])
    rng = np.random.default_rng(_seed(MID, s['T'], 3, 0))
    zn = s['z'] + 0.02*np.abs(s['z'])*(rng.standard_normal(len(s['z'])) + 1j*rng.standard_normal(len(s['z'])))
    ax[0].plot(s['z'].real*1e3, -s['z'].imag*1e3, 'k.-', ms=4, lw=.8, label='measured')
    ax[0].plot(zn.real*1e3, -zn.imag*1e3, 'o', ms=3.5, mfc='none', color='tab:red',
               label='measured + 2 % noise')
    ax[0].set_xlabel("$Z'$ (m$\\Omega$)"); ax[0].set_ylabel("$-Z''$ (m$\\Omega$)")
    ax[0].set_title(f"(a) State {MID}, {s['T']} °C: one 2 % realization"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    # (b) gross-failure rate by measured separation and noise level
    d = pd.DataFrame(rows)
    d['bin'] = pd.cut(d['sep_ref'], SEP_BINS, labels=SEP_LABELS, right=False)
    for lev, mk in zip(NOISES, ['o', 's', '^', 'D']):
        g = d[d.noise_pct == lev].groupby('bin', observed=False)['gross_failure'].mean()*100
        ax[1].plot(range(len(SEP_LABELS)), g.reindex(SEP_LABELS).values, mk+'-', ms=4,
                   label=f'{lev:g} % noise')
    ax[1].set_xticks(range(len(SEP_LABELS))); ax[1].set_xticklabels(SEP_LABELS)
    ax[1].set_xlabel('measured $\\tau_2/\\tau_1$ of the reference fit')
    ax[1].set_ylabel('gross failures (% of refits)')
    ax[1].set_title('(b) Measured spectra, all retained states'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    # (c) E_a of the mid state from complete noisy campaigns: R_ct and the swap-invariant sum
    c = pd.DataFrame(camp)
    pos = np.arange(len(NOISES))
    for name, col, dx, lab in [('Rct', 'tab:blue', -0.18, '$R_{ct}$'), ('Rsum', 'tab:green', 0.18, '$R_{SEI}+R_{ct}$')]:
        data = [c[c.noise_pct == lev]['Ea_' + name].values for lev in NOISES]
        bp = ax[2].boxplot(data, positions=pos + dx, widths=0.3, patch_artist=True, manage_ticks=False)
        for b in bp['boxes']:
            b.set_facecolor(col); b.set_alpha(.35)
        ax[2].axhline(ref['Ea_' + name], color=col, ls='--', lw=1, label=f'{lab}: unperturbed data')
        ax[2].axhspan(ref['Ea_' + name] - ref['CI_' + name], ref['Ea_' + name] + ref['CI_' + name], color=col,
                      alpha=.08)
    ax[2].set_xticks(pos); ax[2].set_xticklabels([f'{lev:g}' for lev in NOISES])
    ax[2].set_xlabel('added noise (% of |Z|)'); ax[2].set_ylabel('$E_a$ (kJ mol$^{-1}$)')
    ax[2].set_title(f'(c) State {MID}: {len(c[c.noise_pct == NOISES[0]])} campaigns per level')
    ax[2].legend(fontsize=8, loc='upper left'); ax[2].grid(alpha=.3)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(fig_dir, f'FigS1_noise_injection.{ext}'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    need = [os.path.join(OUT, 'unisa_parsed.pkl'), os.path.join(OUT, 'unisa_arrhenius_input.csv')]
    if not all(os.path.exists(f) for f in need):
        print("[abort] run 01_parse_unisa.py and 02_fit_unisa.py first -> "
              "identifiability_vs_noise.csv NOT written."); sys.exit(0)
    t0 = time.time()
    spec = load_selected_spectra(OUT)
    print(f"{len(spec)} retained spectra, {len({s['ridx'] for s in spec})} states")

    cache = _cache_file()
    print(f"checkpoint: {os.path.relpath(cache, REPO_ROOT)}")
    # reference fits (unperturbed measured spectra, same procedure as the refits)
    tasks = [((s['ridx'], s['T'], -1, 0), s['fr'], s['z'], 0.0, 0) for s in spec]
    refs = {r['key'][:2]: r for r in _run_cached(tasks, cache)}

    tasks = []
    for s in spec:
        nrep = N_REP_MID if s['ridx'] == MID else N_REP_OTHER
        for li, lev in enumerate(NOISES):
            for rep in range(nrep):
                tasks.append(((s['ridx'], s['T'], li, rep), s['fr'], s['z'], lev,
                              _seed(s['ridx'], s['T'], li, rep)))
    # mid-state campaigns first, then the other states realization by realization, so that an
    # interrupted run has complete campaigns and an even design over the other states
    tasks.sort(key=lambda t: (t[0][0] != MID, t[0][3], t[0][0], t[0][1], t[0][2]))
    print(f"{len(tasks)} noisy refits ...", flush=True)
    res = _run_cached(tasks, cache)

    rows = []
    for r in res:
        ridx, T, li, rep = r['key']; ref = refs[(ridx, T)]
        e_ct = (r['Rct']-ref['Rct'])/ref['Rct']; e_sei = (r['Rsei']-ref['Rsei'])/ref['Rsei']
        s_ref = ref['Rsei'] + ref['Rct']; e_sum = (r['Rsei'] + r['Rct'] - s_ref)/s_ref
        rows.append(dict(ridx=ridx, T=T, noise_pct=NOISES[li], rep=rep,
                         Rohm=r['Rohm'], Rsei=r['Rsei'], Rct=r['Rct'], Rsum=r['Rsei']+r['Rct'],
                         rms=r['rms'], sep=r['sep'],
                         sep_ref=ref['sep'], rel_err_Rct=e_ct, rel_err_Rsei=e_sei, rel_err_Rsum=e_sum,
                         rel_err_Rohm=(r['Rohm']-ref['Rohm'])/ref['Rohm'],
                         gross_failure=bool(abs(e_ct) > 0.5), gross_failure_Rsum=bool(abs(e_sum) > 0.5)))
    d = pd.DataFrame(rows)
    pd.DataFrame([dict(ridx=k[0], T=k[1], Rohm=v['Rohm'], Rsei=v['Rsei'], Rct=v['Rct'],
                       rms=v['rms'], sep=v['sep']) for k, v in sorted(refs.items())]
                 ).to_csv(os.path.join(OUT, 'noise_injection_reference.csv'), index=False)
    d.to_csv(os.path.join(OUT, 'noise_injection_spectra.csv'), index=False)

    # summary: overall and by measured separation
    d['bin'] = pd.cut(d['sep_ref'], SEP_BINS, labels=SEP_LABELS, right=False)
    summ = []
    for lev in NOISES:
        x = d[d.noise_pct == lev]
        row = dict(noise_pct=lev, n_refits=len(x),
                   gross_failure_pct=round(100*x.gross_failure.mean(), 1),
                   median_abs_err_Rct_pct=round(100*x.rel_err_Rct.abs().median(), 2),
                   p95_abs_err_Rct_pct=round(100*x.rel_err_Rct.abs().quantile(.95), 1),
                   median_abs_err_Rsei_pct=round(100*x.rel_err_Rsei.abs().median(), 2),
                   median_abs_err_Rsum_pct=round(100*x.rel_err_Rsum.abs().median(), 2),
                   p95_abs_err_Rsum_pct=round(100*x.rel_err_Rsum.abs().quantile(.95), 1),
                   gross_failure_Rsum_pct=round(100*x.gross_failure_Rsum.mean(), 1),
                   median_abs_err_Rohm_pct=round(100*x.rel_err_Rohm.abs().median(), 3))
        for lab in SEP_LABELS:
            xb = x[x.bin == lab]
            row[f'fail_pct_sep{lab}'] = round(100*xb.gross_failure.mean(), 1) if len(xb) else np.nan
            row[f'n_sep{lab}'] = len(xb)
        summ.append(row)
    summ = pd.DataFrame(summ)
    safe_write(summ, 'identifiability_vs_noise.csv', True,
               "out/unisa_parsed.pkl and out/unisa_arrhenius_input.csv")
    print(summ.to_string(index=False))

    # campaigns: mid state, all setpoints re-noised; realization r of every T = campaign r
    mid = d[d.ridx == MID]
    temps = sorted(mid['T'].unique()); TK = np.array(temps)+273.15
    refrow = {}
    for name in ['Rohm', 'Rsei', 'Rct', 'Rsum']:
        vals = [refs[(MID, T)]['Rsei'] + refs[(MID, T)]['Rct'] if name == 'Rsum' else refs[(MID, T)][name]
                for T in temps]
        Ea, ci, _ = arrhenius(TK, vals)
        refrow['Ea_'+name], refrow['CI_'+name] = Ea, ci
    camp = []
    for lev in NOISES:
        m = mid[mid.noise_pct == lev]
        for rep in sorted(m.rep.unique()):
            mm = m[m.rep == rep].set_index('T')
            if len(mm) != len(temps):
                continue
            row = dict(noise_pct=lev, campaign=int(rep))
            for name in ['Rohm', 'Rsei', 'Rct', 'Rsum']:
                row['Ea_'+name] = arrhenius(TK, mm.loc[temps, name].values)[0]
            camp.append(row)
    c = pd.DataFrame(camp)
    c.to_csv(os.path.join(OUT, 'noise_injection_campaigns.csv'), index=False)
    print(f"\nmid state, unperturbed: Ea(Rohm/Rsei/Rct) = {refrow['Ea_Rohm']:.2f} / "
          f"{refrow['Ea_Rsei']:.2f} / {refrow['Ea_Rct']:.2f} ± {refrow['CI_Rct']:.2f} kJ/mol")
    print(f"                         Ea(Rsei+Rct) = {refrow['Ea_Rsum']:.2f} ± {refrow['CI_Rsum']:.2f} kJ/mol")
    for lev in NOISES:
        for name in ['Rct', 'Rsum']:
            x = c[c.noise_pct == lev]['Ea_' + name]
            print(f"  {lev:>4g} % noise: Ea({name}) median {x.median():6.2f}, sd {x.std():5.2f}, "
                  f"2.5-97.5 % [{x.quantile(.025):6.2f}, {x.quantile(.975):6.2f}] kJ/mol "
                  f"({len(x)} campaigns)")
    _figure(spec, refrow, rows, camp, os.path.join(REPO_ROOT, 'figs'))
    print(f"wrote figs/FigS1_noise_injection.png/.pdf   ({time.time()-t0:.0f} s)")
