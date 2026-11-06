"""Monte-Carlo identifiability study.

Design note: both branch time constants are kept inside the measured band by placing
them symmetrically about its geometric centre (tau_geo = 5 ms ~ 32 Hz for a
0.1 Hz-10 kHz window).  An earlier version scaled tau_2 upward with the separation,
which pushed the slow arc below the lowest measured frequency and confounded
"arc separation" with "window truncation"; the failure rates were then non-monotonic
in separation.  With the corrected design the study isolates separation alone.

Reports the gross-failure rate (fitted R_ct in error by >50%) for four procedures:
single start / single start + tau-ordering / six starts / six starts + tau-ordering.
"""
import numpy as np, pandas as pd, os
from scipy.optimize import least_squares
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

FR = np.logspace(-1, 4, 41); W = 2*np.pi*FR
TAU_GEO = 5e-3
LB = [1e-4, 1e-6, 1e-5, 0.5, 1e-6, 1e-5, 0.5]
UB = [1, 1, 1e3, 1, 1, 1e3, 1]

def z2rc(p, w):
    Rohm, R1, t1, P1, R2, t2, P2 = p; jw = 1j*w
    return Rohm + R1/(1+(t1*jw)**P1) + R2/(1+(t2*jw)**P2)

rows = []
print(f"{'tau2/tau1':>10s}{'f_c1 (Hz)':>11s}{'f_c2 (Hz)':>11s}"
      f"{'1 start':>10s}{'1st+order':>11s}{'6 starts':>10s}{'6st+order':>11s}")
for sep in [2, 3, 5, 10, 30, 100]:
    rng = np.random.default_rng(1)
    t1, t2 = TAU_GEO/np.sqrt(sep), TAU_GEO*np.sqrt(sep)
    truth = [0.045, 0.006, t1, 0.85, 0.010, t2, 0.9]
    c = [0, 0, 0, 0]; n = 200
    for _ in range(n):
        z = z2rc(truth, W)
        z = z + 0.005*np.abs(z)*(rng.standard_normal(len(W)) + 1j*rng.standard_normal(len(W)))
        def resid(p):
            e = (z2rc(p, W)-z)/np.abs(z); return np.concatenate([e.real, e.imag])
        starts = [[0.045, 0.008*rng.uniform(.3, 3), 10**rng.uniform(-4, 0), 0.85,
                   0.008*rng.uniform(.3, 3), 10**rng.uniform(-4, 0), 0.9] for _ in range(6)]
        sols = []
        for p0 in starts:
            try: sols.append(least_squares(resid, np.clip(p0, LB, UB), bounds=(LB, UB), max_nfev=3000))
            except Exception: pass
        if not sols: continue
        for j, r in enumerate([sols[0], min(sols, key=lambda s: s.cost)]):
            R1f, t1f, R2f, t2f = r.x[1], r.x[2], r.x[4], r.x[5]
            c[2*j]   += abs(R2f-truth[4])/truth[4] > 0.5
            Rc2 = R1f if t1f > t2f else R2f
            c[2*j+1] += abs(Rc2-truth[4])/truth[4] > 0.5
    print(f"{sep:>10d}{1/(2*np.pi*t1):>11.1f}{1/(2*np.pi*t2):>11.2f}"
          f"{c[0]/n*100:>10.1f}{c[1]/n*100:>11.1f}{c[2]/n*100:>10.1f}{c[3]/n*100:>11.1f}")
    rows.append(dict(sep=sep, fc1_Hz=round(1/(2*np.pi*t1), 1), fc2_Hz=round(1/(2*np.pi*t2), 2),
                     one_start=c[0]/n*100, one_start_ordered=c[1]/n*100,
                     six_starts=c[2]/n*100, six_starts_ordered=c[3]/n*100))
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'mc_identifiability.csv'), index=False)
print("wrote", os.path.join(OUT, 'mc_identifiability.csv'))
