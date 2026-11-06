"""Large-signal (Butler-Volmer) forward model: amplitude bias of R_ct and E_a (section 2.6, 3.1).

Kinetics-only, symmetric one-electron Butler-Volmer interface under a sinusoidal
galvanostatic drive I(t) = I_ac sin(wt); R_ct(true) is prescribed through
R_ct(T) = R_ct(25 C) exp[(Ea/Rg)(1/T - 1/298.15)] and the lumped exchange current is
I0 = RgT/(F R_ct). eislib.bv_ratio returns R_ct(measured, fundamental)/R_ct(true).

Writes (all values quoted in the manuscript come from these files):
  out/bv_amplitude_bias.csv    measured E_a and bias for I_ac = 0.5, 0.2, 0.05, 0.03 A
                               (in-house grid -10...50 C), true E_a = 50 and 35 kJ/mol
  out/bv_amplitude_limits.csv  largest I_ac keeping the R_ct error <= 1 % at -10 C for
                               R_ct(25 C) = 8, 14, 25 mOhm (E_a = 50 kJ/mol)
  out/bv_ranking_example.csv   two cells (50 kJ/mol, 8 mOhm) and (45 kJ/mol, 14 mOhm)
                               measured at a common 0.5 A: true and apparent E_a gap
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import linregress
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import bv_ratio, RG, F_CONST

TEMPS = np.array([263.15, 273.15, 288.15, 298.15, 313.15, 323.15])   # in-house grid


def rct_true(T, Ea, R25):
    return R25*np.exp(Ea/RG*(1/T-1/298.15))


def measured_ea(I, Ea, R25, temps=TEMPS):
    Rm = []
    for T in temps:
        Rt = rct_true(T, Ea, R25); I0 = RG*T/(F_CONST*Rt)
        Rm.append(bv_ratio(I, I0, T)*Rt)
    reg = linregress(1/temps, np.log(1/np.array(Rm)))
    return -reg.slope*RG/1000, Rm[0]/rct_true(temps[0], Ea, R25)


def amplitude_limit(T, Ea, R25, tol=0.01):
    """Largest I_ac (A) for which 1 - Rmeas/Rtrue <= tol (bisection)."""
    Rt = rct_true(T, Ea, R25); I0 = RG*T/(F_CONST*Rt)
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if 1-bv_ratio(mid, I0, T) > tol: hi = mid
        else: lo = mid
    return lo


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for Ea_true in [50e3, 35e3]:
        print(f"true Ea = {Ea_true/1000:.0f} kJ/mol, Rct(25C) = 10 mOhm")
        for I in [0.5, 0.2, 0.05, 0.03]:
            Ea_m, ratio = measured_ea(I, Ea_true, 0.010)
            rows.append(dict(Ea_true_kJmol=Ea_true/1000, Rct25_mOhm=10.0, I_ac_A=I,
                             Ea_measured_kJmol=round(Ea_m, 3),
                             Ea_bias_kJmol=round(Ea_m-Ea_true/1000, 3),
                             Rct_ratio_minus10C=round(ratio, 4)))
            print(f"  I={I*1e3:4.0f} mA: measured Ea={Ea_m:6.2f}  bias {Ea_m-Ea_true/1000:+.2f}  "
                  f"Rct(-10C) meas/true={ratio:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'bv_amplitude_bias.csv'), index=False)

    lim = []
    for R25 in [0.008, 0.014, 0.025]:
        I = amplitude_limit(263.15, 50e3, R25)
        lim.append(dict(Rct25_mOhm=R25*1e3, Ea_true_kJmol=50.0, T_C=-10.0,
                        I_ac_max_mA=round(I*1e3, 1)))
        print(f"  1 % limit at -10 C, Rct(25C) = {R25*1e3:4.0f} mOhm: {I*1e3:6.1f} mA")
    pd.DataFrame(lim).to_csv(os.path.join(OUT, 'bv_amplitude_limits.csv'), index=False)

    e1, _ = measured_ea(0.5, 50e3, 0.008); e2, _ = measured_ea(0.5, 45e3, 0.014)
    pd.DataFrame([dict(cell='A', Ea_true_kJmol=50.0, Rct25_mOhm=8.0, Ea_measured_0p5A=round(e1, 3)),
                  dict(cell='B', Ea_true_kJmol=45.0, Rct25_mOhm=14.0, Ea_measured_0p5A=round(e2, 3))]
                 ).to_csv(os.path.join(OUT, 'bv_ranking_example.csv'), index=False)
    print(f"  ranking example at 0.5 A: true gap 5.0, apparent gap {e1-e2:.2f} kJ/mol")
    print("wrote out/bv_amplitude_bias.csv, out/bv_amplitude_limits.csv, out/bv_ranking_example.csv")
