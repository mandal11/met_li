import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import fit_ecm, arrhenius, zmodel, branch_tau
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
pkl = os.path.join(OUT, 'unisa_parsed.pkl')
if not os.path.exists(pkl):
    raise SystemExit("out/unisa_parsed.pkl not found - run scripts/01_parse_unisa.py first.")
alldata = pickle.load(open(pkl, 'rb'))
if not alldata:
    raise SystemExit("unisa_parsed.pkl is empty - re-run 01_parse_unisa.py; it now fails loudly "
                     "if UNISA_GLOB matches no files (edit scripts/00_paths.py).")
temps_avail = sorted(alldata)
print("temperatures available:", temps_avail)

def groups(specs): return [specs[i:i+3] for i in range(0, len(specs), 3)]

# OCV reference set: 25 C if present, else the median available temperature
T_ref = 25 if 25 in alldata else temps_avail[len(temps_avail)//2]
print("reference temperature for state matching:", T_ref, "C")
ref = [np.median([s[3] for s in g]) for g in groups(alldata[T_ref])]

# Plot directories under OUT
ecm_plot_dir = os.path.join(OUT, 'ecm_fits')
arr_plot_dir = os.path.join(OUT, 'arrhenius_fits')
os.makedirs(ecm_plot_dir, exist_ok=True)
os.makedirs(arr_plot_dir, exist_ok=True)

rows = []
for T in sorted(alldata, reverse=True):          # warm -> cold
    prev = None
    for gi, g in enumerate(groups(alldata[T])):
        fr = g[0][1]
        zmed = np.median(np.array([s[2] for s in g]), axis=0)   # triplicate median spectrum
        ocv = np.median([s[3] for s in g])
        socs = [s[5] for s in g if len(s) > 5 and not pd.isna(s[5])]
        soc = float(np.median(socs)) if socs else float('nan')   # nominal SoC label, %
        ridx = int(np.argmin([abs(ocv-r) for r in ref]))        # state matched by OCV
        p, rms = fit_ecm(fr, zmed, p0=prev, with_L=True); prev = p.copy()
        t1, t2 = branch_tau(p)
        rows.append(dict(T=T, ridx=ridx, ocv=ocv, soc=soc, Rohm=p[0], Rsei=p[1], Rct=p[4], rms=rms,
                         gi=gi, A1=p[2], P1=p[3], A2=p[5], P2=p[6], Rw0=p[7], Aw0=p[8], L=p[9],
                         tau1=t1, tau2=t2, sep=t2/t1))

        # Save a Nyquist plot for each ECM fit
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(zmed.real*1e3, -zmed.imag*1e3, 'o', markersize=3, label='data')
        zfit = zmodel(p, 2*np.pi*fr, with_L=True)
        ax.plot(zfit.real*1e3, -zfit.imag*1e3, '-', lw=1.8, label=f'ECM fit (rms={rms:.2f}%)')
        ax.set_xlabel('Z\' (mΩ)')
        ax.set_ylabel('-Z\'\' (mΩ)')
        soc_label = 'nan' if np.isnan(soc) else f'{soc:.1f}%'
        ax.set_title(f'ECM fit — T={T} °C, state {ridx}, OCV={ocv:.3f} V, SOC={soc_label}')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(ecm_plot_dir, f'ecm_fit_T{T}_state{ridx}_SOC{soc_label}.png'), dpi=200)
        plt.close(fig)

df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'unisa_fits.csv'), index=False)
print("fits:", df.shape, "median rms%:", round(df.rms.median(), 2))
print(f"ECM fit plots saved in: {ecm_plot_dir}")

# one point per (state, temperature): median over any duplicate OCV-matched groups
# dfm = df.groupby(['ridx', 'T'], as_index=False)[['Rohm', 'Rsei', 'Rct', 'soc']].median()
# # Keep the full state-matched input table used for Arrhenius regression, including
# # temperature, OCV, and SOC for each ECM case before the fit is performed.
# arr_input = []
# for _, row in dfm.iterrows():
#     ridx = int(row['ridx'])
#     T = float(row['T'])
#     # Use the actual measured OCV of the state-matched group, not the reference OCV
#     # used only for assignment. The reference OCV is the anchor for matching, but the
#     # real operating point for the Arrhenius input should be the actual OCV at that T.
#     actual_ocv = df[(df['T'] == T) & (df['ridx'] == ridx)]['ocv'].median()
#     arr_input.append({
#         'ridx': ridx,
#         'T_C': T,
#         'OCV_V': float(actual_ocv) if pd.notna(actual_ocv) else float('nan'),
#         'SOC_pct': float(row['soc']) if pd.notna(row['soc']) else float('nan'),
#         'Rohm_ohm': float(row['Rohm']),
#         'Rsei_ohm': float(row['Rsei']),
#         'Rct_ohm': float(row['Rct'])
#     })
# arr_input = pd.DataFrame(arr_input).sort_values(['ridx', 'T_C']).reset_index(drop=True)
# arr_input_path = os.path.join(OUT, 'unisa_arrhenius_input.csv')
# arr_input.to_csv(arr_input_path, index=False)
# try:
#     arr_input.to_excel(os.path.join(OUT, 'unisa_arrhenius_input.xlsx'), index=False)
#     print(f"saved Arrhenius input table -> {os.path.join(OUT, 'unisa_arrhenius_input.xlsx')}")
# except Exception as e:
#     print(f"[warn] Arrhenius Excel export skipped: {e}")
# print(f"saved Arrhenius input CSV -> {arr_input_path}")

# ndup = len(df) - len(dfm)
# if ndup:
#     print(f"note: {ndup} duplicate (state,temperature) fits averaged before regression")
# one point per (state, temperature):
# retain the measurement with the smallest OCV difference from the
# corresponding 25 C reference OCV.
# If the OCV distances are identical, retain the fit with the lower RMS.
# If both are identical, retain the first measurement in the original order.

df = df.reset_index(drop=False).rename(columns={'index': 'original_order'})

# Calculate the absolute OCV distance from the assigned reference state
df['ocv_distance'] = df.apply(
    lambda row: abs(row['ocv'] - ref[int(row['ridx'])]),
    axis=1
)

# Sort according to the deterministic selection hierarchy:
# 1. state and temperature
# 2. smallest OCV distance
# 3. lowest ECM fitting RMS
# 4. original measurement order
df_sorted = df.sort_values(
    ['ridx', 'T', 'ocv_distance', 'rms', 'original_order'],
    ascending=[True, True, True, True, True]
)

# Retain only one measurement for each (state, temperature)
dfm = (
    df_sorted
    .drop_duplicates(subset=['ridx', 'T'], keep='first')
    .sort_values(['ridx', 'T'])
    .reset_index(drop=True)
)

# Keep the full state-matched input table used for Arrhenius regression
arr_input = []
for _, row in dfm.iterrows():
    ridx = int(row['ridx'])
    T = float(row['T'])

    # OCV of the actual measurement retained by the selection rule
    actual_ocv = float(row['ocv'])

    arr_input.append({
        'ridx': ridx,
        'T_C': T,
        'gi': int(row['gi']),
        'OCV_V': actual_ocv,
        'SOC_pct': float(row['soc']) if pd.notna(row['soc']) else float('nan'),
        'Rohm_ohm': float(row['Rohm']),
        'Rsei_ohm': float(row['Rsei']),
        'Rct_ohm': float(row['Rct']),
        'Rsum_ohm': float(row['Rsei'] + row['Rct'])
    })

arr_input = (
    pd.DataFrame(arr_input)
    .sort_values(['ridx', 'T_C'])
    .reset_index(drop=True)
)

arr_input_path = os.path.join(OUT, 'unisa_arrhenius_input.csv')
arr_input.to_csv(arr_input_path, index=False)

try:
    arr_input.to_excel(
        os.path.join(OUT, 'unisa_arrhenius_input.xlsx'),
        index=False
    )
    print(
        f"saved Arrhenius input table -> "
        f"{os.path.join(OUT, 'unisa_arrhenius_input.xlsx')}"
    )
except Exception as e:
    print(f"[warn] Arrhenius Excel export skipped: {e}")

print(f"saved Arrhenius input CSV -> {arr_input_path}")

ndup = len(df) - len(dfm)
if ndup:
    print(
        f"note: {ndup} duplicate (state,temperature) fits removed; "
        "smallest OCV-distance measurement retained "
        "(RMS used as tie-breaker)"
    )
arr = []
for ridx, gg in dfm.groupby('ridx'):
    gg = gg.sort_values('T')
    if len(gg) < 8: continue
    TK = gg['T'].values + 273.15
    soc_mean = float(np.nanmedian(gg['soc'].values)) if 'soc' in gg.columns and gg['soc'].notna().any() else float('nan')
    row = {'ridx': ridx, 'ocv': round(ref[ridx], 3), 'soc_mean': soc_mean, 'nT': len(gg)}
    gg = gg.assign(Rsum=gg['Rsei'] + gg['Rct'])   # swap-invariant interfacial sum
    for proc in ['Rohm', 'Rsei', 'Rct', 'Rsum']:
        x = 1.0 / TK
        y = np.log(1.0 / gg[proc].values)
        slope, intercept = np.polyfit(x, y, 1)
        Ea, ci, r2 = arrhenius(TK, gg[proc].values)
        row[f'Slope_{proc}'] = float(slope)
        row[f'Ea_{proc}'], row[f'CI_{proc}'], row[f'R2_{proc}'] = Ea, ci, r2
    arr.append(row)

    # Save Arrhenius regression plot for each OCV-matched state
    fig, ax = plt.subplots(figsize=(6, 5))
    for proc in ['Rohm', 'Rsei', 'Rct']:
        x = 1.0 / TK
        y = np.log(1.0 / gg[proc].values)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = intercept + slope * x
        r_value = np.corrcoef(x, y)[0, 1]
        ax.plot(x, y, 'o', markersize=4, label=f'{proc} data')
        ax.plot(x, y_fit, '-', lw=1.8, label=f'{proc} fit (R²={r_value**2:.3f})')
    ax.set_xlabel('1/T (K$^{-1}$)')
    ax.set_ylabel('ln(1/R)')
    soc_label = 'nan' if np.isnan(soc_mean) else f'{soc_mean:.1f}%'
    ax.set_title(f'Arrhenius regression — state {ridx}, OCV={ref[ridx]:.3f} V, SOC={soc_label}')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(arr_plot_dir, f'arrhenius_state{ridx}_SOC{soc_label}.png'), dpi=200)
    plt.close(fig)

arr = pd.DataFrame(arr).sort_values('ocv', ascending=False)
arr.to_csv(os.path.join(OUT, 'unisa_arrhenius.csv'), index=False)
print(f"Arrhenius plots saved in: {arr_plot_dir}")
print(arr.round(2).to_string(index=False))
