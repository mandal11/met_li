"""Grid subsampling of the eleven-setpoint reference grid (Table 4, Fig. 6).

Every contiguous window of k = 3 ... 11 setpoints of the mid state (state 5, OCV ~3.67 V,
nominal SoC 50 %) is regressed separately. The 95 % CI uses the exact Student-t quantile
t(0.975, k-2). Writes out/unisa_subsampling.csv for R_ct (the manuscript's table) and
out/unisa_subsampling_sum.csv for the swap-invariant sum R_SEI + R_ct (robustness check).
Also writes out/unisa_subsampling_constant_50K.xlsx, enumerating subsets with fixed
50 K endpoint span for k = 11, 9, 5, and 3, for both resistance measures, and
out/unisa_subsampling_constant_50K_temperatures.csv mapping result rows to their
interior temperature selections. Individual regression plots are saved under
out/arrhenius_subsampling_plots/. Use --plots-only to regenerate plots without
overwriting the tabular outputs.
"""
import os
import sys
from itertools import combinations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, t as student_t
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
df = pd.read_csv(os.path.join(OUT, 'unisa_fits.csv'))
mid = 5   # mid-state index (OCV ~3.67 V)
gg = df[df.ridx == mid].sort_values('T')
soc = float(gg['soc'].median()) if 'soc' in gg.columns and gg['soc'].notna().any() else float('nan')
TK = gg['T'].values + 273.15
PLOTS_ONLY = '--plots-only' in sys.argv


def save_arrhenius_plot(R, indices, path, title, Ea, CI, R2):
    x = 1000 / TK[list(indices)]
    y = np.log(1 / R[list(indices)])
    reg = linregress(1 / TK[list(indices)], y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = reg.intercept + reg.slope * x_fit / 1000

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, color='tab:blue', zorder=3)
    ax.plot(x_fit, y_fit, color='tab:orange', label='Linear regression')
    ax.set_xlabel('1000/T (K$^{-1}$)')
    ax.set_ylabel('ln(1/R)')
    ax.set_title(title)
    ax.text(0.04, 0.96, f'$E_a$ = {Ea:.2f} kJ/mol\n95% CI = ±{CI:.2f} kJ/mol\n$R^2$ = {R2:.4f}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def subsample(R):
    rows = []; n = len(TK)
    for k in range(3, n+1):
        for i in range(0, n-k+1):
            s = slice(i, i+k)
            reg = linregress(1/TK[s], np.log(1/R[s]))
            t_critical = student_t.ppf(0.975, df=k-2)  # two-sided 95% CI for the slope
            rows.append(dict(SOC=soc, k=k, Tmin=TK[s][0]-273.15, span=TK[s][-1]-TK[s][0],
                             Ea=-reg.slope*8.314/1000,
                             CI=reg.stderr*8.314/1000*t_critical,
                             R2=reg.rvalue**2))
    return pd.DataFrame(rows)


def constant_span_selections(k_values=(11, 9, 5, 3)):
    n = len(TK)
    if n < 3 or not np.isclose(TK[-1] - TK[0], 50.0):
        raise ValueError("Constant-span subsampling requires endpoints exactly 50 K apart.")
    selections = []
    for k in k_values:
        if k < 3 or k > n:
            raise ValueError(f"k={k} is outside the available range 3..{n}.")
        for subset_number, interior in enumerate(combinations(range(1, n-1), k-2), start=1):
            indices = (0, *interior, n-1)
            selections.append((k, subset_number, indices))
    return selections


def constant_span_subsample(R, selections, plot_dir, label):
    rows = []
    os.makedirs(plot_dir, exist_ok=True)
    for subset_number, (k, _, indices) in enumerate(selections, start=1):
        reg = linregress(1/TK[list(indices)], np.log(1/R[list(indices)]))
        t_critical = student_t.ppf(0.975, df=k-2)
        result = dict(SOC=soc, k=k, Tmin=TK[0]-273.15, span=TK[-1]-TK[0],
                      Ea=-reg.slope*8.314/1000,
                      CI=reg.stderr*8.314/1000*t_critical,
                      R2=reg.rvalue**2)
        rows.append(result)
        save_arrhenius_plot(
            R, indices,
            os.path.join(plot_dir, f'k{k:02d}_subset_{subset_number:03d}.png'),
            f'{label}: fixed 50 K span, k={k}, subset={subset_number}',
            result['Ea'], result['CI'], result['R2'])
    return pd.DataFrame(rows)


plot_root = os.path.join(OUT, 'arrhenius_subsampling_plots')
for R, name, label, plot_label in [
        (gg['Rct'].values, 'unisa_subsampling.csv', 'R_ct', 'R_ct'),
        ((gg['Rsei'] + gg['Rct']).values, 'unisa_subsampling_sum.csv', 'R_SEI + R_ct', 'R_SEI_plus_R_ct')]:
    sub = subsample(R)
    if not PLOTS_ONLY:
        sub.to_csv(os.path.join(OUT, name), index=False)
    plot_dir = os.path.join(plot_root, 'contiguous', plot_label)
    os.makedirs(plot_dir, exist_ok=True)
    plot_row = 0
    for k in range(3, len(TK)+1):
        for i in range(len(TK)-k+1):
            indices = tuple(range(i, i+k))
            result = sub.iloc[plot_row]
            start_temp = TK[i] - 273.15
            start_label = f'{start_temp:g}C'.replace('-', 'm').replace('.', 'p')
            save_arrhenius_plot(
                R, indices,
                os.path.join(plot_dir, f'k{k:02d}_start_{start_label}.png'),
                f'{label}: contiguous window, k={k}, start={start_temp:g} degC',
                result.Ea, result.CI, result.R2)
            plot_row += 1
    print(f"{label}:")
    for k in sorted(sub.k.unique()):
        s = sub[sub.k == k]
        print(f"  k={k:2d}: Ea in [{s.Ea.min():6.1f},{s.Ea.max():6.1f}]  median CI ±{s.CI.median():5.1f}")

constant_span_path = os.path.join(OUT, 'unisa_subsampling_constant_50K.xlsx')
selections = constant_span_selections()
temperature_rows = []
for result_row, (k, subset_number, indices) in enumerate(selections, start=1):
    interior_temperatures = [TK[i]-273.15 for i in indices[1:-1]]
    temperature_rows.append(dict(
        result_row=result_row,
        k=k,
        subset_number=subset_number,
        interior_temperatures_C='; '.join(f'{temperature:g}' for temperature in interior_temperatures)
    ))
if not PLOTS_ONLY:
    pd.DataFrame(temperature_rows).to_csv(
        os.path.join(OUT, 'unisa_subsampling_constant_50K_temperatures.csv'), index=False)

if PLOTS_ONLY:
    for R, label, plot_label in [
            (gg['Rct'].values, 'R_ct', 'R_ct'),
            ((gg['Rsei'] + gg['Rct']).values, 'R_SEI + R_ct', 'R_SEI_plus_R_ct')]:
        plot_dir = os.path.join(plot_root, 'constant_50K', plot_label)
        sub = constant_span_subsample(R, selections, plot_dir, label)
        print(f"{label}, constant 50 K span:")
        for k in sorted(sub.k.unique(), reverse=True):
            s = sub[sub.k == k]
            print(f"  k={k:2d}: Ea in [{s.Ea.min():6.1f},{s.Ea.max():6.1f}]  median CI ±{s.CI.median():5.1f}")
    print(f"Plots only; tabular outputs were not written. Plots saved under: {plot_root}")
else:
    with pd.ExcelWriter(constant_span_path) as writer:
        for R, sheet, label, plot_label in [
                (gg['Rct'].values, 'R_ct', 'R_ct', 'R_ct'),
                ((gg['Rsei'] + gg['Rct']).values, 'R_SEI + R_ct', 'R_SEI + R_ct', 'R_SEI_plus_R_ct')]:
            plot_dir = os.path.join(plot_root, 'constant_50K', plot_label)
            sub = constant_span_subsample(R, selections, plot_dir, label)
            sub.to_excel(writer, sheet_name=sheet, index=False)
            print(f"{label}, constant 50 K span:")
            for k in sorted(sub.k.unique(), reverse=True):
                s = sub[sub.k == k]
                print(f"  k={k:2d}: Ea in [{s.Ea.min():6.1f},{s.Ea.max():6.1f}]  median CI ±{s.CI.median():5.1f}")
