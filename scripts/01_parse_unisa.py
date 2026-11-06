"""Parse UNISA-CAU EIS workbooks (sheet ACIM_chan_*) -> out/unisa_parsed.pkl."""
import re, pickle, os
import pandas as pd
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())
os.makedirs(OUT, exist_ok=True)

files = find_unisa()
if not files:
    raise SystemExit(
        "\nNo UNISA workbooks found.\n"
        f"  UNISA_RAW  = {UNISA_RAW}\n  resolved to = {UNISA_GLOB}\n  CHANNEL     = {CHANNEL}\n"
        f"  repo root   = {REPO_ROOT}\n"
        "Fix: open scripts/00_paths.py and set UNISA_RAW to your extracted dataset, e.g.\n"
        '  UNISA_RAW = "C:/Users/you/Downloads/250121-Dataset UNISA-CAU/**/EIS_allsoc_*.xlsx"\n'
        "(keep the ** - it makes the search recursive). Auto-discovery also scans the repo\n"
        "root, its parent/grandparent and ~/Downloads, ~/Desktop, ~/Documents.")

def parse_file(path):
    m = re.search(r'allsoc_(n?)(\d+)deg', os.path.basename(path))
    if not m:
        return None, None
    T = int(m.group(2)) * (-1 if m.group(1) == 'n' else 1)
    sheets = [s for s in pd.ExcelFile(path).sheet_names if s.startswith('ACIM')]
    if not sheets:
        return None, None
    df = pd.read_excel(path, sheet_name=sheets[0])
    df.columns = [str(c).strip() for c in df.columns]
    need = {'EIS_Test_ID', 'Frequency', 'Zreal', 'Zimg', 'OCV'}
    if not need.issubset(df.columns):
        print(f"  skipped (columns {sorted(need - set(df.columns))} missing): {os.path.basename(path)}")
        return None, None

    soc_col = None
    for c in df.columns:
        if c.lower() == 'soc':
            soc_col = c; break
    if soc_col is None:
        for c in df.columns:
            if 'soc' in c.lower():
                soc_col = c; break

    out = []
    for tid, g in df.groupby('EIS_Test_ID'):
        g = g.sort_values('Frequency')
        soc = float('nan') if soc_col is None else g[soc_col].astype(float).mean()
        out.append((int(tid),
                    g['Frequency'].astype(float).values,
                    g['Zreal'].astype(float).values + 1j*g['Zimg'].astype(float).values,
                    g['OCV'].astype(float).mean(),
                    g['AC_Amp_RMS'].astype(float).mean() if 'AC_Amp_RMS' in g else float('nan'),
                    soc))
    out.sort(key=lambda r: r[0])
    # The corpus stores the nominal state of charge as a fraction (1.0 ... 0.0); report it
    # in percent everywhere downstream (pickle, CSV, plots), so 0.5 becomes 50 %.
    socs = [r[5] for r in out if pd.notna(r[5])]
    if socs and max(socs) <= 1.0 + 1e-9:
        out = [r[:5] + (r[5]*100.0 if pd.notna(r[5]) else r[5],) for r in out]
    return T, out

alldata = {}
for p in files:
    T, specs = parse_file(p)
    if T is None:
        continue
    if T in alldata:
        print(f"  warning: duplicate file for T={T} C, keeping the first")
        continue
    alldata[T] = specs
    amp = specs[0][4]
    print(f"T={T:>4d} C: {len(specs):2d} spectra   AC_Amp_RMS={amp:.4f} A   [{os.path.basename(p)}]")

if not alldata:
    raise SystemExit("Files were found but none could be parsed - check that these are the "
                     "EIS_allsoc_*deg_Channel_*.xlsx workbooks of the UNISA-CAU release.")

rows = []
for T, specs in sorted(alldata.items()):
    for spec_idx, (tid, freq, z, ocv, amp_rms, soc) in enumerate(specs):
        for f, val in zip(freq, z):
            rows.append({
                'temperature_C': T,
                'spectrum_id': tid,
                'spectrum_index': spec_idx,
                'frequency_Hz': float(f),
                'Zreal_ohm': float(val.real),
                'Zimg_ohm': float(val.imag),
                'Zmag_ohm': float(abs(val)),
                'OCV_V': float(ocv),
                'SOC': float(soc) if pd.notna(soc) else float('nan'),
                'AC_Amp_RMS_A': float(amp_rms) if pd.notna(amp_rms) else float('nan'),
            })

flat = pd.DataFrame(rows)

pickle.dump(alldata, open(os.path.join(OUT, 'unisa_parsed.pkl'), 'wb'))
print(f"parsed {len(alldata)} temperatures -> {os.path.join(OUT, 'unisa_parsed.pkl')}")

csv_path = safe_write(flat, 'unisa_parsed.csv', complete=True,
                      needs='UNISA workbooks parsed into a flat frequency-resolved CSV table')
print(f"parsed {len(alldata)} temperatures -> {csv_path}")

xlsx_path = os.path.join(OUT, 'unisa_parsed.xlsx')
try:
    if os.path.exists(xlsx_path):
        os.remove(xlsx_path)
    flat.to_excel(xlsx_path, index=False)
    print(f"saved Excel workbook -> {xlsx_path}")
except PermissionError as e:
    print(f"[warn] Excel export skipped: {e} (close any open Excel workbook for this file first)")
except Exception as e:
    print(f"[warn] Excel export skipped: {e}")
