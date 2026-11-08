# ---------------------------------------------------------------------------
# Paths configuration.  Edit the RAW paths below to point at your local copies
# of the datasets (they are not redistributed; see README).
#
# Notes
#  * Forward slashes work on Windows too: "C:/Users/you/Downloads/...".
#  * Relative paths are resolved against the REPOSITORY ROOT (the folder that
#    contains scripts/ and eislib/), never against your current shell
#    directory, so the scripts behave the same wherever you launch them from.
#  * If UNISA_RAW matches nothing and AUTO_DISCOVER is True, the scripts search
#    a few likely folders (repo root, its parent and grandparent, ~/Downloads,
#    ~/Desktop, ~/Documents) for files named EIS_allsoc_*deg_Channel_*.xlsx.
#  * CHANNEL selects ONE physical cell of the UNISA-CAU corpus.  The study used
#    Channel_2; files of other channels are ignored so that temperatures are
#    never mixed across cells.
# ---------------------------------------------------------------------------
import os as _os, glob as _glob

UNISA_RAW     = "data/unisa_S3/**/EIS_allsoc_*.xlsx"
CHANNEL       = "Channel_2"
AUTO_DISCOVER = True

INHOUSE_HE2_RAW   = "data/inhouse/LGHE2_S1.xlsx"
INHOUSE_VTC5A_RAW = "data/inhouse/all_temp_sony_S1.xlsx"
# INHOUSE_AMS_RAW   = "data/inhouse/ams"             # folder with (T)_AMS*.xlsx
SIXT_FILE_A_RAW   = "data/inhouse/Samsung_S2a.xlsx"    # 6-temperature file A
SIXT_FILE_B_RAW   = "data/inhouse/Panasonic_S2b.xlsx"  # 6-temperature file B
SAMSUNG26J_RAW    = "data/corpora/impedance_S5.csv"
LGM50_FITS_RAW    = "data/LGM50_SoH_S4/all_ecm_SoH_temp_S4.xlsx"
OUT_RAW           = "out"

# --------------------------- resolution machinery --------------------------
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

def resolve(p):
    """Absolute path; relative inputs are taken relative to the repo root."""
    p = _os.path.expanduser(p)
    return p if _os.path.isabs(p) else _os.path.normpath(_os.path.join(REPO_ROOT, p))

def _exists(p):
    return p if _os.path.exists(p) else None

INHOUSE_HE2   = _exists(resolve(INHOUSE_HE2_RAW))
INHOUSE_VTC5A = _exists(resolve(INHOUSE_VTC5A_RAW))
# INHOUSE_AMS   = _exists(resolve(INHOUSE_AMS_RAW))
SIXT_FILE_A   = _exists(resolve(SIXT_FILE_A_RAW))
SIXT_FILE_B   = _exists(resolve(SIXT_FILE_B_RAW))
SAMSUNG26J    = _exists(resolve(SAMSUNG26J_RAW))
LGM50_FITS    = _exists(resolve(LGM50_FITS_RAW))
OUT           = resolve(OUT_RAW)
UNISA_GLOB    = resolve(UNISA_RAW)

def find_unisa():
    """Return the sorted list of UNISA workbooks for the configured CHANNEL."""
    hits = [f for f in _glob.glob(UNISA_GLOB, recursive=True)
            if CHANNEL in _os.path.basename(f)]
    if hits or not AUTO_DISCOVER:
        return sorted(hits)
    roots, seen = [], set()
    for r in [REPO_ROOT,
              _os.path.dirname(REPO_ROOT),
              _os.path.dirname(_os.path.dirname(REPO_ROOT)),
              _os.path.expanduser("~/Downloads"),
              _os.path.expanduser("~/Desktop"),
              _os.path.expanduser("~/Documents")]:
        if r and _os.path.isdir(r) and r not in seen:
            seen.add(r); roots.append(r)
    for r in roots:
        # NOTE: the release is inconsistent about the underscore before "Channel"
        # (e.g. EIS_allsoc_40degChannel_2_Wb_1.xlsx), so match loosely and filter.
        found = [f for f in _glob.glob(_os.path.join(r, "**", "EIS_allsoc_*.xlsx"),
                                       recursive=True)
                 if CHANNEL in _os.path.basename(f)]
        if found:
            print(f"[auto-discover] using UNISA files found under: {r}")
            return sorted(found)
    return []


# --------------------------- safe output writing ---------------------------
# Released tables must never be silently replaced by a partial or empty result
# computed on a machine that lacks some of the (non-redistributed) input data.
# Scripts therefore write through safe_write(), which:
#   * refuses to write at all if no rows were produced;
#   * writes to <name>.partial.csv, leaving the released file untouched, if only
#     some of the required inputs were found;
#   * writes normally only when the run is complete.

def safe_write(df, filename, complete, needs=""):
    """Write df to OUT/filename. Returns the path written, or None if nothing was."""
    path = _os.path.join(OUT, filename)
    _os.makedirs(OUT, exist_ok=True)
    if df is None or len(df) == 0:
        print(f"[abort] no rows produced -> {filename} NOT written.")
        if needs:
            print(f"        needs: {needs}")
        print("        The released CSV (if present) has been left untouched.")
        print("        See data/README.md for where to place the input files.")
        return None
    if not complete:
        alt = path[:-4] + '.partial.csv'
        df.to_csv(alt, index=False)
        print(f"[partial] only some inputs were found; wrote {alt}")
        if needs:
            print(f"          full table needs: {needs}")
        print(f"          the released {filename} has been left untouched.")
        return alt
    df.to_csv(path, index=False)
    print("wrote", path)
    return path
