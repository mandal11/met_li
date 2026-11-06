"""Regenerates manuscript Figs 1-10 into <repo>/figs.

Every figure is a function guarded by the data it needs: figures whose sources
are not present locally are skipped with a message instead of crashing the run.
Prerequisites: scripts 01-03 (for out/*.csv and out/unisa_parsed.pkl).
"""
import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import linregress
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eislib.eislib import linkk, fit_ecm, zmodel, bv_ratio, RG, F_CONST
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

FIGS = os.path.join(REPO_ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
TEMPS_IH = ['-10', '0', '15', '25', '40', '50']

def have(*paths):
    return all(p is not None and os.path.exists(p) for p in paths)

def load(f, s, scale=1.0, imag_flip=False):
    d = pd.read_excel(f, sheet_name=s).dropna()
    fr = d.iloc[:, 0].astype(float).values
    z = d.iloc[:, 1].astype(float).values + 1j*d.iloc[:, 2].astype(float).values
    i = np.argsort(fr); fr, z = fr[i], z[i]*scale
    if imag_flip: z = z.real - 1j*np.abs(z.imag)
    return fr, z

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, name), dpi=200); plt.close(fig)
    print("  wrote", os.path.join('figs', name))

# ------------------------------------------------------------------ loaders
PKL = os.path.join(OUT, 'unisa_parsed.pkl')
ALL = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
TS = sorted(ALL)
def csv(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None

def fig1():

    """
    Generate Fig. 1 as four separate publication-quality plots.

    (a) In-house LG HE2
    (b) In-house VTC5A
    (c) Reference corpus
    (d) Constrained ECM fits

    Each plot is saved as:
        PDF
        600 dpi PNG
    """

    if not have(INHOUSE_HE2, INHOUSE_VTC5A) or not ALL:
        print("[skip] Fig 1: needs in-house workbooks + parsed corpus")
        return

    # =========================================================
    # GLOBAL JOURNAL FONT SETTINGS
    # =========================================================
    plt.rcParams.update({
        "font.family": "Times New Roman",
        # "font.size": 10,

        "axes.titlesize": 24,
        "axes.labelsize": 24,

        "xtick.labelsize": 20,
        "ytick.labelsize": 20,

        "legend.fontsize": 16,

        # Make mathematical text closer to Times
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",

        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # =========================================================
    # COMMON AXIS FORMAT
    # =========================================================
    def format_axis(ax):

        ax.set_xlabel(
            r"$Z^\prime$ (m$\Omega$)",
            fontsize=20,
            labelpad=6
        )

        ax.set_ylabel(
            r"$-Z^{\prime\prime}$ (m$\Omega$)",
            fontsize=20,
            labelpad=6
        )

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=5,
            width=0.8,
            size=20,
            pad=4
        )

        # Grid
        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.5,
            alpha=0.20
        )

        # Clean borders
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

        # IMPORTANT:
        # Do NOT use:
        # ax.set_aspect("equal")
        #
        # The X and Y impedance ranges are very different,
        # so equal aspect makes the figure look stretched.

    # =========================================================
    # SAVE FUNCTION
    # =========================================================
    def save_individual(fig, name):

        fig.savefig(
            os.path.join(FIGS, f"{name}.pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )

        fig.savefig(
            os.path.join(FIGS, f"{name}.png"),
            format="png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )

        plt.close(fig)

        print(f"[saved] {name}.pdf")
        print(f"[saved] {name}.png")

    # =========================================================
    # (a) LG HE2
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(5.8, 4.4),
        constrained_layout=True
    )

    for t in TEMPS_IH:

        fr, z = load(INHOUSE_HE2, t)

        ax.plot(
            z.real * 1e3,
            -z.imag * 1e3,
            ".-",
            markersize=2.8,
            linewidth=1.0,
            label=f"{t} °C"
        )

    ax.set_title(
        r"In-house platform, LG HE2, $I_{\mathrm{ac}}$ = 0.5 A",
        fontsize=20,
        pad=8
    )

    # Explicit limits give a clean frame
    # ax.set_xlim(39, 120)
    # ax.set_ylim(-1, 16)

    # Legend below plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.8,
        handletextpad=0.5
    )

    format_axis(ax)

    save_individual(
        fig,
        "Fig1a_LG_HE2"
    )

    # =========================================================
    # (b) VTC5A
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(5.8, 4.4),
        constrained_layout=True
    )

    for t in TEMPS_IH:

        fr, z = load(INHOUSE_VTC5A, t)

        ax.plot(
            z.real * 1e3,
            -z.imag * 1e3,
            ".-",
            markersize=2.8,
            linewidth=1.0,
            label=f"{t} °C"
        )

    ax.set_title(
        r"In-house platform, VTC5A, $I_{\mathrm{ac}}$ = 0.5 A",
        fontsize=20,
        pad=8
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.8,
        handletextpad=0.5
    )

    format_axis(ax)

    save_individual(
        fig,
        "Fig1b_VTC5A"
    )

    # =========================================================
    # (c) REFERENCE CORPUS
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(5.8, 4.4),
        constrained_layout=True
    )

    for T in TS:

        s_ = ALL[T][len(ALL[T]) // 2]

        ax.plot(
            s_[2].real * 1e3,
            -s_[2].imag * 1e3,
            ".-",
            markersize=2.8,
            linewidth=1.0,
            label=f"{T} °C"
        )

    ax.set_title(
        r"Reference corpus, 0.106 A rms, mid-state",
        fontsize=20,
        pad=8
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.46),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.6,
        handletextpad=0.5
    )

    format_axis(ax)

    save_individual(
        fig,
        "Fig1c_reference_corpus"
    )

    # =========================================================
    # (d) CONSTRAINED ECM FITS
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(5.8, 4.4),
        constrained_layout=True
    )

    for T, mk in [
        (max(TS), "o"),
        (min(TS), "s")
    ]:

        s_ = ALL[T][len(ALL[T]) // 2]

        # ECM fitting
        p, rms = fit_ecm(
            s_[1],
            s_[2],
            with_L=True
        )

        # ECM impedance
        zf = zmodel(
            p,
            2 * np.pi * s_[1],
            True
        )

        # Experimental data
        ax.plot(
            s_[2].real * 1e3,
            -s_[2].imag * 1e3,
            marker=mk,
            markersize=4,
            linestyle="None",
            markeredgewidth=0.6,
            label=f"{T} °C data"
        )

        # ECM fit
        ax.plot(
            zf.real * 1e3,
            -zf.imag * 1e3,
            "-",
            linewidth=1.4,
            label=f"Fit ({rms:.2f}% RMS)"
        )

    ax.set_title(
        "Constrained ECM fits",
        fontsize=20,
        pad=8
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.8,
        handletextpad=0.5
    )

    format_axis(ax)

    save_individual(
        fig,
        "Fig1d_constrained_ECM_fits"
    )

    print("\nFig. 1 plots generated successfully.")
##########################################
# ------------------------------------------------------------------ Fig 2
def fig2v2():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.logspace(-2, 1.2, 80)
    ax[0].plot(x, [bv_ratio(xx*2*1.0, 1.0, 298) for xx in x])
    ax[0].set_xscale('log'); ax[0].set_xlabel('$I_{ac}/2I_0$', fontsize=16); ax[0].set_ylabel('$R_{ct}^{meas}/R_{ct}^{true}$', fontsize=16)
    ax[0].axhline(.99, color='gray', ls=':'); ax[0].set_title('(a) Fundamental-response attenuation', fontsize=16)
    ax[0].tick_params(axis='both', labelsize=16)
    Tlist = np.array([263.15, 273.15, 288.15, 298.15, 313.15, 323.15])
    for I, c in [(0.5, 'crimson'), (0.2, 'darkorange'), (0.05, 'seagreen'), (0.03, 'royalblue')]:
        Rm = []
        for T in Tlist:
            Rt = 0.010*np.exp(50000/RG*(1/T-1/298.15)); i0 = RG*T/(F_CONST*Rt)
            Rm.append(bv_ratio(I, i0, T)*Rt)
        reg = linregress(1/Tlist, np.log(1/np.array(Rm)))
        ax[1].bar(str(int(I*1000)), -reg.slope*RG/1000-50, color=c)
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_xlabel('$I_{ac}$ (mA peak)', fontsize=16)
    ax[1].set_ylabel('$E_a$ bias (kJ mol$^{-1}$)', fontsize=16)
    ax[1].tick_params(axis='both', labelsize=16)
    ax[1].set_title('(b) $E_a$ bias, true $E_a$=50, $R_{ct,25°C}$=10 mΩ', fontsize=16)
    Ts2 = np.linspace(253, 323, 29)
    for Rct25, lab in [(0.008, '8 mΩ'), (0.014, '14 mΩ'), (0.025, '25 mΩ')]:
        Imax = []
        for T in Ts2:
            Rt = Rct25*np.exp(50000/RG*(1/T-1/298.15)); i0 = RG*T/(F_CONST*Rt)
            lo, hi = 1e-3, 5
            for _ in range(36):
                mid = (lo+hi)/2
                if 1-bv_ratio(mid, i0, T) > 0.01: hi = mid
                else: lo = mid
            Imax.append(mid*1000)
        ax[2].semilogy(Ts2-273.15, Imax, label='$R_{ct}$(25°C)='+lab)
    ax[2].axhline(500, color='crimson', ls='--'); ax[2].text(-8, 560, '0.5 A campaign', color='crimson', fontsize=8)
    ax[2].axhline(150, color='darkorange', ls=':'); ax[2].text(18, 165, '0.106 A rms reference', color='darkorange', fontsize=8)
    ax[2].set_xlabel('Temperature (°C)', fontsize=16); ax[2].set_ylabel('max $I_{ac}$ for ≤1% error (mA)', fontsize=16)
    ax[2].tick_params(axis='both', labelsize=16)
    ax[2].set_title('(c) Amplitude design rule', fontsize=16); ax[2].legend(fontsize=10); ax[2].grid(alpha=.3)
    save(fig, 'Fig2v2_bv_amplitude.png')
    save(fig, 'Fig2v2_bv_amplitude.pdf')
######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress




# ============================================================
# FIGURE SAVER
# ============================================================

def save_fig(fig, name):

    fig.savefig(
        os.path.join(FIGS, name + ".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        os.path.join(FIGS, name + ".pdf"),
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# FIGURE 2(a)
# ============================================================

def fig2a():

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    x = np.logspace(-2, 1.2, 80)

    y = [
        bv_ratio(xx * 2 * 1.0, 1.0, 298)
        for xx in x
    ]

    ax.plot(
        x,
        y,
        linewidth=2.5
    )

    ax.axhline(
        0.99,
        linestyle=":"
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"$I_{\mathrm{ac}}/2I_0$",
        fontsize=22
    )

    ax.set_ylabel(
        r"$R_{\mathrm{ct}}^{\mathrm{meas}}/"
        r"R_{\mathrm{ct}}^{\mathrm{true}}$",
        fontsize=22
    )

    ax.set_title(
        "Fundamental-response attenuation",
        fontsize=20,
        # fontweight="bold",
        pad=12
    )

    ax.tick_params(
        axis="both",
        labelsize=20,
        direction="in",
        length=6,
        width=1.2
    )

    ax.grid(
        which="major",
        alpha=0.25
    )

    ax.grid(
        which="minor",
        alpha=0.12
    )

    fig.tight_layout()

    save_fig(
        fig,
        "Fig2a_bv_fundamental_response"
    )


# ============================================================
# FIGURE 2(b)
# ============================================================

def fig2b():

    # plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    Tlist = np.array([
        263.15,
        273.15,
        288.15,
        298.15,
        313.15,
        323.15
    ])

    # --------------------------------------------------------
    # KEEP YOUR ORIGINAL COLOURS
    # --------------------------------------------------------

    current_colors = [
        (0.5, "crimson"),
        (0.2, "darkorange"),
        (0.05, "seagreen"),
        (0.03, "royalblue")
    ]

    I_values = []
    Ea_bias_values = []
    colors = []

    for I, c in current_colors:

        Rm = []

        for T in Tlist:

            Rt = (
                0.010
                * np.exp(
                    50000 / RG
                    * (1 / T - 1 / 298.15)
                )
            )

            i0 = (
                RG * T
                / (F_CONST * Rt)
            )

            Rmeas = (
                bv_ratio(I, i0, T)
                * Rt
            )

            Rm.append(Rmeas)

        Rm = np.array(Rm)

        reg = linregress(
            1 / Tlist,
            np.log(1 / Rm)
        )

        Ea_bias = (
            -reg.slope * RG / 1000
            - 50
        )

        I_values.append(I * 1000)
        Ea_bias_values.append(Ea_bias)
        colors.append(c)

    # --------------------------------------------------------
    # Bar plot
    # --------------------------------------------------------

    ax.bar(
        [str(int(I)) for I in I_values],
        Ea_bias_values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.7
    )

    # Zero reference
    ax.axhline(
        0,
        color="k",
        linewidth=1.0
    )

    ax.set_xlabel(
        r"$I_{ac}$ (mA peak)",
        fontsize=22
    )

    ax.set_ylabel(
        r"$E_a$ bias (kJ mol$^{-1}$)",
        fontsize=22
    )

    ax.set_title(
        r"(b) $E_a$ bias, true $E_a$=50, "
        r"$R_{ct,25^\circ C}$=10 m$\Omega$",
        fontsize=20,
        pad=12
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=20,
        direction="in",
        length=6,
        width=1.2
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    # PNG
    fig.savefig(
        os.path.join(FIGS, "Fig2b_bv_Ea_bias.png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    # PDF
    fig.savefig(
        os.path.join(FIGS, "Fig2b_bv_Ea_bias.pdf"),
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

# ============================================================
# FIGURE 2(c)
# ============================================================

def fig2c():

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    Ts2 = np.linspace(
        253,
        323,
        29
    )

    for Rct25, lab in [
        (0.008, "8 mΩ"),
        (0.014, "14 mΩ"),
        (0.025, "25 mΩ")
    ]:

        Imax = []

        for T in Ts2:

            Rt = (
                Rct25
                * np.exp(
                    50000 / RG
                    * (1 / T - 1 / 298.15)
                )
            )

            i0 = (
                RG * T
                / (F_CONST * Rt)
            )

            lo, hi = 1e-3, 5

            for _ in range(36):

                mid = (lo + hi) / 2

                if (
                    1 - bv_ratio(
                        mid,
                        i0,
                        T
                    ) > 0.01
                ):
                    hi = mid
                else:
                    lo = mid

            Imax.append(mid * 1000)

        ax.semilogy(
            Ts2 - 273.15,
            Imax,
            linewidth=2.5,
            label=rf"$R_{{\mathrm{{ct}}}}$(25°C) = {lab}"
        )

    ax.axhline(
        500,
        linestyle="--",
        linewidth=1.8,
        color='darkorange'
    )

    ax.text(
        -8,
        560,
        "0.5 A campaign",
        color='darkorange',
        fontsize=16
    )

    ax.axhline(
        150,
        linestyle=":",
        linewidth=1.8,
        color='crimson'
    )

    ax.text(
        18,
        165,
        "0.106 A rms reference",
        color='crimson',
        fontsize=16
    )

    ax.set_xlabel(
        "Temperature (°C)",
        fontsize=22
    )

    ax.set_ylabel(
        r"Maximum $I_{\mathrm{ac}}$ for ≤1% error (mA)",
        fontsize=22
    )

    ax.set_title(
        "Amplitude design rule",
        fontsize=20,
        # fontweight="bold",
        pad=12
    )

    ax.legend(
        fontsize=16,
        frameon=True
    )

    ax.tick_params(
        axis="both",
        labelsize=16,
        direction="in",
        length=6,
        width=1.2
    )

    ax.grid(
        which="major",
        alpha=0.25
    )

    ax.grid(
        which="minor",
        alpha=0.12
    )

    fig.tight_layout()

    save_fig(
        fig,
        "Fig2c_bv_amplitude_design_rule"
    )


# ============================================================
# MAIN FIGURE 2 FUNCTION
# ============================================================

def fig2():

    print("Generating Fig. 2(a)...")
    fig2a()

    print("Generating Fig. 2(b)...")
    fig2b()

    print("Generating Fig. 2(c)...")
    fig2c()

    print("Fig. 2 complete.")
######################################
# ------------------------------------------------------------------ Fig 3
def fig3v2():
    """(a) amplitude axis, documented sources only; (b) residual vs temperature incl.
    undocumented sources; (c),(d) representative KK residual spectra."""
    pts = []
    for f, lab, c, m in [(INHOUSE_HE2, 'in-house HE2 (0.5 A)', 'crimson', 'o'),
                         (INHOUSE_VTC5A, 'in-house VTC5A (0.5 A)', 'firebrick', 'D')]:
        if not have(f): continue
        for t in TEMPS_IH:
            fr, z = load(f, t)
            pts.append((0.5*np.abs(z).max()*1e3, linkk(fr, z), c, m, lab if t == '-10' else None))
    for T in TS:
        s_ = ALL[T][len(ALL[T])//2]
        pts.append((0.15*np.abs(s_[2]).max()*1e3, linkk(s_[1], s_[2]), 'royalblue', '^',
                    'reference corpus (0.106 A rms)' if T == TS[0] else None))
    _need = [have(INHOUSE_HE2), have(INHOUSE_VTC5A), have(SIXT_FILE_A), have(SIXT_FILE_B)]
    if not pts:
        print("[skip] Fig 3: no spectra available"); return
    if not all(_need):
        print("[skip] Fig 3: needs both in-house workbooks AND both six-setpoint files; "
              "a corpus-only panel would omit the amplitude boundary that Fig. 3 exists to show. "
              "Released Fig3_kk_audit.png left untouched."); return
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for x, y, c, m, lab in pts:
        ax[0].scatter(x, max(y, .05), c=c, marker=m, s=30, label=lab, alpha=.85, edgecolors='none')
    ax[0].axvspan(27, 34, color='gray', alpha=.18)
    ax[0].axvline(10, color='gray', ls='--', lw=1); ax[0].axhline(2, color='gray', ls=':', lw=1)
    ax[0].set_ylim(0.08, 30); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    
    ax[0].text(10.6, 12, '10–15 mV\nconvention', fontsize=10, color='dimgray', va='center', ha='left')
    ax[0].text(30, 17, 'observed\nonset', fontsize=10, color='dimgray', va='center', ha='center')
    ax[0].set_xlabel('peak voltage response $I_{ac}\\,|Z|_{max}$ (mV)', fontsize=14)
    ax[0].set_ylabel('KK residual ρ (%)', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=13)
    ax[0].set_title('(a) Documented amplitudes', fontsize=15)
    ax[0].legend(fontsize=9, loc='lower right', framealpha=.95); ax[0].grid(alpha=.3)
    bars = [(INHOUSE_HE2, 'HE2 (0.5 A)', 'crimson', {}),
            (SIXT_FILE_A, 'S2a (amplitude n/d)', 'darkred', dict(scale=1e-3, imag_flip=True)),
            (SIXT_FILE_B, 'S2b (amplitude n/d)', 'seagreen', dict(scale=1e-3, imag_flip=True))]
    bars = [b for b in bars if have(b[0])]
    width = .8/max(len(bars), 1)
    for i2, (f, lab, c, kw) in enumerate(bars):
        vals = [linkk(*load(f, t, **kw)) for t in TEMPS_IH]
        ax[1].bar(np.arange(len(TEMPS_IH))+i2*width, vals, width, color=c, label=lab)
    ax[1].axhline(2, color='gray', ls=':'); ax[1].set_xticks(np.arange(len(TEMPS_IH))+width)
    ax[1].set_xticklabels(TEMPS_IH, fontsize=13)
    ax[1].tick_params(axis='both', labelsize=13)
    ax[1].set_xlabel('temperature (°C)', fontsize=14)
    ax[1].set_ylabel('KK residual ρ (%)', fontsize=14)
    ax[1].set_title('(b) Residual by temperature', fontsize=15)
    if bars: ax[1].legend(fontsize=10)
    if have(INHOUSE_VTC5A):
        for a, t, ttl in [(ax[2], '25', '(c) VTC5A, 25 °C (pass)'), (ax[3], '-10', '(d) VTC5A, −10 °C (fail)')]:
            fr, z = load(INHOUSE_VTC5A, t)
            rr, ri = linkk(fr, z, ret_spec=True)
            a.semilogx(fr, rr, '.-', label='Re'); a.semilogx(fr, ri, '.-', label='Im')
            a.set_title(ttl, fontsize=15); a.axhline(0, color='k', lw=.6); a.axhspan(-2, 2, color='gray', alpha=.15)
            a.tick_params(axis='both', labelsize=13)
            a.set_xlabel('f (Hz)', fontsize=14); a.set_ylabel('KK residual (% of |Z|)', fontsize=14); a.legend(fontsize=10)
        ax[2].set_ylim(-3, 3)
    save(fig, 'Fig3v2_kk_audit.png')
    save(fig, 'Fig3v2_kk_audit.pdf')
###############################
def fig3():
    """Create Fig. 3 as four separate plots:
    (a) amplitude axis,
    (b) residual vs temperature,
    (c) representative KK residual spectrum at 25 °C,
    (d) representative KK residual spectrum at −10 °C.
    """

    # ============================================================
    # (a) Linearity audit — documented amplitudes
    # ============================================================
    pts = []

    for f, lab, c, m in [
        (INHOUSE_HE2, 'in-house HE2 (0.5 A)', 'crimson', 'o'),
        (INHOUSE_VTC5A, 'in-house VTC5A (0.5 A)', 'firebrick', 'D')
    ]:
        if not have(f):
            continue

        for t in TEMPS_IH:
            fr, z = load(f, t)
            pts.append((
                0.5 * np.abs(z).max() * 1e3,
                linkk(fr, z),
                c,
                m,
                lab if t == '-10' else None
            ))

    for T in TS:
        s_ = ALL[T][len(ALL[T]) // 2]
        pts.append((
            0.15 * np.abs(s_[2]).max() * 1e3,
            linkk(s_[1], s_[2]),
            'royalblue',
            '^',
            'reference corpus (0.106 A rms)' if T == TS[0] else None
        ))

    _need = [
        have(INHOUSE_HE2),
        have(INHOUSE_VTC5A),
        have(SIXT_FILE_A),
        have(SIXT_FILE_B)
    ]

    if not pts:
        print("[skip] Fig 3: no spectra available")
        return

    if not all(_need):
        print(
            "[skip] Fig 3: needs both in-house workbooks AND both "
            "six-setpoint files; a corpus-only panel would omit the "
            "amplitude boundary that Fig. 3 exists to show. "
            "Released Fig3_kk_audit.png left untouched."
        )
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    for x, y, c, m, lab in pts:
        ax.scatter(
            x, max(y, .05),
            c=c,
            marker=m,
            s=30,
            label=lab,
            alpha=.85,
            edgecolors='none'
        )

    ax.axvspan(27, 34, color='gray', alpha=.18)
    ax.axvline(10, color='gray', ls='--', lw=1)
    ax.axhline(2, color='gray', ls=':', lw=1)

    ax.set_ylim(0.08, 30)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.annotate(
        '10-15 mV\nconvention',
        xy=(10, 15),
        xytext=(4.5, 16),
        fontsize=7,
        color='dimgray',
        va='center',
        ha='center'
    )

    ax.annotate(
        'observed onset\n27-34 mV',
        xy=(30, 0.35),
        xytext=(30, 0.15),
        fontsize=7,
        color='dimgray',
        ha='center'
    )

    ax.set_xlabel('peak voltage response $I_{ac}\\,|Z|_{max}$ (mV)', fontsize=17)
    ax.set_ylabel('KK-consistency residual ρ (% of |Z|)', fontsize=17)
    ax.set_title('Linearity audit, documented amplitudes', fontsize=17)
    # ax.ticks(axis=both, fontsize=17)
    ax.legend(fontsize=10)
    ax.grid(alpha=.3)

    save(fig, 'Fig3a_kk_linearity_audit.png')
    # (a)
    fig.savefig(os.path.join(FIGS, 'Fig3a_kk_linearity_audit.pdf'), bbox_inches='tight')
    plt.close(fig)
    plt.close(fig)


    # ============================================================
    # (b) Residuals by temperature
    # ============================================================
    bars = [
        (INHOUSE_HE2, 'in-house HE2 (0.5 A)', 'crimson', {}),
        (SIXT_FILE_A, 'file A (amplitude n/d)', 'darkred',
         dict(scale=1e-3, imag_flip=True)),
        (SIXT_FILE_B, 'file B (amplitude n/d)', 'seagreen',
         dict(scale=1e-3, imag_flip=True))
    ]

    bars = [b for b in bars if have(b[0])]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    width = .8 / max(len(bars), 1)

    for i2, (f, lab, c, kw) in enumerate(bars):
        vals = [
            linkk(*load(f, t, **kw))
            for t in TEMPS_IH
        ]

        ax.bar(
            np.arange(len(TEMPS_IH)) + i2 * width,
            vals,
            width,
            color=c,
            label=lab
        )

    ax.axhline(2, color='gray', ls=':')

    ax.set_xticks(
        np.arange(len(TEMPS_IH)) + width
    )

    ax.set_xticklabels(
        [l + '°C' for l in TEMPS_IH],
        fontsize=17
    )

    ax.set_ylabel('KK-consistency residual ρ (%)', fontsize=17)
    ax.set_title('Residuals by temperature', fontsize=17)
    ax.tick_params(axis='y', labelsize=17)

    if bars:
        ax.legend(fontsize=10)

    save(fig, 'Fig3b_kk_residual_temperature.png')
    

        # (b)
    fig.savefig(os.path.join(FIGS, 'Fig3b_kk_residual_temperature.pdf'), bbox_inches='tight')
    plt.close(fig)
    plt.close(fig)


    # ============================================================
    # (c) VTC5A 25 °C — pass
    # ============================================================
    if have(INHOUSE_VTC5A):

        fr, z = load(INHOUSE_VTC5A, '25')
        rr, ri = linkk(fr, z, ret_spec=True)

        fig, ax = plt.subplots(figsize=(5.2, 4.0))

        ax.semilogx(
            fr, rr, '.-',
            label='Re'
        )

        ax.semilogx(
            fr, ri, '.-',
            label='Im'
        )

        ax.set_title('VTC5A 25 °C — pass', fontsize=17)
        ax.axhline(0, color='k', lw=.6)
        ax.axhspan(-2, 2, color='gray', alpha=.15)

        ax.set_xlabel('f (Hz)', fontsize=17)
        ax.set_ylabel('KK residual (% of |Z|)', fontsize=17)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=12)

        ax.set_ylim(-3, 3)
        

        # (c)
        fig.savefig(os.path.join(FIGS, 'Fig3c_VTC5A_25C_KK.pdf'), bbox_inches='tight')
        plt.close(fig)
        save(fig, 'Fig3c_VTC5A_25C_KK.png')
        plt.close(fig)


        # ========================================================
        # (d) VTC5A −10 °C — fail
        # ========================================================
        fr, z = load(INHOUSE_VTC5A, '-10')
        rr, ri = linkk(fr, z, ret_spec=True)

        fig, ax = plt.subplots(figsize=(5.2, 4.0))

        ax.semilogx(
            fr, rr, '.-',
            label='Re'
        )

        ax.semilogx(
            fr, ri, '.-',
            label='Im'
        )

        ax.set_title('VTC5A −10 °C — fail', fontsize=17)
        ax.axhline(0, color='k', lw=.6)
        ax.axhspan(-2, 2, color='gray', alpha=.15)

        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('KK residual (% of |Z|)', fontsize=17)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=12)

        # ax.set_ylim(-3, 3)

        save(fig, 'Fig3d_VTC5A_minus10C_KK.png')

        

        # (d)
        fig.savefig(os.path.join(FIGS, 'Fig3d_VTC5A_minus10C_KK.pdf'), bbox_inches='tight')
        plt.close(fig)
        plt.close(fig)
        #################################
# ------------------------------------------------------------------ Fig 4
SHORT = {'file A': 'S2a', 'file B': 'S2b', 'in-house HE2': 'HE2', 'in-house VTC5A': 'VTC5A'}


def fig4():
    comp = csv('pipeline_comparison.csv'); mc = csv('mc_identifiability.csv')
    enum = csv('label_enumeration.csv')
    if (comp is None or len(comp) < 4 or mc is None or enum is None or len(enum) < 4
            or not have(LGM50_FITS)):
        print("[skip] Fig 4: the released figure has four panels and needs ALL of "
              "out/mc_identifiability.csv, out/label_enumeration.csv (script 13), a complete "
              "out/pipeline_comparison.csv (script 11, all four datasets) and "
              "data/corpora/all_ecm_SoH_temp.xlsx for panel (d). "
              "Released Fig4_identifiability.png left untouched."); return
    enum = enum.set_index('source').loc[list(comp.source)].reset_index()
    ncol = (mc is not None) + 2*(comp is not None) + have(LGM50_FITS)
    fig, ax = plt.subplots(1, ncol, figsize=(3.9*ncol, 3.7)); k = 0
    if mc is not None:
        x = np.arange(len(mc)); w = .38
        ax[k].bar(x-w/2, mc.one_start, w, label='single start, no ordering', color='indianred')
        ax[k].bar(x+w/2, mc.six_starts_ordered, w, label='multi-start + canonicalized', color='steelblue')
        ax[k].set_xticks(x); ax[k].set_xticklabels(mc.sep)
        # ax[k].set_yticklabels(mc.sep, labelsize=14)
        ax[k].tick_params(axis='both', labelsize=12)
        ax[k].axvspan(-0.5, 0.5, color='gray', alpha=.15)
        ax[k].text(0, max(mc.one_start)*1.04, 'non-identifiable\nregime', fontsize=9, ha='center', va='bottom', color='black')
        ax[k].set_ylim(0, max(mc.one_start)*1.3)
        ax[k].set_xlabel(r'time-constant separation $\tau_2/\tau_1$', fontsize=14)
        ax[k].set_ylabel('gross failure rate (%)', fontsize=14)
        ax[k].set_title('(a) Synthetic ground truth:\nremedies and their limit', fontsize=14); ax[k].legend(fontsize=9)
        k += 1
    if comp is not None:
        # Panel (b): the label term is the EXACT enumerated bound over all 2^k labellings
        # (script 13), not a sampled estimate - so this panel is platform-invariant and
        # matches Table 4. The admissible interval is drawn as a vertical span.
        x = np.arange(len(comp)); w = .38
        lo = enum.Ea_min_over_labels.values; hi = enum.Ea_max_over_labels.values
        can = enum.Ea_Rct_canonical.values
        ax[k].vlines(x-w/2, lo, hi, color='indianred', lw=7, alpha=.35,
                     label='admissible range, \nall $2^6$ labellings')
        ax[k].plot(x-w/2, can, 'o', color='indianred', ms=5, label='canonical labelling')
        for xi, c, l, h in zip(x-w/2, can, lo, hi):
            ax[k].annotate(f'{max(h-c, c-l):.1f}', (xi, h), textcoords='offset points',
                           xytext=(0, 8), ha='center', fontsize=10, color='indianred')
        ax[k].bar(x+w/2, comp.residual_term, w*0.8, color='steelblue',
                  label='near-degenerate solutions')
        ax[k].set_xticks(x); ax[k].set_xticklabels([SHORT.get(x, x) for x in comp.source], rotation=20, ha='right', fontsize=14)
        ax[k].set_ylabel('$E_a(R_{ct})$ and $|\\Delta E_a|$ (kJ mol$^{-1}$)', fontsize=14)
        ax[k].set_title('(b) Label degeneracy (exact)\nand near-degeneracy (measured)', fontsize=14)
        ax[k].legend(fontsize=10); k += 1
        ax[k].bar(x-w/2, comp.Ea_Rohm_reference, w, label='reference pipeline', color='seagreen')
        ax[k].bar(x+w/2, comp.Ea_Rohm_naive_raw, w, label='alternative pipeline', color='darkkhaki')
        ax[k].set_xticks(x); ax[k].set_xticklabels([SHORT.get(x, x) for x in comp.source], rotation=30, ha='right', fontsize=14)
        ax[k].set_ylabel('$E_a(R_{ohm})$ (kJ mol$^{-1}$)', fontsize=14)
        ax[k].set_title('(c) Ohmic channel is pipeline-invariant', fontsize=14); ax[k].legend(fontsize=10); k += 1
    if have(LGM50_FITS):
        d3 = pd.read_excel(LGM50_FITS, sheet_name='Sheet3')
        d3.columns = [str(c).strip() for c in d3.columns]
        srow = d3[(d3['SoH'] == 100) & (d3['SoC'] == 95) & (d3['Temp'] == 25)].reset_index()
        idx = np.arange(len(srow))
        ax[k].bar(idx-0.18, srow['Rsei']*1e3, 0.36, label='$R_{sei}$')
        ax[k].bar(idx+0.18, srow['Rct']*1e3, 0.36, label='$R_{ct}$')
        ax[k].set_xticks(idx); ax[k].set_xticklabels([f'{i+1}' for i in idx], fontsize=14)
        ax[k].set_xlabel('cell', fontsize=14); ax[k].set_ylabel('fitted R (mΩ)', fontsize=14)
        ax[k].set_title('(d) Label swaps in a\npublished parameter set', fontsize=14); ax[k].legend(fontsize=10)
    save(fig, 'Fig4_identifiability.png')
    save(fig, 'Fig4_identifiability.pdf')
########################

############################
# ------------------------------------------------------------------ Fig 5
def fig5():
    sel, arr = csv('unisa_arrhenius_input.csv'), csv('unisa_arrhenius.csv')
    if sel is None or arr is None:
        print("[skip] Fig 5: run 02_fit_unisa.py first"); return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    # the spectra retained by script 02 (one per state and temperature), i.e. exactly the
    # inputs of Table 3; state 5 is the mid state used throughout the manuscript
    med = sel.rename(columns={'T_C': 'T', 'Rohm_ohm': 'Rohm', 'Rsei_ohm': 'Rsei', 'Rct_ohm': 'Rct'})
    ridx = 5
    gg = med[med.ridx == ridx].sort_values('T'); TK = gg['T']+273.15
    for proc, mk in [('Rohm', '^'), ('Rsei', 's'), ('Rct', 'o')]:
        y = np.log(1/gg[proc]); reg = linregress(1/TK, y)
        ax[0].plot(1000/TK, y, mk, ms=4,
                   label=f'{proc}: $E_a$={-reg.slope*8.314/1000:.1f} kJ/mol, $R^2$={reg.rvalue**2:.3f}')
        xs = np.linspace((1/TK).min(), (1/TK).max(), 10)
        ax[0].plot(1000*xs, reg.intercept+reg.slope*xs, '-', lw=.8, color='gray')
    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(lo, hi + 0.35*(hi - lo))          # headroom so the legend does not cover points
    ax[0].set_xlabel('1000/T (K$^{-1}$)', fontsize=14); ax[0].set_ylabel('ln(1/R)', fontsize=14)
    ax[0].legend(fontsize=10.5, loc='upper right')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].set_title('(a) Arrhenius at mid-state', fontsize=14)
    a = arr.sort_values('ocv')
    series = [('Rct', 'o', '$E_a(R_{ct})$'), ('Rsei', 's', '$E_a(R_{sei})$'), ('Rohm', '^', '$E_a(R_{ohm})$')]
    if 'Ea_Rsum' in a.columns:
        series.insert(2, ('Rsum', 'D', '$E_a(R_{sei}+R_{ct})$'))
    for proc, mk, lab in series:
        ax[1].errorbar(a.ocv, a[f'Ea_{proc}'], yerr=a[f'CI_{proc}'], fmt=mk+'-', capsize=3, label=lab)
    ax[1].set_xlabel('OCV (V)', fontsize=14); ax[1].set_ylabel('$E_a$ (kJ mol$^{-1}$)', fontsize=14)
    ax[1].set_ylim(-3, 72)
    ax[1].legend(fontsize=11, loc='lower center', bbox_to_anchor=(0.5, 0.11), ncol=2, framealpha=0.9)
    ax[1].set_title('(b) State-resolved $E_a$ with 95% CI', fontsize=14); ax[1].grid(alpha=.3)
    ax[1].tick_params(axis='both', labelsize=14)
    save(fig, 'Fig5_arrhenius_state.png')
    save(fig, 'Fig5_arrhenius_state.pdf')


##########################
def fig6():
    sub = csv('unisa_subsampling.csv')
    if sub is None:
        print("[skip] Fig 6: run 03_subsampling.py first"); return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ks = sorted(sub.k.unique())
    ax[0].fill_between(ks, [sub[sub.k == k].Ea.min() for k in ks], [sub[sub.k == k].Ea.max() for k in ks],
                       alpha=.25, label='estimate range')
    ax[0].plot(ks, [sub[sub.k == k].Ea.median() for k in ks], 'o-', label='median')
    ax[0].axhline(sub[sub.k == max(ks)].Ea.iloc[0], color='k', ls=':', lw=1)
    axr = ax[0].twinx(); axr.plot(ks, [sub[sub.k == k].CI.median() for k in ks], 'r^--')
    axr.set_ylabel('median 95% CI (kJ mol$^{-1}$)', color='r', fontsize=16); axr.set_ylim(0, 60); axr.tick_params(axis='y', colors='r', labelsize=16)
    ax[0].set_xlabel('setpoints $k$', fontsize=16); ax[0].set_ylabel('$E_a(R_{ct})$ (kJ mol$^{-1}$)', fontsize=16)
    ax[0].tick_params(axis='both', labelsize=16)
    ax[0].set_title('(a) Grid sufficiency', fontsize=16); ax[0].legend(fontsize=10); ax[0].set_ylim(-10, 80)
    ncol = max(len(sub[sub.k == k]) for k in ks)
    M = np.full((len(ks), ncol), np.nan)
    for i, k in enumerate(ks):
        s = sub[sub.k == k].sort_values('Tmin')
        for j, (_, r) in enumerate(s.iterrows()): M[i, j] = r.Ea
    im = ax[1].imshow(M, aspect='auto', cmap='coolwarm', vmin=10, vmax=75, origin='lower')
    ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels(ks); ax[1].set_ylabel('k', fontsize=16)
    ax[1].set_xticks(range(0, ncol, 2))               # window index (integer)
    ax[1].set_xlabel('window position (cold → warm)', fontsize=16); ax[1].set_title('(b) $E_a$ by window (kJ mol$^{-1}$)', fontsize=16)
    ax[1].tick_params(axis='both', labelsize=16)
    # fig.colorbar(im, ax=ax[1])
    cbar = fig.colorbar(im, ax=ax[1])
    cbar.ax.tick_params(labelsize=13)
    save(fig, 'Fig6_subsampling.png')
    save(fig, 'Fig6_subsampling.pdf')
    # fig.savefig('Fig6_subsampling.png', dpi=300, bbox_inches='tight')
    # fig.savefig('Fig6_subsampling.pdf', bbox_inches='tight')
# ------------------------------------------------------------------ Fig 7
def fig7():
    if not have(SAMSUNG26J) or not ALL:
        print("[skip] Fig 7: the released figure needs BOTH data/corpora/impedance.csv and the "
              "parsed reference corpus; a single-panel version would drop the state confound it "
              "exists to show. Released Fig7_soc_confound.png left untouched."); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    if have(SAMSUNG26J):
        sam = pd.read_csv(SAMSUNG26J)
        freqs = [0.05, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200, 400, 1000]
        sam['f'] = sam['FREQUENCY_ID'].map({i: f for i, f in enumerate(freqs)})
        for bid, g in sam[sam.f == 0.1].groupby('BATTERY_ID'):
            gm = g.groupby('SOC')['R'].mean().sort_index()
            ax[0].plot(gm.index, gm.values*1e3, 'o-', ms=3, label=f'cell {bid}')
        ax[0].legend(fontsize=12)
    else:
        ax[0].text(.5, .5, '26J corpus not available', ha='center', va='center', transform=ax[0].transAxes)
    ax[0].set_xlabel('SoC (%)', fontsize=16); ax[0].set_ylabel("Z′(0.1 Hz) (mΩ)", fontsize=16)
    ax[0].set_title('(a) SoC sweep at fixed T', fontsize=16); ax[0].grid(alpha=.3)
    ax[0].tick_params(axis='both', labelsize=16)
    if ALL:
        zs = []
        for T in TS:
            s = ALL[T][len(ALL[T])//2]
            j = np.argmin(np.abs(s[1]-0.1)); zs.append(s[2].real[j]*1e3)
        ax[1].plot(TS, zs, 'o-', color='royalblue')
    ax[1].set_xlabel('Temperature (°C)', fontsize=16); ax[1].set_ylabel("Z′(0.1 Hz) (mΩ)", fontsize=16)
    ax[1].set_title('(b) Temperature sweep at fixed state', fontsize=16); ax[1].grid(alpha=.3)
    ax[1].tick_params(axis='both', labelsize=16)
    save(fig, 'Fig7_soc_confound.png')
    save(fig, 'Fig7_soc_confound.pdf')

# ------------------------------------------------------------------ Fig 8
def fig8():
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis('off')
    boxes = [("1. Protocol\n$I_{ac}(T)$ from cold-limit\ndesign curve (Fig. 2c)\n+ amplitude-halving check", 0.02),
             ("2. Screening\nKK consistency per spectrum\nρ ≤ 2%\nelse re-measure/flag", 0.22),
             ("3. Fitting\nfixed ECM, published\nbounds + τ-ordering +\nsequential initialization", 0.42),
             ("4. Regression\nk ≥ 8 setpoints, span ≥ 35 K\nreport $R^2$, slope SE,\n95% CI", 0.62),
             ("5. Reporting\noperating point (OCV/SoC/SoH)\nreplicates n, dispersion\ncode + data DOI", 0.82)]
    for txt, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.25), 0.16, 0.55, fc='#eef3fb', ec='steelblue'))
        ax.text(x+0.08, 0.52, txt, ha='center', va='center', fontsize=8)
    for x in [0.18, 0.38, 0.58, 0.78]:
        ax.annotate('', xy=(x+0.04, 0.52), xytext=(x, 0.52), arrowprops=dict(arrowstyle='->', color='steelblue'))
    ax.set_title('The portability contract: workflow emitted by the released tool')
    save(fig, 'Fig8_workflow.png')

if __name__ == '__main__':
    # Fig. 8 of the manuscript (workflow) is drawn by 07b_workflow_figure.py; fig8() above is
    # the earlier schematic (Fig8_workflow.png) and is no longer part of the run.
    for fn in [fig1, fig2v2, fig2, fig3v2, fig3, fig4, fig5, fig6, fig7]:
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {type(e).__name__}: {e}")
    print("done ->", FIGS)
