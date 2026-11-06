"""Fig. 8: workflow for temperature-dependent EIS analysis (figs/EIS_analysis_workflow.pdf/.png).

Adapted from the authors' block_dia.py; output is written to figs/ (not the current
directory) and the window is not opened, so the script also runs headless.
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '00_paths.py')).read())

FIGS = os.path.join(REPO_ROOT, 'figs'); os.makedirs(FIGS, exist_ok=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12

# The figure is printed at full text width (about 7 in), so it is drawn at 14 in with short
# lines: the body text then prints at about 7.5 pt (18 in and long lines gave about 4 pt).
fig, ax = plt.subplots(figsize=(14, 4.9))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.9); ax.axis("off")
box_w, box_h, y, gap = 2.45, 3.75, 0.15, 0.35
x_positions = [0.2 + i*(box_w + gap) for i in range(5)]
titles = ["1. Excitation\nprotocol", "2. Data validity\nscreening", "3. ECM parameter\nextraction",
          "4. Arrhenius\nanalysis", "5. Reproducible\nreporting"]
texts = [
    r"$I_{\mathrm{ac}}(T)$ from the" "\ncold-limit design" "\ncurve" "\nAmplitude-halving" "\nverification",
    "KK consistency test" "\nfor each spectrum" "\n" + r"$\rho \leq$ acceptance" "\nthreshold, otherwise" "\nremeasure or flag",
    "Fixed ECM with" "\npredefined bounds" "\nBroad deterministic" "\nmulti-start" "\n" + r"$\tau$-ordered labels;"
    "\n" + r"$R_{\mathrm{SEI}}+R_{\mathrm{ct}}$ if the" "\nsplit is not resolved",
    r"$k \geq 8$ setpoints" "\n" + r"span $\geq 35$ K" "\n" + r"Report $E_{\mathrm{a}}$, $R^2$," "\nslope SE, 95% CI",
    "Operating state" "\n(OCV/SoC/SoH) and" "\nstate-matching rule" "\nReplicates and" "\ndispersion" "\nCode and data DOI",
]
for x, title, body in zip(x_positions, titles, texts):
    ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.05",
                                linewidth=1.4, edgecolor="steelblue", facecolor="aliceblue"))
    ax.text(x + box_w/2, y + box_h - 0.48, title, ha="center", va="center", fontsize=16,
            fontweight="bold", linespacing=1.15)
    ax.text(x + box_w/2, y + 1.45, body, ha="center", va="center", fontsize=15, linespacing=1.3)
for i in range(len(x_positions) - 1):
    ax.add_patch(FancyArrowPatch((x_positions[i] + box_w + 0.06, y + box_h/2),
                                 (x_positions[i+1] - 0.06, y + box_h/2),
                                 arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="dimgray"))
ax.text(7, 4.62, "Workflow for temperature-dependent EIS analysis", ha="center", va="center",
        fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "EIS_analysis_workflow.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "EIS_analysis_workflow.png"), dpi=600, bbox_inches="tight")
print("wrote figs/EIS_analysis_workflow.pdf/.png")
