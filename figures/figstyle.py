# ===========================================================================
# Shared matplotlib styling for the manuscript figure panels.
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
# ===========================================================================
"""figstyle.py

One place for the figure font scale, so every scripted panel shares a single
legible, consistent typography (camera-ready readability pass addressing
Reviewer 2's "increase axis-label, scale, and legend font sizes").

Panels are composited (and downscaled) into multi-panel manuscript figures, so
native font sizes are set generously here. Call ``apply()`` once at the top of
a plotting script, before any figure is created. In-plot annotations (bar-value
labels, n= counts, inline notes) keep explicit sizes via the named constants
below rather than the rcParams defaults, which govern axis labels, ticks,
legends, and titles.
"""
import matplotlib as mpl

# In-plot annotation sizes (used explicitly where rcParams don't apply).
ANNOT = 10      # primary numeric annotations (bar values, inline means)
ANNOT_SM = 9    # secondary annotations where horizontal space is tight
ANNOT_XS = 7.5  # very tight in-track / in-bar labels (dense multi-panel figs)

# Display labels for probe attributes / event grouping (nicer than raw keys).
ATTR_LABELS = {
    "psi":            "PSI",
    "donor_score":    "Donor score",
    "acceptor_score": "Acceptor score",
    "gc_content":     "GC content",
    "element_length": "Element length",
    "cell_type":      "Cell type",
}


def apply(compact=False, poster=False):
    """Install the shared, legible font scale into matplotlib's rcParams.

    compact=True uses a smaller, tuned scale for dense multi-panel figures
    (e.g. Figure 1's genome-track and 4-up distribution panels), which are
    wide and get downscaled in the LaTeX composite; the default scale suits
    the Figure 2/3 panels that composite closer to 1:1.

    poster=True uses a larger scale for conference-poster assets, viewed at
    ~1-1.5 m rather than on a journal page; it takes precedence over compact.
    Manuscript figures never pass poster=True, so their output is unchanged.
    """
    if poster:
        s = dict(base=15, title=19, label=17, tick=15, legend=14, figtitle=22)
    elif compact:
        s = dict(base=10, title=11, label=10.5, tick=9.5, legend=9, figtitle=12)
    else:
        s = dict(base=12, title=14, label=13, tick=12, legend=11, figtitle=15)
    mpl.rcParams.update({
        "font.size":        s["base"],
        "axes.titlesize":   s["title"],
        "axes.titleweight": "bold",
        "axes.labelsize":   s["label"],
        "xtick.labelsize":  s["tick"],
        "ytick.labelsize":  s["tick"],
        "legend.fontsize":  s["legend"],
        "legend.title_fontsize": s["legend"],
        "figure.titlesize": s["figtitle"],
        "figure.titleweight": "bold",
        # Embed TrueType fonts so composited PDFs stay text-editable (and keep
        # real glyphs when rasterized in GIMP) rather than flattened to outlines.
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })
