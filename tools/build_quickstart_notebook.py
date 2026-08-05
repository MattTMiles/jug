"""Build (and execute) notebooks/quickstart/jug_api_quickstart.ipynb.

Run tools/make_quickstart_dataset.py first -- it writes the J1909-3744 par/tim
the notebook loads. Then regenerate the appendix usage example with:

    JAX_PLATFORMS=cpu python tools/build_quickstart_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO = Path(__file__).resolve().parent.parent
WORKDIR = REPO / "notebooks" / "quickstart"
OUT = WORKDIR / "jug_api_quickstart.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# JUG quick start\n"
        "\n"
        "The core JUG Python API in six steps, on J1909-3744 (MeerKAT, 2828 TOAs).\n"
        "The par file carries a noise model (EFAC, EQUAD, ECORR, power-law DM noise),\n"
        "so `fit_parameters()` runs a generalised least-squares fit.\n"
        "\n"
        "`J1909-3744.par` is a *deliberately perturbed* starting ephemeris: every\n"
        "fitted parameter has been offset by 3 times its par-file uncertainty, so the\n"
        "example shows the fit recovering a solution rather than starting at one.\n"
        "Those par-file uncertainties are considerably larger than the fit's own\n"
        "formal errors, so the offsets are tens of post-fit sigma and the pre-fit\n"
        "residuals are correspondingly large."
    ),
    md(
        "## 0. Backend\n"
        "\n"
        "JUG runs on whatever JAX backend is installed. This example pins the CPU\n"
        "backend so it is reproducible anywhere; export `JAX_PLATFORMS=cuda` before\n"
        "starting the kernel to run on a GPU instead."
    ),
    code(
        'import os\n'
        '\n'
        'os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must be set before JAX is imported'
    ),
    md("## 1. Open a session"),
    code(
        'from jug.engine.session import TimingSession\n'
        '\n'
        'session = TimingSession("J1909-3744.par", "J1909-3744.tim")\n'
        'print(session)'
    ),
    md("## 2. Compute pre-fit residuals"),
    code(
        'pre = session.compute_residuals()\n'
        'print(f"pre-fit RMS = {pre[\'rms_us\']:.3f} us  ({pre[\'n_toas\']} TOAs)")\n'
        'print("free parameters:", session.free_params)'
    ),
    md(
        "## 3. Fit the timing model\n"
        "\n"
        "`fit_parameters()` fits every parameter flagged free in the par file and\n"
        "auto-detects the noise model, so this is a GLS fit. Starting from the\n"
        "perturbed ephemeris, the RMS should drop by more than two orders of\n"
        "magnitude."
    ),
    code(
        'fit = session.fit_parameters()\n'
        'print(f"post-fit RMS = {fit[\'final_rms\']:.3f} us   chi2 = {fit[\'final_chi2\']:.1f}")\n'
        'print(f"{fit[\'iterations\']} iterations, converged={fit[\'converged\']}")'
    ),
    md("Pre/post-fit parameter comparison with uncertainties:"),
    code('session.parameter_table(fit)'),
    md(
        "## 4. Estimate the noise model (MAP)\n"
        "\n"
        "Stochastic-parameter estimation by SVI at fixed timing model. The default\n"
        "estimates EFAC, EQUAD, ECORR, red noise and DM noise."
    ),
    code(
        'est = session.estimate_noise()\n'
        'for name, value in est.params.items():\n'
        '    print(f"{name:<16} {value:>10.4f}")'
    ),
    md(
        "Pass any estimator option through to choose what is estimated — e.g. white\n"
        "noise and DM noise only, with a shorter SVI run:"
    ),
    code(
        'white_dm = session.estimate_noise(include_red_noise=False, max_num_batches=10)\n'
        'print(list(white_dm.params))'
    ),
    md("## 5. Write the post-fit ephemeris"),
    code(
        'session.save_par("J1909-3744_postfit.par", fit_result=fit)\n'
        'print(open("J1909-3744_postfit.par").read()[:400])'
    ),
    md(
        "## 6. Controlling which parameters are fitted\n"
        "\n"
        "`set_frozen()` / `set_free()` edit the fit flags in memory, so the fitted\n"
        "set can be changed without touching the par file. Freezing everything else\n"
        "is how you fit a subset: `fit_parameters(fit_params=[...])` *adds* to the\n"
        "par-file free parameters rather than replacing them."
    ),
    code(
        'sub = TimingSession("J1909-3744.par", "J1909-3744.tim")\n'
        'sub.set_frozen([p for p in sub.free_params if p not in ("F0", "F1", "DM1")])\n'
        'print("fitting:", sub.free_params)\n'
        'print("design matrix:", sub.fit_parameters()["design_matrix_labels"])'
    ),
    md(
        "`set_free()` also works for a parameter the par file carries but does not\n"
        "flag (`DM` here), and for a spin term the par file omits entirely (`F2`,\n"
        "which starts from 0)."
    ),
    code(
        'sub.set_free("DM", "F2")\n'
        'sub_fit = sub.fit_parameters()\n'
        'for name in ("DM", "F2"):\n'
        '    print(f"{name:<4}{sub_fit[\'final_params\'][name]:>18.9g}"\n'
        '          f" +/- {sub_fit[\'uncertainties\'][name]:.3g}")'
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(WORKDIR)}})

# Drop XLA/PjRt stderr chatter so the stored example shows only real output.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = [o for o in cell.outputs if o.get("name") != "stderr"]

nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
