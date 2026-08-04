"""Build (and execute) notebooks/jug_api_quickstart.ipynb.

Regenerate the appendix usage example with:

    JAX_PLATFORMS=cpu python tools/build_quickstart_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "jug_api_quickstart.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# JUG quick start\n"
        "\n"
        "The core JUG Python API in six steps, on J1909-3744 (MeerKAT, 2828 TOAs).\n"
        "The par file carries a noise model (EFAC, EQUAD, ECORR, power-law DM noise),\n"
        "so `fit_parameters()` runs a generalised least-squares fit."
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
        'session = TimingSession("../tests/data_golden/J1909_parity_noise.par",\n'
        '                        "../tests/data_golden/J1909_parity.tim")\n'
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
        "auto-detects the noise model, so this is a GLS fit."
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
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(REPO / "notebooks")}})

# Drop XLA/PjRt stderr chatter so the stored example shows only real output.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = [o for o in cell.outputs if o.get("name") != "stderr"]

nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
