"""Load the pretrained per-ring alpha coefficients for the spherical-aberration
piecewise-quadratic pupil approximation (see spherical.py, closed_form.py).

Each ring of the piecewise-quadratic pupil approximation (spherical.spherical_approx)
needs a quadratic phase coefficient alpha[i] that isn't itself closed-form in
(Cd, Cs) -- it was found empirically, by sweeping alpha for a grid of (Cd, Cs)
pairs and keeping the values that best match the true combined defocus +
spherical-aberration pupil phase. The results of that sweep are stored in
data/alpha_coefficients.csv (one row per (Cd, Cs) pair) rather than as inline
Python literals, so they can be inspected/edited without touching code.
"""

import csv
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ALPHA_TABLE_PATH = os.path.join(_REPO_ROOT, "data", "alpha_coefficients.csv")


def load_alpha_table(path=DEFAULT_ALPHA_TABLE_PATH):
    """Load the full (Cd, Cs, alpha_0..alpha_4) sweep table from CSV.

    Parameters
    ----------
    path : str, optional
        Path to the alpha-coefficients CSV. Defaults to
        data/alpha_coefficients.csv at the repository root.

    Returns
    -------
    list of list[float]
        One row per (Cd, Cs) combination:
        [Cd, Cs, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4].
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [[float(v) for v in row] for row in reader]


def get_alpha_row(table, Cd, Cs):
    """Return the [Cd, Cs, alpha_0..alpha_4] row matching (Cd, Cs) exactly.

    Parameters
    ----------
    table : list of list[float]
        A table as returned by `load_alpha_table`.
    Cd, Cs : float
        Defocus / spherical aberration coefficients to look up.

    Returns
    -------
    list[float]
        The matching row.

    Raises
    ------
    ValueError
        If no row in `table` matches (Cd, Cs) exactly.
    """
    for row in table:
        if row[0] == Cd and row[1] == Cs:
            return row
    raise ValueError(f"No alpha coefficients found for Cd={Cd}, Cs={Cs}")
