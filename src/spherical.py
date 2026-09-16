"""Piecewise-quadratic approximation of the spherically-aberrated pupil function.

The true pupil phase under combined defocus + spherical aberration contains a
quartic term (Cs * r^4 / R^4) that has no closed-form Hankel-transform solution
(paper Eq. 6). To work around this, the pupil is partitioned into a small
number of radial rings and the phase within each ring is replaced with a
locally-fit quadratic (paper Eq. 7, Fig. 2b). Each ring can then be evaluated
with the same closed-form defocus solution used elsewhere in this package
(see closed_form.closed_form_variable).
"""

import numpy as np


def spherical_approx(N, num_partitions, alpha, R, R_m, spacing='linear', spacing_exp=0.5):
    """Build a piecewise-quadratic approximation of the aberrated pupil function.

    Partitions the aperture [0, R_m] into `num_partitions` radial rings and
    assigns each ring a purely quadratic phase exp(j * alpha[i] * r^2), where
    each alpha[i] is a coefficient (pre-fit or predicted, see
    scripts/training_for_alpha.py) that best approximates the true combined
    defocus + spherical phase within that ring.

    Parameters
    ----------
    N : int
        Side length of the (N, N) pupil-plane grid.
    num_partitions : int
        Number of radial rings (M in the paper) to partition the aperture into.
    alpha : sequence of float
        Per-ring quadratic phase coefficient, one entry per partition.
    R : np.ndarray, shape (N, N)
        Radial coordinate at each pupil-plane grid point.
    R_m : float
        Aperture (pupil) radius.
    spacing : {'linear', 'poly'}, optional
        Ring-edge spacing scheme. 'linear' uses evenly spaced edges; 'poly'
        warps the normalized edges by `r_edges ** spacing_exp` before scaling
        to R_m (e.g. to pack narrower rings near the aperture edge, where the
        quartic term varies fastest).
    spacing_exp : float, optional
        Exponent used when spacing == 'poly'. Ignored otherwise.

    Returns
    -------
    pupil : np.ndarray, shape (N, N), complex128
        The piecewise-quadratic approximate pupil function.
    r_edges : np.ndarray, shape (num_partitions + 1,)
        Radii of the ring boundaries, from 0 to R_m.
    """
    # Ring boundaries in normalized [0, 1] space, optionally warped, then
    # scaled out to the true aperture radius.
    r_edges = np.linspace(0, 1, num_partitions + 1)
    if spacing == 'poly':
        r_edges = r_edges ** spacing_exp
    r_edges *= R_m

    pupil = np.zeros((N, N), dtype=np.complex128)

    # Assign every grid point its ring's quadratic phase coefficient.
    for i in range(num_partitions):
        idxs = (R >= r_edges[i]) & (R < r_edges[i + 1])
        pupil[idxs] = np.exp(1j * alpha[i] * R[idxs] ** 2)

    return pupil, r_edges
