"""Ground-truth Hankel-integral evaluation via direct numerical quadrature.

This is the "brute force" reference used to sanity-check the closed-form
approximations in closed_form.py / closed_form_accelerated.py: it evaluates
the diffraction integral

    h(k) = 2*pi * int_0^R  exp(j * (Cs*(r/R)^4 + 2*Cd*(r/R)^2)) * J0(2*pi*k*r) * r  dr

directly on a dense linear radial grid with the trapezoid rule, with no
Bessel-function or pupil-phase approximation involved. It is accurate but
scales as O(N) numerical integration *per* output k, i.e. the reference
baseline the closed-form method is compared against for both accuracy and
runtime (paper Sec. 3).
"""

import numpy as np
from scipy.special import j0


def direct_hankel_h(k, R_m, Cd, Cs, Nr):
    """Evaluate the radial PSF amplitude h(k) by direct trapezoidal quadrature.

    h(k) = 2*pi * int_0^R_m  exp(j*(Cs*(r/R_m)^4 + 2*Cd*(r/R_m)^2)) * J0(2*pi*k*r) * r  dr

    Parameters
    ----------
    k : np.ndarray
        Radial spatial frequencies (sensor-plane coordinates) at which to
        evaluate h.
    R_m : float
        Aperture (pupil) radius.
    Cd : float
        Defocus coefficient.
    Cs : float
        Spherical aberration coefficient (0 for defocus-only).
    Nr : int
        Number of radial quadrature samples used to discretize [0, R_m].

    Returns
    -------
    np.ndarray, same shape as k
        Complex radial PSF amplitude h(k).
    """
    # Dense linear radial grid over the aperture, and the pupil phase at
    # each sample (defocus + spherical aberration terms).
    r = np.linspace(0, R_m, Nr)
    f = np.exp(1j * (Cs * (r / R_m) ** 4 + 2.0 * Cd * (r / R_m) ** 2))

    # Broadcast r against every requested k: shape (Nk, Nr).
    kr = 2.0 * np.pi * k[:, None] * r[None, :]
    integrand = f[None, :] * j0(kr) * r[None, :]

    # Trapezoid rule in r, then apply the leading 2*pi factor from the
    # Hankel transform.
    val = np.trapezoid(integrand, r, axis=1)
    return 2.0 * np.pi * val


def direct_hankel_batch(k, R_m, Cd_list, Cs_list, N):
    """Evaluate direct_hankel_h for a batch of (Cd, Cs) pairs.

    Parameters
    ----------
    k : np.ndarray
        Radial spatial frequencies shared by every PSF in the batch.
    R_m : float
        Aperture (pupil) radius.
    Cd_list, Cs_list : sequence of float
        Per-PSF defocus and spherical aberration coefficients; must be the
        same length.
    N : int
        Number of radial quadrature samples passed through to
        direct_hankel_h (also the output width).

    Returns
    -------
    np.ndarray, shape (len(Cd_list), N)
        Stacked radial PSF amplitudes, one row per (Cd, Cs) pair.
    """
    if len(Cd_list) != len(Cs_list):
        print("Please have Cd list and Cs list be same length")
        return

    batch = np.zeros((len(Cd_list), N))
    for i in range(len(Cd_list)):
        batch[i] = direct_hankel_h(k, R_m, Cd_list[i], Cs_list[i], N)

    return batch
