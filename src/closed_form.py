"""Closed-form evaluation of the defocus / spherical-aberration diffraction integral.

Implements the paper's main result: an approximate closed-form solution to
the Hankel diffraction integral

    h(k) = 2*pi * int_0^R  P(r) * J0(2*pi*k*r) * r  dr

obtained by substituting a piecewise rational/cosine approximation of the
zeroth-order Bessel function J0 (paper Eq. 3) into the integral, which
reduces it to six Gaussian-type sub-integrals (int_A1-3 near the origin,
int_B1-3 further out, split at the crossover radius TP where the Bessel
approximation switches branches) that each admit a closed form in terms of
the error function `erf` / imaginary error function `erfi` (paper Appendix,
Eq. 12-16).

- `closed_form_defocus` handles pure defocus over the full aperture [0, R_m].
- `closed_form_variable` is the same solution generalized to an arbitrary
  radial sub-interval [r_lower, r_upper], which is what lets each ring of
  the piecewise spherical-aberration approximation (see spherical.py) reuse
  the defocus-only closed form.
- `closed_form_spherical` sums closed_form_variable over all rings to
  approximate the full defocus + spherical-aberration PSF (paper Eq. 6-7).

See closed_form_accelerated.py for a faster variant that drops the smallest
correction terms.
"""

import numpy as np
from scipy.special import erf


def erfi(z):
    """Imaginary error function: erfi(z) = -1j * erf(1j * z)."""
    return -1j*erf(1j*z)


def closed_form_defocus(N, k, Cd, R_m, TLC):
    """Closed-form radial PSF amplitude for pure defocus (Cs = 0).

    Evaluates h(k) = 2*pi * int_0^R_m exp(j*2*Cd*(r/R_m)^2) * J0(2*pi*k*r) * r dr
    in closed form by splitting the integration range at the crossover
    radius TP = min(2/(2*pi*k), R_m) between the two branches of the
    piecewise Bessel approximation (paper Eq. 3) and summing the six
    resulting Gaussian-type sub-integrals (paper Appendix Eq. 12-16).

    Parameters
    ----------
    N : int
        Kernel/grid size (unused directly here; kept for interface parity
        with the other PSF-generation functions in this repo).
    k : np.ndarray
        Radial spatial frequencies (sensor-plane coordinates) at which to
        evaluate h. Modified in place to replace exact zeros with a small
        epsilon (division by k appears in the derivation).
    Cd : float
        Defocus coefficient.
    R_m : float
        Aperture (pupil) radius.
    TLC : float
        Operating-point hyperparameter (`alpha` in the paper) controlling
        where the piecewise Bessel approximation switches branches.

    Returns
    -------
    np.ndarray, same shape as k
        Complex radial PSF amplitude h(k).
    """
    k[k==0] = 1e-10
    # some constants to help with calculations
    pi = np.pi
    a = 2*Cd / (R_m**2)
    sqa = np.sqrt(a)
    sqi = np.sqrt(1j)
    A_tl = -0.5 * np.power(TLC, -1.5)
    B_tl = (1/np.sqrt(TLC)) + (np.power(TLC, -1.5) * (TLC/2.0))
    b = 2*pi*k
    TP = np.minimum(2/b, R_m)

    # integrals ( 6 parts: A (1-3) and B (1-3))
    # Part A: near-origin branch of the Bessel approximation, integrated
    # over [0, TP].
    int_A1 = (1 / (2j * a)) * (np.exp(1j*a*TP*TP) - 1)
    int_A2 = -(b*b / 8) * ((np.exp(1j*a*TP*TP)/(a*a)) - ((1j*TP*TP*np.exp(1j*a*TP*TP)) / a) - (1 / (a*a)))
    int_A3 = (b*b*b*b/ (128*a*a*a)) * ((-1*((1j*a*a*TP*TP*TP*TP) - (2*a*TP*TP) - (2j))*np.exp(1j*a*TP*TP)) - (2j))
    int_A = int_A1 + int_A2 + int_A3

    # Constants shared by the Part B (far-field branch) sub-integrals.
    B1_const1 = (A_tl/2)*b*np.exp(-1j*pi/4)*np.exp((-1j*b*b)/(4*a))*(1/(8*a*a*a))
    B1_const2 = sqi*sqi*sqi * np.sqrt(pi*a) * (b*b + 2j*a)
    B2_const1 = (B_tl/2) * np.exp(-1j * pi / 4)
    B2_const2 = (1j*np.sqrt(pi)*b*np.exp(-1j*b*b / (4*a))) / (4*sqi*a*sqa)
    B3_const1 = np.exp(-1j*pi/4)*(1/(2*b))*(-1/(2*sqi*sqa)) * np.sqrt(pi) * 1j * np.exp((-1j*b*b)/(4*a))

    # Part B: far-field branch of the Bessel approximation, integrated over
    # [TP, R_m], expressed via erf (B2, B3) and erfi (B1).
    int_B3_1 = B3_const1 * (erf((sqi*1j*(2*a*R_m + b))/ (2*sqa)) - erf((sqi*1j*(2*a*TP + b))/ (2*sqa)))
    int_B3_2 = B3_const1 * (erf((sqi*1j*(2*a*R_m - b))/ (2*sqa)) - erf((sqi*1j*(2*a*TP - b))/ (2*sqa)))

    int_B2_1 = B2_const1 * ( (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*R_m + b)) - (1j/(2*a))*np.exp(1j*a*R_m*R_m + 1j*b*R_m)) - (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*TP + b)) - (1j/(2*a))*np.exp(1j*a*TP*TP + 1j*b*TP)))
    int_B2_2 = B2_const1 * ( (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*R_m - b)) - (1j/(2*a))*np.exp(1j*a*R_m*R_m - 1j*b*R_m))     - (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*TP - b)) - (1j/(2*a))*np.exp(1j*a*TP*TP - 1j*b*TP)));

    int_B1_1 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*R_m + b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*R_m + b)*(2*a*R_m + b) / (4*a)) * (2*a*R_m - b)) - ((B1_const2*erfi((sqi*(2*a*TP + b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*TP + b)*(2*a*TP + b) / (4*a)) * (2*a*TP - b))))

    int_B1_2 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*R_m - b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*R_m - b)*(2*a*R_m - b) / (4*a)) * (2*a*R_m + b)) - ((B1_const2*erfi((sqi*(2*a*TP - b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*TP - b)*(2*a*TP - b) / (4*a)) * (2*a*TP + b))))

    int_B = int_B1_1 + (np.exp(1j*pi/2) * int_B1_2) + int_B2_1 + (np.exp(1j*pi/2)*int_B2_2) + int_B3_1 + (np.exp(1j*pi/2)*int_B3_2)
    # If TP == R_m, the far-field branch never applies (the near-origin
    # branch covers the whole aperture), so skip adding int_B.
    int_whole = np.where(TP == R_m, int_A, int_A + int_B)
    h = int_whole * 2*np.pi

    return h


# closed form solution with variable bounds, used for spherical approximation
def closed_form_variable(N, k, Cd, r_lower, r_upper, R_m, TLC):
    """Closed-form radial PSF contribution from a single radial ring [r_lower, r_upper].

    Same derivation as closed_form_defocus, generalized to an arbitrary
    sub-interval of the aperture instead of the full [0, R_m] range. This is
    the building block closed_form_spherical uses to sum contributions from
    each ring of the piecewise-quadratic spherical-aberration approximation
    (see spherical.spherical_approx). Each output element is evaluated with
    either the near-origin (A) or far-field (B) branch of the Bessel
    approximation depending on where the ring [r_lower, r_upper] falls
    relative to the crossover radius TP for that k -- or split across both
    if TP falls inside the ring.

    Parameters
    ----------
    N : int
        Kernel/grid size (unused directly here; kept for interface parity).
    k : np.ndarray
        Radial spatial frequencies at which to evaluate h. Modified in place
        to replace exact zeros with a small epsilon.
    Cd : float
        Effective defocus coefficient for this ring (alpha[i] / 2 when
        called from closed_form_spherical).
    r_lower, r_upper : float
        Inner and outer radius of this ring.
    R_m : float
        Aperture (pupil) radius.
    TLC : float
        Operating-point hyperparameter for this ring's Bessel approximation.

    Returns
    -------
    np.ndarray, same shape as k
        Complex radial PSF amplitude contribution from this ring.
    """
    k[k==0] = 1e-10
    UB = r_upper * np.ones(k.shape,np.dtype(np.complex128))
    LB = r_lower * np.ones(k.shape,np.dtype(np.complex128))
    # some constants to help with calculations
    pi = np.pi
    a = 2*Cd / (R_m**2)
    sqa = np.sqrt(a)
    sqi = np.sqrt(1j)
    A_tl = -0.5 * np.power(TLC, -1.5)
    B_tl = (1/np.sqrt(TLC)) + (np.power(TLC, -1.5) * (TLC/2.0))
    b = 2*pi*k
    b_true = b
    TP = np.minimum(2/b, R_m)


    # integrals ( 6 parts: A (1-3) and B (1-3))
    int_A = np.zeros(k.shape,np.dtype(np.complex128))
    int_B = np.zeros(k.shape,np.dtype(np.complex128))

    # Integral A: the portion of [r_lower, r_upper] that falls below the
    # per-k crossover radius TP (near-origin Bessel branch), i.e.
    # [r_lower, min(r_upper, TP)]. Only evaluated where that sub-interval is
    # non-empty.
    effective_LB_A = r_lower * np.ones(k.shape)
    effective_UB_A = np.minimum(r_upper, TP)
    mask_A = effective_UB_A > effective_LB_A

    if np.any(mask_A):
        LB = effective_LB_A[mask_A]
        UB = effective_UB_A[mask_A]
        b = b_true[mask_A]
        int_A1 = (1 / (2j * a)) * (np.exp(1j*a*UB*UB) - np.exp(1j*a*LB*LB))
        int_A2 = -(b*b / 8) * ((np.exp(1j*a*UB*UB)/(a*a)) - ((1j*UB*UB*np.exp(1j*a*UB*UB)) / a) - ((np.exp(1j*a*LB*LB)/(a*a)) - ((1j*LB*LB*np.exp(1j*a*LB*LB)) / a)))
        int_A3 = (b*b*b*b/ (128*a*a*a)) * ((-1*((1j*a*a*UB**4) - (2*a*UB*UB) - (2j))*np.exp(1j*a*UB*UB)) - (-1*((1j*a*a*LB**4) - (2*a*LB*LB) - (2j))*np.exp(1j*a*LB*LB)))
        int_A[mask_A] = int_A1 + int_A2 + int_A3

    # Integral B: the portion of [r_lower, r_upper] at or beyond TP
    # (far-field Bessel branch), i.e. [max(r_lower, TP), r_upper].
    effective_LB_B = np.maximum(r_lower, TP)
    effective_UB_B = r_upper * np.ones(k.shape)
    mask_B = effective_UB_B > effective_LB_B
    if np.any(mask_B):
        LB = effective_LB_B[mask_B]
        UB = effective_UB_B[mask_B]
        b = b_true[mask_B]
        B1_const1 = (A_tl/2)*b*np.exp(-1j*pi/4)*np.exp((-1j*b*b)/(4*a))*(1/(8*a*a*a))
        B1_const2 = sqi*sqi*sqi * np.sqrt(pi*a) * (b*b + 2j*a)
        B2_const1 = (B_tl/2) * np.exp(-1j * pi / 4)
        B2_const2 = (1j*np.sqrt(pi)*b*np.exp(-1j*b*b / (4*a))) / (4*sqi*a*sqa)
        B3_const1 = np.exp(-1j*pi/4)*(1/(2*b))*(-1/(2*sqi*sqa)) * np.sqrt(pi) * 1j * np.exp((-1j*b*b)/(4*a))

        int_B3_1 = B3_const1 * (erf((sqi*1j*(2*a*UB + b))/ (2*sqa)) - erf((sqi*1j*(2*a*LB + b))/ (2*sqa)))
        int_B3_2 = B3_const1 * (erf((sqi*1j*(2*a*UB - b))/ (2*sqa)) - erf((sqi*1j*(2*a*LB - b))/ (2*sqa)))

        int_B2_1 = B2_const1 * ( (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*UB + b)) - (1j/(2*a))*np.exp(1j*a*UB*UB + 1j*b*UB)) - (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*LB + b)) - (1j/(2*a))*np.exp(1j*a*LB*LB + 1j*b*LB)))
        int_B2_2 = B2_const1 * ( (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*UB - b)) - (1j/(2*a))*np.exp(1j*a*UB*UB - 1j*b*UB))- (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*LB - b)) - (1j/(2*a))*np.exp(1j*a*LB*LB - 1j*b*LB)))

        int_B1_1 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*UB + b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*UB + b)*(2*a*UB + b) / (4*a)) * (2*a*UB - b)) - ((B1_const2*erfi((sqi*(2*a*LB + b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*LB + b)*(2*a*LB + b) / (4*a)) * (2*a*LB - b))))

        int_B1_2 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*UB - b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*UB - b)*(2*a*UB - b) / (4*a)) * (2*a*UB + b)) - ((B1_const2*erfi((sqi*(2*a*LB - b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*LB - b)*(2*a*LB - b) / (4*a)) * (2*a*LB + b))))

        int_B[mask_B] = int_B1_1 + (np.exp(1j*pi/2) * int_B1_2) + int_B2_1 + (np.exp(1j*pi/2)*int_B2_2) + int_B3_1 + (np.exp(1j*pi/2)*int_B3_2)

    int_whole = int_A + int_B
    h = int_whole * 2*np.pi

    return h


def closed_form_spherical(N, k, r_edges, alpha, R_m, TLCs):
    """Closed-form radial PSF for combined defocus + spherical aberration.

    Sums closed_form_variable over each radial ring of the piecewise-quadratic
    pupil approximation (see spherical.spherical_approx), coherently adding
    every ring's complex amplitude contribution (paper Eq. 6-7).

    Parameters
    ----------
    N : int
        Kernel/grid size (unused directly here; kept for interface parity).
    k : np.ndarray
        Radial spatial frequencies at which to evaluate h.
    r_edges : np.ndarray, shape (len(alpha) + 1,)
        Ring boundary radii, as returned by spherical.spherical_approx.
    alpha : sequence of float
        Per-ring quadratic phase coefficient; the effective defocus
        coefficient for ring i is alpha[i] / 2.
    R_m : float
        Aperture (pupil) radius.
    TLCs : sequence of float
        Per-ring Bessel-approximation operating-point hyperparameter.

    Returns
    -------
    np.ndarray, same shape as k
        Complex radial PSF amplitude h(k), summed over all rings.
    """
    h = np.zeros(k.shape, np.dtype(np.complex128))

    for i in range(0, len(alpha)):
        Cd = alpha[i] / 2
        h += closed_form_variable(N, k, Cd, r_edges[i], r_edges[i+1], R_m, TLCs[i])
    return h
