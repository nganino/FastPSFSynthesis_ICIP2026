"""Reduced-term ("accelerated") variant of the closed-form PSF solution.

This mirrors closed_form.py's decomposition of the Hankel integral into six
closed-form sub-integrals (int_A1-3, int_B1-3; see closed_form.py and the
paper's Appendix, Eq. 12-16), but drops the int_A3 and int_B1 terms -- the
highest-order correction terms in the piecewise Bessel-function fit -- by
forcing them to zero instead of evaluating them. This trades a small amount
of accuracy for reduced arithmetic per call. Kept for benchmarking/comparison
against the full closed-form solution; prefer closed_form.py for accuracy.

The overall structure, variable names, and every surviving term are kept
identical to closed_form.py so the two can be diffed directly.
"""

import numpy as np
from scipy.special import erf


def erfi(z):
    """Imaginary error function: erfi(z) = -1j * erf(1j * z)."""
    return -1j*erf(1j*z)


def defocus_accelerated(N, k, Cd, R_m, TLC):
    """Accelerated closed-form PSF for pure defocus (fixed bounds [0, R_m]).

    Same interface and result definition as closed_form.closed_form_defocus,
    but with the int_A3 and int_B1 (int_B1_1, int_B1_2) correction terms
    zeroed out for speed. See that function's docstring for the full
    parameter/derivation description.
    """
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
    int_A1 = (1 / (2j * a)) * (np.exp(1j*a*TP*TP) - 1)
    int_A2 = -(b*b / 8) * ((np.exp(1j*a*TP*TP)/(a*a)) - ((1j*TP*TP*np.exp(1j*a*TP*TP)) / a) - (1 / (a*a)))
    int_A3 = (b*b*b*b/ (128*a*a*a)) * ((-1*((1j*a*a*TP*TP*TP*TP) - (2*a*TP*TP) - (2j))*np.exp(1j*a*TP*TP)) - (2j))
    # Accelerated: drop the A3 correction term.
    int_A3 = 0
    int_A = int_A1 + int_A2 + int_A3


    B1_const1 = (A_tl/2)*b*np.exp(-1j*pi/4)*np.exp((-1j*b*b)/(4*a))*(1/(8*a*a*a))
    B1_const2 = sqi*sqi*sqi * np.sqrt(pi*a) * (b*b + 2j*a)
    B2_const1 = (B_tl/2) * np.exp(-1j * pi / 4)
    B2_const2 = (1j*np.sqrt(pi)*b*np.exp(-1j*b*b / (4*a))) / (4*sqi*a*sqa)
    B3_const1 = np.exp(-1j*pi/4)*(1/(2*b))*(-1/(2*sqi*sqa)) * np.sqrt(pi) * 1j * np.exp((-1j*b*b)/(4*a))

    int_B3_1 = B3_const1 * (erf((sqi*1j*(2*a*R_m + b))/ (2*sqa)) - erf((sqi*1j*(2*a*TP + b))/ (2*sqa)))
    int_B3_2 = B3_const1 * (erf((sqi*1j*(2*a*R_m - b))/ (2*sqa)) - erf((sqi*1j*(2*a*TP - b))/ (2*sqa)))

    int_B2_1 = B2_const1 * ( (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*R_m + b)) - (1j/(2*a))*np.exp(1j*a*R_m*R_m + 1j*b*R_m)) - (B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*TP + b)) - (1j/(2*a))*np.exp(1j*a*TP*TP + 1j*b*TP)))
    int_B2_2 = B2_const1 * ( (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*R_m - b)) - (1j/(2*a))*np.exp(1j*a*R_m*R_m - 1j*b*R_m))     - (-B2_const2 * erf((1j/(2*sqa))*sqi*(2*a*TP - b)) - (1j/(2*a))*np.exp(1j*a*TP*TP - 1j*b*TP)));

    # Accelerated: the int_B1_1 / int_B1_2 correction terms (erfi-based) are
    # skipped entirely -- most expensive terms to evaluate -- and forced to 0.
    #int_B1_1 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*R_m + b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*R_m + b)*(2*a*R_m + b) / (4*a)) * (2*a*R_m - b)) - ((B1_const2*erfi((sqi*(2*a*TP + b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*TP + b)*(2*a*TP + b) / (4*a)) * (2*a*TP - b))))

    #int_B1_2 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*R_m - b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*R_m - b)*(2*a*R_m - b) / (4*a)) * (2*a*R_m + b)) - ((B1_const2*erfi((sqi*(2*a*TP - b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*TP - b)*(2*a*TP - b) / (4*a)) * (2*a*TP + b))))

    int_B1_2 = 0
    int_B1_1 = 0

    int_B = int_B1_1 + (np.exp(1j*pi/2) * int_B1_2) + int_B2_1 + (np.exp(1j*pi/2)*int_B2_2) + int_B3_1 + (np.exp(1j*pi/2)*int_B3_2)
    int_whole = np.where(TP == R_m, int_A, int_A + int_B)
    h = int_whole * 2*np.pi

    return h


def defocus_accelerated_variable(N, k, Cd, r_lower, r_upper, R_m, TLC):
    """Accelerated closed-form PSF for defocus over an arbitrary [r_lower, r_upper] ring.

    Same interface and result definition as
    closed_form.closed_form_variable, but with the int_A3 and int_B1
    correction terms zeroed out for speed (see defocus_accelerated above).
    Used by spherical_accelerated to evaluate each ring of the piecewise
    spherical-aberration approximation.
    """
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

    #Integral A
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

        # Accelerated: int_B1_1 / int_B1_2 skipped, forced to 0 (see above).
        #int_B1_1 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*UB + b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*UB + b)*(2*a*UB + b) / (4*a)) * (2*a*UB - b)) - ((B1_const2*erfi((sqi*(2*a*LB + b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*LB + b)*(2*a*LB + b) / (4*a)) * (2*a*LB - b))))

        #int_B1_2 = B1_const1 * ( ((B1_const2*erfi((sqi*(2*a*UB - b)) / (2*sqa)))) + (2*1j*a*np.exp(1j*(2*a*UB - b)*(2*a*UB - b) / (4*a)) * (2*a*UB + b)) - ((B1_const2*erfi((sqi*(2*a*LB - b)) / (2*sqa))) + (2*1j*a*np.exp(1j*(2*a*LB - b)*(2*a*LB - b) / (4*a)) * (2*a*LB + b))))

        int_B1_2 = 0
        int_B1_1 = 0

        int_B[mask_B] = int_B1_1 + (np.exp(1j*pi/2) * int_B1_2) + int_B2_1 + (np.exp(1j*pi/2)*int_B2_2) + int_B3_1 + (np.exp(1j*pi/2)*int_B3_2)

    int_whole = int_A + int_B
    h = int_whole * 2*np.pi

    return h


def spherical_accelerated(N, k, r_edges, alpha, R_m, TLCs):
    """Accelerated closed-form PSF for combined defocus + spherical aberration.

    Sums the accelerated per-ring defocus solution (defocus_accelerated_variable)
    over each radial ring of the piecewise-quadratic pupil approximation
    (see spherical.spherical_approx). Mirrors
    closed_form.closed_form_spherical, but using the reduced-term ring
    integral.

    Parameters
    ----------
    N : int
        Pupil-plane grid size (unused directly here, kept for interface
        parity with closed_form_spherical).
    k : np.ndarray
        Radial spatial frequencies at which to evaluate h.
    r_edges : np.ndarray, shape (len(alpha) + 1,)
        Ring boundary radii, as returned by spherical.spherical_approx.
    alpha : sequence of float
        Per-ring quadratic phase coefficient; Cd for ring i is alpha[i] / 2.
    R_m : float
        Aperture (pupil) radius.
    TLCs : sequence of float
        Per-ring Bessel-approximation operating-point hyperparameter,
        passed through to defocus_accelerated_variable.

    Returns
    -------
    np.ndarray, same shape as k
        Complex radial PSF amplitude h(k), summed over all rings.
    """
    h = np.zeros(k.shape, np.dtype(np.complex128))

    for i in range(0, len(alpha)):
        Cd = alpha[i] / 2
        h += 0
        h += defocus_accelerated_variable(N, k, Cd, r_edges[i], r_edges[i+1], R_m, TLCs[i])
    return h
