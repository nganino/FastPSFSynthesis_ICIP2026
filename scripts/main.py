"""Core PSF-comparison library and entry point for reproducing the paper's
accuracy/speed comparisons (Figs. 2-4).

Provides the shared pupil/FFT utilities (`create_pupil`, `fft`,
`interpolate2D`) used across this repo's scripts, plus the comparison
routines that benchmark the closed-form solution (src/closed_form.py)
against the FFT and direct-quadrature baselines for defocus-only and
combined defocus + spherical-aberration PSFs, both for accuracy
(RMSE/PSNR/cross-correlation) and runtime.

Running this file directly (`python -m scripts.main`) reproduces the
defocus-only and defocus+spherical accuracy comparison figures; the
runtime-benchmarking and cross-correlation-sweep routines
(`compare_speed`, `compare_speed_batches`, `compare_correlation`) are
defined below but not called by default -- invoke them from `main()` (or
interactively) to reproduce those results.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.closed_form import closed_form_defocus, closed_form_spherical, closed_form_variable
from src.direct_quadrature import direct_hankel_h
from src.spherical import spherical_approx
from src.closed_form_accelerated import defocus_accelerated, spherical_accelerated
from src.apply_psf import apply_psf_to_image
from src.alpha_table import load_alpha_table, get_alpha_row

from PIL import Image
from scipy.interpolate import interp1d
from scipy import signal, stats
import time



# Pretrained optimal alpha coefficients for approximating the spherical
# aberration for various combinations of Cd and Cs, obtained via an alpha
# sweep and loaded from data/alpha_coefficients.csv (see src/alpha_table.py).
# Each row is [Cd, Cs, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4].
_alpha_table = load_alpha_table()

# Default (Cd, Cs) points plotted by compare_spherical() below -- matches
# the combinations used in paper_figures/spherical_comparsion.pdf.
alpha_matrix = [
    get_alpha_row(_alpha_table, 5, 5),
    get_alpha_row(_alpha_table, 10, 5),
    get_alpha_row(_alpha_table, 10, 2),
    get_alpha_row(_alpha_table, 10, 4),
]



def create_pupil(R, R_m, Cd, Cs):
    """Build the exact (non-approximated) defocus + spherical-aberration pupil function.

    P(r) = exp(j * (Cs*(r/R_m)^4 + 2*Cd*(r/R_m)^2)) inside the aperture,
    zero outside it (paper Eq. 6). Used as the ground-truth pupil for the
    FFT baseline in `fft` below.

    Parameters
    ----------
    R : np.ndarray
        Radial coordinate at each pupil-plane grid point.
    R_m : float
        Aperture (pupil) radius.
    Cd : float
        Defocus coefficient.
    Cs : float
        Spherical aberration coefficient.

    Returns
    -------
    np.ndarray, complex128, same shape as R
        The pupil function.
    """
    pupil = np.exp(1j * (Cs*(R/R_m)**4 + 2*Cd*(R/R_m)**2))
    pupil[R > R_m] = 0

    return pupil

def fft(dx, pupil):
    """FFT-based PSF amplitude: 2D FFT of the pupil function (Eq. 9 baseline).

    Parameters
    ----------
    dx : float
        Pupil-plane grid spacing (used to scale the FFT to approximate the
        continuous Fourier transform).
    pupil : np.ndarray
        2D complex pupil function (e.g. from `create_pupil`).

    Returns
    -------
    np.ndarray, complex, same shape as pupil
        The (centered) 2D PSF amplitude H. |H|**2 is the PSF intensity.
    """
    H = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil))) * dx *dx
    return H



# compares fft, closed form integral, direct quadrature for various defocus values, Cs = 0
def compare_defocus(N, R_max, k, K,  R, dx, two_dim_compare = False):
    """Plot FFT vs. closed-form vs. direct-quadrature radial PSFs for pure defocus.

    For each Cd in Cd_list (Cs = 0), computes the radial PSF amplitude via
    all three methods and reports RMSE/PSNR/energy-deviation/cross-correlation
    of the closed-form result against the FFT reference. One row per Cd.

    When two_dim_compare is True, each row has three panels: the 2D FFT
    reference PSF, the 2D closed-form PSF, and the 1D radial slice
    comparison with the metrics annotated (matching the layout used in
    scripts/website_figure_generators.py). When False, each row is just
    the 1D radial slice comparison. Saves the resulting figure to
    poster_figures/defocus_comparsion.pdf.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    R_max : float
        Aperture (pupil) radius.
    k : np.ndarray
        Radial sensor-plane spatial frequencies (1D slice).
    K : np.ndarray
        2D sensor-plane spatial-frequency magnitude grid, used to
        interpolate the closed-form 1D result onto a 2D PSF.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    dx : float
        Pupil-plane grid spacing.
    two_dim_compare : bool, optional
        If True, plot 2D FFT / closed-form PSF panels alongside the 1D
        radial slice; if False, plot only the 1D radial slice per Cd.
    """
    Cd_list = [0.1, 1, 5, 10] # defocus strength values
    TLCs = [100, 100, 100, 200]
    Cs = 0
    ncols = 3 if two_dim_compare else 1
    fig1, ax1 = plt.subplots(len(Cd_list), ncols,
                              figsize=(14, 4 * len(Cd_list)) if two_dim_compare else (7, 4 * len(Cd_list)),
                              dpi=300, squeeze=False)

    for i in range(len(Cd_list)):
        Cd = Cd_list[i]
        pupil = create_pupil(R, R_max, Cd, Cs)
        H_fft = fft(dx, pupil) # 2d fft
        h_fft = H_fft[N//2][:] # 1d slice
        h_cf = closed_form_defocus(N, k, Cd, R_max, TLCs[i]) #closed form
        H_cf = interpolate2D(k, h_cf, K)
        h_direct = direct_hankel_h(k, R_max, Cd, Cs, N) # direct quadrature

        psf_fft = np.abs(H_fft)**2
        psf_cf  = np.abs(H_cf)**2

        rmse = np.sqrt(np.mean((np.abs(h_fft) - np.abs(h_cf))**2))
        psnr = 10*np.log10(np.max(np.abs(h_fft)**2) / (rmse**2))
        energy_deviation = np.abs(np.sum(np.abs(h_cf)**2) - np.sum(np.abs(h_fft)**2)) / np.sum(np.abs(h_fft)**2)
        x_corr = cross_correlation(np.abs(h_fft)**2, np.abs(h_cf)**2, k, coeff = True)

        if two_dim_compare:
            # 2D PSF panels on the left, radial slice on the right.
            ax1[i, 0].imshow(psf_fft, cmap='magma')
            ax1[i, 0].set_title(f"Numerical Reference (FFT)\n$C_d = {Cd}$", fontsize=12)
            ax1[i, 0].axis('off')

            ax1[i, 1].imshow(psf_cf, cmap='magma')
            ax1[i, 1].set_title(f"Closed-Form Approximation\n$C_d = {Cd}$", fontsize=12)
            ax1[i, 1].axis('off')

            slice_ax = ax1[i, 2]
        else:
            slice_ax = ax1[i, 0]

        slice_ax.plot(k, abs(h_fft), label = "fft")
        slice_ax.plot(k, abs(h_cf), label = "closed-form")
        slice_ax.plot(k, abs(h_direct), label = "direct")
        slice_ax.set_title(f"Defocus Strength = {Cd}")
        slice_ax.set_ylabel("|h|")
        slice_ax.set_xlabel("k")

        metrics_str = f"RMSE: {rmse:.4f}  PSNR: {psnr:.2f} dB\nEnergy Dev: {energy_deviation:.3f}\nCross-Corr: {x_corr:.4f}"
        slice_ax.text(0.97, 0.95, metrics_str, transform=slice_ax.transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        if i == 0:
            slice_ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(0.97, 0.62))

    plt.tight_layout()
    fig1.savefig("poster_figures/defocus_comparsion.pdf")
    plt.show()

# compares fft, closed form integral, direct quadrature for various defocus/spherical values
def compare_spherical(N, R_max, k, K, R, dx, two_dim_compare = False):
    """Plot FFT vs. closed-form (piecewise) vs. direct-quadrature radial PSFs
    for combined defocus + spherical aberration.

    For each (Cd, Cs, alpha) entry in `alpha_matrix`, builds the true and
    piecewise-quadratic-approximated pupils and computes the radial PSF via
    all three methods, reporting RMSE/PSNR/energy-deviation/cross-correlation
    of the closed-form result against the FFT reference. One row per
    (Cd, Cs) entry.

    When two_dim_compare is True, each row has three panels: the 2D FFT
    reference PSF, the 2D closed-form PSF, and the 1D radial slice
    comparison with the metrics annotated (matching the layout used in
    scripts/website_figure_generators.py). When False, each row is just
    the 1D radial slice comparison. Saves the resulting figure to
    poster_figures/spherical_comparsion.pdf.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    R_max : float
        Aperture (pupil) radius.
    k : np.ndarray
        Radial sensor-plane spatial frequencies (1D slice).
    K : np.ndarray
        2D sensor-plane spatial-frequency magnitude grid.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    dx : float
        Pupil-plane grid spacing.
    two_dim_compare : bool, optional
        If True, plot 2D FFT / closed-form PSF panels alongside the 1D
        radial slice; if False, plot only the 1D radial slice per
        (Cd, Cs) entry.
    """
    num_partitions = 5
    TLCs = [100, 100, 100, 200, 200] # for each spherical ring
    C_list = [row[:2] for row in alpha_matrix]
    alphas = [row[-5:] for row in alpha_matrix]

    ncols = 3 if two_dim_compare else 1
    fig1, ax1 = plt.subplots(len(C_list), ncols,
                              figsize=(14, 4 * len(C_list)) if two_dim_compare else (7, 4 * len(C_list)),
                              dpi=300, squeeze=False)
    for i in range(len(C_list)):
        Cd = C_list[i][0]
        Cs = C_list[i][1]
        alpha = alphas[i]

        pupil_true = create_pupil(R, R_max, Cd, Cs)
        pupil_approx, r_edges = spherical_approx(N, num_partitions, alpha, R, R_max)

        h_cf_spherical = closed_form_spherical(N, k, r_edges, alpha, R_max, TLCs)
        H_cf_spherical = interpolate2D(k, h_cf_spherical, K)
        h_spherical_direct = direct_hankel_h(k, R_max, Cd, Cs, N) # direct quadrature
        H_fft_spherical = fft(dx, pupil_true)
        h_fft_spherical = H_fft_spherical[N//2][:]

        psf_fft = np.abs(H_fft_spherical)**2
        psf_cf  = np.abs(H_cf_spherical)**2

        rmse = np.sqrt(np.mean((np.abs(h_fft_spherical) - np.abs(h_cf_spherical))**2))
        psnr = 10*np.log10( np.max(np.abs(h_fft_spherical)**2) / (rmse**2))
        energy_deviation = np.abs(np.sum(np.abs(h_cf_spherical)**2) - np.sum(np.abs(h_fft_spherical)**2)) / np.sum(np.abs(h_fft_spherical)**2)
        x_corr = cross_correlation(np.abs(h_fft_spherical), np.abs(h_cf_spherical), k, coeff = True)

        if two_dim_compare:
            # 2D PSF panels on the left, radial slice on the right.
            ax1[i, 0].imshow(psf_fft, cmap='magma')
            ax1[i, 0].set_title(f"Numerical Reference (FFT)\n$C_d={Cd}, C_s={Cs}$", fontsize=12)
            ax1[i, 0].axis('off')

            ax1[i, 1].imshow(psf_cf, cmap='magma')
            ax1[i, 1].set_title(f"Closed-Form Approximation\n$C_d={Cd}, C_s={Cs}$", fontsize=12)
            ax1[i, 1].axis('off')

            slice_ax = ax1[i, 2]
        else:
            slice_ax = ax1[i, 0]

        slice_ax.plot(k, abs(h_fft_spherical), label = "fft")
        slice_ax.plot(k, abs(h_cf_spherical), label = "closed-form spherical approx.")
        slice_ax.plot(k, abs(h_spherical_direct), label = "direct")
        slice_ax.set_title(f"Defocus Strength = {Cd} | Spherical Strength = {Cs}")
        slice_ax.set_ylabel("|h|")
        slice_ax.set_xlabel("k")

        metrics_str = f"RMSE: {rmse:.4f}  PSNR: {psnr:.2f} dB\nEnergy Dev: {energy_deviation:.3f}\nCross-Corr: {x_corr:.4f}"
        slice_ax.text(0.97, 0.95, metrics_str, transform=slice_ax.transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        if i == 0:
            slice_ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(0.97, 0.62))

    plt.tight_layout()
    fig1.savefig("poster_figures/spherical_comparsion.pdf")
    plt.show()
 
    
# compares 2D computation speed for fft, closed form integral, direct quadrature for a single Cd/Cs across many N (resolution)
def compare_speed(plane_dim, R_max, num_partitions = 5, num_rounds = 5):
    """Benchmark FFT vs. closed-form vs. direct-quadrature runtime across grid sizes.

    For a fixed, strongly-aberrated (Cd, Cs) = (10, 10) case,
    times the FFT, closed-form (spherical), and direct-quadrature PSF
    computation at each grid size in N_list, averaged over `num_rounds`
    repeats. Plots and saves execution time vs. grid size (paper Fig. 3a)
    to poster_figures/speed_comparsion.pdf.

    Parameters
    ----------
    plane_dim : float
        Physical extent of the pupil plane (overwritten to 10 internally;
        kept as a parameter for interface compatibility).
    R_max : float
        Aperture (pupil) radius.
    num_partitions : int, optional
        Number of radial rings for the spherical-aberration approximation.
    num_rounds : int, optional
        Number of timing repeats to average over.

    Returns
    -------
    np.ndarray, shape (len(N_list), 3)
        Mean [FFT, closed-form, direct] execution time (seconds) per grid size.
    """
    N_list = [256, 512, 1024, 2048, 4096]
    TLCs = [300, 300, 400, 400, 400]
    full_time = []

    batch_times = []
    plane_dim = 10
    alpha_row = get_alpha_row(_alpha_table, 10, 10)
    Cd = alpha_row[0]
    Cs = alpha_row[1]
    alpha = alpha_row[2:]
    
    for i in range(num_rounds):
        times_round = []
        for i in range(len(N_list)):
            N = N_list[i]
        
            x = np.linspace(-plane_dim/2, plane_dim/2, N)
            dx = plane_dim / N
            X,Y = np.meshgrid(x,x)
            R = np.sqrt(X*X + Y*Y)
            
            df_x = 1 / (dx * N)
            f_x = np.linspace(-(N/2)*df_x, (N/2)*df_x - df_x, N)
            F_x, F_y = np.meshgrid(f_x, f_x)
            K = np.sqrt(F_x**2 + F_y**2)
            k = K[N//2][:]
            #fft
            start_time = time.time()
            
            pupil_true = create_pupil(R, R_max, Cd, Cs)
            H_fft_spherical = fft(dx, pupil_true)
            
            fft_time = time.time()
            #cf 1d
            pupil_approx, r_edges = spherical_approx(N, num_partitions, alpha, R, R_max)
            h_cf_spherical = closed_form_spherical(N, k, r_edges, alpha, R_max, TLCs)
            H_cf_spherical2D = interpolate2D(k, h_cf_spherical, K)
            
            closed_form_time_standard = time.time()   
        
            #direct
            h_spherical_direct = direct_hankel_h(k, R_max, Cd, Cs, N) # direct quadrature
            H_spherical_direct2D = interpolate2D(k, h_spherical_direct, K)
            
            direct_time = time.time()
            
            times_round.append([fft_time - start_time, closed_form_time_standard - fft_time, direct_time - closed_form_time_standard])
        full_time.append(times_round)  
        
    ## PLOTS ##
    times_list = np.array(np.mean(full_time, 0))
    plt.figure(figsize=(10, 6), dpi = 300)
   
    plt.plot(N_list, times_list[:, 0], label="FFT", marker='o')
    plt.plot(N_list, times_list[:, 1], label="Closed Form", marker='s')
    plt.plot(N_list, times_list[:, 2], label="Direct", marker='*')
   
    plt.xlabel("Grid Size (N)", fontsize=16)
    plt.ylabel("Execution Time (seconds)", fontsize=16)
    plt.title(f"Computation Speed Comparison Averaged Over {num_rounds} Runs", fontsize=20) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.legend()
    plt.savefig("poster_figures/speed_comparsion.pdf")
    plt.show()
    
    return times_list

def compare_speed_batches(N, k, K, R, R_max, dx, num_partitions = 5):
    """Benchmark FFT vs. closed-form vs. direct-quadrature runtime vs. batch size.

    Times each method's total runtime when evaluating an increasing number
    of (Cd, Cs, alpha) entries from `alpha_matrix` back to back (1, 2, 3, ...
    up to all entries), at a fixed grid size N. Plots execution time vs.
    batch size.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    k : np.ndarray
        Radial sensor-plane spatial frequencies (1D slice).
    K : np.ndarray
        2D sensor-plane spatial-frequency magnitude grid.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    R_max : float
        Aperture (pupil) radius.
    dx : float
        Pupil-plane grid spacing.
    num_partitions : int, optional
        Number of radial rings for the spherical-aberration approximation.
    """
    batch_times = []
    for size in range(1, len(alpha_matrix)+1):
        current_alpha_matrix = alpha_matrix[:size]
        #FFT
        fft_start_time = time.time()
        for alpha_arr in current_alpha_matrix:
            Cd = alpha_arr[0]
            Cs = alpha_arr[1]
            pupil_true = create_pupil(R, R_max, Cd, Cs)
            H_fft_spherical = fft(dx, pupil_true)
            
        fft_end_time = time.time()
        
        # CLosed Form 1D
        cf_start_time = time.time()
        TLCs = [300, 300, 400, 400, 400]
        for alpha_arr in current_alpha_matrix:
            Cd = alpha_arr[0]
            Cs = alpha_arr[1]
            alpha = alpha_arr[2:]
            pupil_approx, r_edges = spherical_approx(N, num_partitions, alpha, R, R_max)
            h_cf_spherical = closed_form_spherical(N, k, r_edges, alpha, R_max, TLCs)
            H_cf_spherical2D = interpolate2D(k, h_cf_spherical, K)
            
        cf_end_time = time.time()
        
        
        
        # DIRECT
        direct_start_time = time.time()
        for alpha_arr in current_alpha_matrix:
            Cd = alpha_arr[0]
            Cs = alpha_arr[1]
            h_spherical_direct = direct_hankel_h(k, R_max, Cd, Cs, N) # direct quadrature
            H_spherical_direct2D = interpolate2D(k, h_spherical_direct, K)
            
        direct_end_time = time.time()
        batch_times.append([fft_end_time - fft_start_time, cf_end_time - cf_start_time, direct_end_time - direct_start_time])
    
    ## PLOTS ##
    batch_num = np.arange(1, 5)
    times_list= np.array(batch_times)
    plt.figure(figsize=(10, 6))
   
    plt.plot(batch_num, times_list[:, 0], label="FFT", marker='o')
    plt.plot(batch_num, times_list[:, 1], label="Closed Form", marker='s')
    plt.plot(batch_num, times_list[:, 2], label="Direct", marker='*')
   
    plt.xlabel("Batch Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Computation Speed Comparison (N = {N})") 
    plt.legend() 
    plt.show()  

def interpolate2D(k, h, K):
    """Interpolate a radial PSF amplitude h(k) onto a 2D spatial-frequency grid.

    Used to turn the 1D closed-form / direct-quadrature radial output into a
    2D PSF comparable to the FFT baseline's native 2D output.

    Parameters
    ----------
    k : np.ndarray
        1D radial spatial frequencies at which `h` was evaluated.
    h : np.ndarray
        1D (possibly complex) radial PSF amplitude, same shape as k.
    K : np.ndarray
        2D grid of radial spatial-frequency magnitudes to interpolate onto.

    Returns
    -------
    np.ndarray, same shape as K
        |h| linearly interpolated onto K, clipped to be non-negative and
        zero-filled outside the range of k.
    """
    h = np.abs(h)
    
    h_interp = interp1d(k, h, kind="linear", bounds_error = False, fill_value=0)
    h_2d = h_interp(K)
    
    h_2d = np.where(h_2d < 0, 0, h_2d) # remove any negative values generated through interpolation
    
    return h_2d

def cross_correlation(h1, h2, k, coeff = False):
    """Compare two PSF arrays via Pearson correlation or cross-correlation.

    Parameters
    ----------
    h1, h2 : np.ndarray
        PSF arrays to compare; either matching 1D radial slices or matching
        2D PSF images.
    k : np.ndarray
        Radial spatial frequencies, used only for the x-axis when plotting
        the 1D cross-correlation (coeff=False, h1/h2 1D case).
    coeff : bool, optional
        If True, return the scalar Pearson correlation coefficient between
        h1 and h2 (treated as flat data) instead of the full
        cross-correlation.

    Returns
    -------
    float or np.ndarray or None
        Pearson coefficient (coeff=True); full cross-correlation array for
        1D or 2D inputs (coeff=False); None if neither input shape matches.
    """
    c = None
    
    if coeff == True:
        flat_approx = h1
        flat_true = h2
        
        # Calculate the coefficient
        coef, p_value = stats.pearsonr(flat_approx, flat_true)
        return coef
    
    if (h1.ndim == 1 and h2.ndim == 1):
        c = np.correlate(h1,h2, mode='same')
        plt.plot(k,abs(c))
        plt.xlabel("k")
        plt.title("Cross Correlation")
        plt.show()
    
    if (h1.ndim == 2 and h2.ndim == 2):
        c = signal.fftconvolve(h1, np.conj(h2[::-1, ::-1]), mode = 'same')


    return c


def compare_correlation(N, R_max, k, K, R, dx):
    """Sweep every (Cd, Cs) entry in `alpha_matrix` and report closed-form vs.
    FFT cross-correlation for each.

    For Cs == 0 entries, uses the pure-defocus closed form directly; for
    Cs != 0 entries, builds the piecewise spherical-aberration
    approximation first. Useful for producing a (Cd, Cs) -> correlation
    heatmap/table across the tuned alpha coefficients.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    R_max : float
        Aperture (pupil) radius.
    k : np.ndarray
        Radial sensor-plane spatial frequencies (1D slice).
    K : np.ndarray
        2D sensor-plane spatial-frequency magnitude grid.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    dx : float
        Pupil-plane grid spacing.

    Returns
    -------
    list of tuple (Cd, Cs, cross_correlation_coefficient)
        One entry per row of `alpha_matrix`.
    """
    num_partitions = 5
    TLCs = [100, 100, 100, 200, 200] # for each spherical 
    C_list = [row[:2] for row in alpha_matrix]
    alphas = [row[-5:] for row in alpha_matrix]
    r_matrix = []
    
    print(C_list)
    print(len(C_list))
    for i in range(0, len(C_list)):
        Cd = C_list[i][0]
        Cs = C_list[i][1]

        
        pupil_true = create_pupil(R, R_max, Cd, Cs)
        if Cs == 0:
            TLC = 50 + 10*Cd
            h_cf = closed_form_defocus(N, k, Cd, R_max, TLC)
        else:
            alpha = alphas[i]
            pupil_approx, r_edges = spherical_approx(N, num_partitions, alpha, R, R_max)
            h_cf = closed_form_spherical(N, k, r_edges, alpha, R_max, TLCs)
            
        H_fft_spherical = np.abs(fft(dx, pupil_true))
        h_fft_spherical = H_fft_spherical[N//2, :]
        H_cf = np.abs(interpolate2D(k, h_cf, K))

        cross_coeff = cross_correlation(np.abs(h_cf), np.abs(h_fft_spherical), k, coeff=True)
        r_matrix.append((Cd, Cs, cross_coeff))

    return r_matrix


def main():
    """Reproduce the paper's core defocus and defocus+spherical accuracy comparisons.

    Sets up a shared pupil-plane / sensor-plane sampling grid, then runs
    `compare_defocus` and `compare_spherical` (both with 2D PSF plots
    enabled) to regenerate poster_figures/defocus_comparsion.pdf and
    poster_figures/spherical_comparsion.pdf.
    """
    # PLANE SETTINGS
    N = 512
    plane_dim = 20
    R_max = 1
    num_partitions = 5

    x = np.linspace(-plane_dim/2, plane_dim/2, N)
    dx = plane_dim / N
    X,Y = np.meshgrid(x,x)
    R = np.sqrt(X*X + Y*Y)
    r = R[N//2 ][:] #slice

    df_x = 1 / (dx * N)
    f_x = np.linspace(-(N/2)*df_x, (N/2)*df_x - df_x, N)
    F_x, F_y = np.meshgrid(f_x, f_x)
    K = np.sqrt(F_x**2 + F_y**2)
    k = K[N//2 ][:] # slice

    #-------------CORE COMPARISONS-------------------

    compare_defocus(N, R_max, k, K, R, dx, two_dim_compare = True) # compares solely defocus
    compare_spherical(N, R_max, k, K, R, dx, two_dim_compare = True) # compares defocus + spherical

if __name__ == '__main__':
    main()
