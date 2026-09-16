"""Higher-DPI variants of the accuracy-comparison figures, with an SSIM metric.

Same purpose as scripts/main.py's compare_defocus/compare_spherical (FFT vs.
closed-form vs. direct-quadrature accuracy comparisons), but rendered at
higher DPI/resolution and with an additional structural-similarity (SSIM)
metric for the paper/website figures. Also includes an instrumented variant
of the per-pixel DoF renderer (render_imagev2_instrumented) and a
standalone `render_rendering_race_video` function that uses it to
regenerate the "rendering race" comparison video in poster_figures/media/.

Requires `torch` and `torchmetrics` in addition to this repo's core
dependencies, solely for the SSIM computation in `compute_ssim`; and
`opencv-python` (cv2), solely for `render_rendering_race_video`'s video
encoding.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.closed_form import closed_form_defocus, closed_form_spherical, closed_form_variable
from src.direct_quadrature import direct_hankel_h
from src.spherical import spherical_approx
from src.closed_form_accelerated import defocus_accelerated, spherical_accelerated
from src.apply_psf import apply_psf_to_image
from src.alpha_table import load_alpha_table, get_alpha_row
from scripts.main import interpolate2D, fft, create_pupil
from torchmetrics.functional import structural_similarity_index_measure as ssim

from PIL import Image
from scipy.interpolate import interp1d
from scipy import signal, stats
import time
import torch


# Pretrained optimal alpha coefficients for approximating the spherical
# aberration for various combinations of Cd and Cs, obtained via an alpha
# sweep and loaded from data/alpha_coefficients.csv (see src/alpha_table.py).
# Each row is [Cd, Cs, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4].
_alpha_table = load_alpha_table()

# Default (Cd, Cs) points plotted by compare_spherical() below.
# (1, 0) is a pure-defocus point (Cs = 0): compare_spherical special-cases
# Cs == 0 to skip the spherical approximation entirely, so its alpha values
# are unused placeholders.
alpha_matrix = [
    [1, 0, 0, 0, 0, 0, 0],
    get_alpha_row(_alpha_table, 2, 2),
    get_alpha_row(_alpha_table, 5, 5),
]


def compute_ssim(img1, img2, data_range):
    """Compute structural similarity (SSIM) between two 2D PSF intensity images.

    Thin wrapper around torchmetrics' SSIM: converts both inputs to a
    (1, 1, H, W) float tensor and returns a scalar.

    Parameters
    ----------
    img1, img2 : np.ndarray, 2D
        PSF intensity images to compare (must be the same shape).
    data_range : float
        Value range of the inputs (e.g. img1.max() - img1.min()), required
        by SSIM's normalization.

    Returns
    -------
    float
        SSIM score in [-1, 1] (1 = identical).
    """
    # Convert numpy -> torch tensor, cast to float32, reshape to (B, C, H, W)
    t1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
    return ssim(t1, t2, data_range=data_range).item()

def scale_range(arr, new_min, new_max):
    """
    Scales a NumPy array's values to a new minimum and maximum range.
    """
    old_min = arr.min()
    old_max = arr.max()
    # Ensure no division by zero for constant arrays
    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2.0)

    scaled_arr = (arr - old_min) / (old_max - old_min) # Map to [0, 1]
    return new_min + scaled_arr * (new_max - new_min)  # Map to [new_min, new_max]


def generate_psf_cf(N_kernel, k, K, Cd, R_max, TLC):
    """Generate a normalized 2D defocus-only PSF kernel via the closed-form solution.

    See scripts/render_image.py's generate_psf_cf for the same helper
    (duplicated here so this module doesn't depend on render_image.py).
    """
    if Cd == 0:
        kernel = np.zeros((N_kernel, N_kernel))
        kernel[N_kernel//2, N_kernel//2] = 1
        return kernel

    psf = closed_form_defocus(N_kernel, k, Cd, R_max, TLC)
    psf2d = interpolate2D(k, psf, K)
    h = np.abs(psf2d) **2

    # Final re-normalization after crop
    return h /  np.sum(h)

def generate_psf_fft(dx, R, R_max, Cd, Cs):
    """Generate a normalized 2D PSF intensity kernel via brute-force FFT."""
    pupil = create_pupil(R, R_max, Cd, Cs)
    H = fft(dx, pupil)
    H = np.abs(H)**2

    return H / np.sum(H)

def generate_psf_pillbox(sigma, N_kernel, R):
    """Generate a normalized geometric (pillbox / circle-of-confusion) PSF kernel.

    NOTE: references `wavelength` and `Z_s` as globals, but neither is
    defined at module scope in this file (`wavelength` is only a local
    inside `main`) -- this function will raise NameError if called as-is.
    It is not exercised by the current `main()` flow. See
    scripts/render_image.py's generate_psf_pillbox for the working version
    and fuller docstring.
    """
    R = R * wavelength * Z_s
    min_r = np.min(R)
    
    h = np.zeros((N_kernel,N_kernel), dtype=float)
    if (sigma <=(min_r)):
        h = np.zeros((N_kernel, N_kernel))
        h[N_kernel//2, N_kernel//2] = 1
        return h
    
    h[R <= sigma] = 1  / sigma**2
    
    if (np.sum(h) == 0):
        print('sum of pb is 0: sigma/2,', sigma/2, "min r", min_r)
        h = np.zeros((N_kernel, N_kernel))
        h[N_kernel//2, N_kernel//2] = 1
        return h
    return np.abs(h) / np.sum(h)

def generate_psf_hankel(N_kernel, k, K, Cd, R_max):
    """Generate a normalized 2D defocus-only PSF kernel via direct Hankel quadrature."""
    if Cd == 0:
        kernel = np.zeros((N_kernel, N_kernel))
        kernel[N_kernel//2, N_kernel//2] = 1
        return kernel

    psf = direct_hankel_h(k, R_max, Cd, 0, N_kernel)
    psf2d = interpolate2D(k, psf, K)
    h = np.abs(psf2d) **2
    
    return h / np.sum(h)

def compare_defocus(N, R_max, k, K,  R, dx, two_dim_compare = False):
    """Render a 3-column (FFT PSF / closed-form PSF / radial slice) comparison
    figure for pure defocus at several Cd values, with PSNR/SSIM annotations.

    Higher-DPI, SSIM-augmented counterpart to scripts/main.py's
    compare_defocus. Saves to poster_figures/defocus_comparsion.pdf.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    R_max : float
        Aperture (pupil) radius.
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    dx : float
        Pupil-plane grid spacing.
    two_dim_compare : bool, optional
        Unused (kept for interface parity with scripts/main.py's version;
        this function always renders the 2D PSF panels).
    """
    Cd_list = [1, 5, 10] # defocus strength values
    TLCs = [100, 100, 200]
    Cs = 0
    fig, ax = plt.subplots(len(Cd_list), 3, figsize=(14, 12), dpi=600) 
        
    for i in range(len(Cd_list)):
        Cd = Cd_list[i]
        pupil = create_pupil(R, R_max, Cd, Cs)
        H_fft = fft(dx, pupil) # 2d fft
        h_fft = H_fft[N//2][:] # 1d slice
        h_cf = closed_form_defocus(N, k, Cd, R_max, TLCs[i]) #closed form
        h_direct = direct_hankel_h(k, R_max, Cd, Cs, N) # direct quadrature
        H_cf = interpolate2D(k, h_cf, K)
        
        # Calculate intensities (Power Spectral Density)
        psf_fft = np.abs(H_fft)**2
        psf_cf  = np.abs(H_cf)**2
        psf_direct = np.abs(h_direct)**2
        
        #Metrics
        rmse = np.sqrt(np.mean((np.abs(h_fft) - np.abs(h_cf))**2))
        psnr = 10*np.log10( np.max(np.abs(h_fft)**2) / (rmse**2))
        ssim1 = compute_ssim(psf_fft, psf_cf,
                      data_range=(psf_fft).max() - (psf_fft).min())
    
        # Plot 1: Standard FFT Reference
        im0 = ax[i,0].imshow(psf_fft, cmap='magma')
        if (i >= 0):
            ax[i,0].set_title(f"Numerical Reference (FFT)\n$C_d = {Cd}$", fontsize=14, fontweight='bold')
        else:
            ax[i,0].set_title(f"$C_d = {Cd}$", fontsize=14, fontweight='bold')
        ax[i,0].axis('off')
    
        # Plot 2: Your Closed-Form Approximation
        im1 = ax[i,1].imshow(psf_cf, cmap='magma')
        if (i >= 0):
            ax[i,1].set_title(f"Closed-Form Approximation\n$C_d = {Cd}$", fontsize=14, fontweight='bold')
        else:
            ax[i,1].set_title(f"$C_d = {Cd}$", fontsize=14, fontweight='bold')
        ax[i,1].axis('off')
        
        ax[i,2].plot(k, abs(h_fft), label = "fft")
        ax[i,2].plot(k, abs(h_cf), label = "closed-form")
        ax[i,2].plot(k, abs(h_direct), label = "direct")
        ax[i,2].set_title(f"Radial Slice Comparison $C_d = {Cd}$", fontweight='bold')
        ax[i,2].set_ylabel("|h|")
        ax[i,2].set_xlabel("k")

        # Metrics text box on the radial slice plot
        metrics_str = f"PSNR = {psnr:.2f} dB\nSSIM = {ssim1:.5f}"
        ax[i,2].text(0.97, 0.95, metrics_str, transform=ax[i,2].transAxes,
                     fontsize=12, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        if i == 0:
            ax[i,2].legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.97, 0.80))
    
    plt.tight_layout()
    
    # Pull column 2 (index 1) closer to column 1 (index 0), row by row,
    # shifting column 3 (index 2) along with it so its own gap is preserved
    shift = 0.05  # tune this: larger = columns 1&2 sit closer together
    for i in range(len(Cd_list)):
        pos1 = ax[i,1].get_position()
        pos2 = ax[i,2].get_position()
        ax[i,1].set_position([pos1.x0 - shift, pos1.y0, pos1.width, pos1.height])
        ax[i,2].set_position([pos2.x0 - shift, pos2.y0, pos2.width, pos2.height])
    
    plt.savefig("poster_figures/defocus_comparsion.pdf")
    plt.show()

    return
    
# compares fft, closed form integral, direct quadrature for various defocus/spherical values
def compare_spherical(N, R_max, k, K, R, dx, two_dim_compare = True):
    """Render a 3-column (FFT PSF / closed-form PSF / radial slice) comparison
    figure for each (Cd, Cs, alpha) entry in `alpha_matrix`, with PSNR/SSIM
    annotations.

    Higher-DPI, SSIM-augmented counterpart to scripts/main.py's
    compare_spherical. Saves to poster_figures/spherical_comparsion.pdf.

    Parameters
    ----------
    N : int
        Pupil/sensor grid size.
    R_max : float
        Aperture (pupil) radius.
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    dx : float
        Pupil-plane grid spacing.
    two_dim_compare : bool, optional
        Unused (kept for interface parity; this function always renders
        the 2D PSF panels).
    """
    num_partitions = 5
    # Standardizing TLCs - ensuring we have enough for the closed_form_spherical call
    TLC_vals = [100, 100, 200, 200, 200] 
    
    C_list = [row[:2] for row in alpha_matrix]
    alphas = [row[-5:] for row in alpha_matrix]
    num_rows = len(C_list)
    
    # Create a grid like compare_test: [FFT PSF, CF PSF, Radial Slices]
    fig, ax = plt.subplots(num_rows, 3, figsize=(14, 4 * num_rows), dpi=600)
    
    # Handle single-row case to keep indexing consistent
    if num_rows == 1:
        ax = np.expand_dims(ax, axis=0)

    for i in range(num_rows):
        Cd = C_list[i][0]
        Cs = C_list[i][1]
        alpha = alphas[i]
        
        # Data Generation
        pupil_true = create_pupil(R, R_max, Cd, Cs)
        pupil_approx, r_edges = spherical_approx(N, num_partitions, alpha, R, R_max)
        
        if (Cs == 0):
            h_cf_spherical = closed_form_defocus(N, k, Cd, R_max, TLC_vals[i]) #closed form
            H_cf_spherical = interpolate2D(k, h_cf_spherical, K)
        else:
            h_cf_spherical = closed_form_spherical(N, k, r_edges, alpha, R_max, TLC_vals)
            H_cf_spherical = interpolate2D(k, h_cf_spherical, K)
        
        h_spherical_direct = direct_hankel_h(k, R_max, Cd, Cs, N)
        H_fft_spherical = fft(dx, pupil_true)
        h_fft_spherical = H_fft_spherical[N//2][:]

        # Calculate Intensities
        psf_fft_spherical = np.abs(H_fft_spherical)**2
        psf_cf_spherical = np.abs(H_cf_spherical)**2
        
        #Metrics
        rmse = np.sqrt(np.mean((np.abs(h_fft_spherical) - np.abs(h_cf_spherical))**2))
        psnr = 10*np.log10( np.max(np.abs(h_fft_spherical)**2) / (rmse**2))
        ssim1 = compute_ssim(psf_fft_spherical, psf_cf_spherical,
                      data_range=(psf_fft_spherical).max() - (psf_fft_spherical).min())

        # --- Column 1: Numerical Reference ---
        im0 = ax[i, 0].imshow(psf_fft_spherical, cmap='magma')
        ax[i, 0].set_title(f"Numerical Reference (FFT)\n$C_d={Cd}, C_s={Cs}$", fontsize=14, fontweight='bold')
        ax[i, 0].axis('off')
        # --- Column 2: Closed-Form Approximation ---
        im1 = ax[i, 1].imshow(psf_cf_spherical, cmap='magma')
        ax[i, 1].set_title(f"Closed-Form Approximation\n$C_d={Cd}, C_s={Cs}$", fontsize=14, fontweight='bold')
        ax[i, 1].axis('off')
        # --- Column 3: Radial Slice Comparison ---
        ax[i, 2].plot(k, abs(h_fft_spherical), label="FFT")
        ax[i, 2].plot(k, abs(h_cf_spherical), label="Closed-form")
        ax[i, 2].plot(k, abs(h_spherical_direct), label="Direct")
        
        ax[i, 2].set_title(f"Radial Slice Comparison $C_s = {Cs}, C_d = {Cd}$", fontsize=14, fontweight='bold')
        ax[i, 2].set_ylabel("|h|")
        ax[i, 2].set_xlabel("k")

        # Metrics text box
        metrics_str = f"PSNR = {psnr:.2f} dB\nSSIM = {ssim1:.5f}"
        ax[i, 2].text(0.97, 0.95, metrics_str, transform=ax[i, 2].transAxes,
                       fontsize=12, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        if i == 0:
            ax[i, 2].legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.97, 0.80))
            
    plt.tight_layout()

    shift = 0.05  # tune as needed
    for i in range(num_rows):
        pos1 = ax[i, 1].get_position()
        pos2 = ax[i, 2].get_position()
        ax[i, 1].set_position([pos1.x0 - shift, pos1.y0, pos1.width, pos1.height])
        ax[i, 2].set_position([pos2.x0 - shift, pos2.y0, pos2.width, pos2.height])
        
    plt.savefig("poster_figures/spherical_comparsion.pdf")
    plt.show()

def render_imagev2_instrumented(Z, Z_s, f, img, N_kernel, N_img, sensor_r, sensor_R, R_max, R, dx, mode="cf", frame_callback=None):
    """Per-pixel DoF rendering (see render_imagev2) instrumented with a
    per-row progress callback, for recording the "rendering race" comparison
    video (see the commented block at the end of this file).

    Same rendering logic and parameters as `render_imagev2`, but simplified
    (no pillbox mode, no live diagnostic plots) and augmented with
    `frame_callback`.

    Parameters
    ----------
    Z, Z_s, f, img, N_kernel, N_img, sensor_r, sensor_R, R_max, R, dx, mode
        See `render_imagev2`.
    frame_callback : callable(row, t_elapsed, partial_image), optional
        If given, called after each completed image row with the row
        index, elapsed wall-clock time since the start of rendering, and a
        cropped copy of the partially-rendered output so progress can be
        recorded frame-by-frame.

    Returns
    -------
    np.ndarray, same shape as img
        The fully blurred image, cropped back to the input dimensions.
    """
    half = N_kernel // 2
    padded_rows = img.shape[0] + 2 * half
    padded_cols = img.shape[1] + 2 * half
    blurred_buffer = np.zeros((padded_rows, padded_cols))
    
    f_num = f / (2*R_max)
    obj_focusing_dist = 1 / ((1/f) - (1/Z))
    Z_o_ideal = 1 / (1/f - 1/Z_s)
    
    delta_Z_i = np.abs(Z_s - obj_focusing_dist)
    Cd = np.pi * R_max**2 * np.abs(Z_o_ideal - Z) / (2 * wavelength * Z * Z_o_ideal)
    #Cd = (np.pi * delta_Z_i) / (8 * wavelength * f_num **2) 
    #Cd = (np.pi * delta_Z_i * R_max**2) / (wavelength * Z_s**2)
    Cd = np.clip(Cd, 0.01, Cd.max())
    TLC = 50 + 10 * Cd
    
    t_start = time.perf_counter()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (mode == 'cf'):
                h = generate_psf_cf(N_kernel, sensor_r, sensor_R, Cd[i][j], R_max, TLC[i][j])
            
            elif (mode == "fft"):
                h = generate_psf_fft(dx, R, R_max, Cd[i][j], 0)
                
            elif (mode == "direct"):
                h = generate_psf_hankel(N_kernel, sensor_r, sensor_R, Cd[i][j], R_max)

            blurred_buffer[i:i+N_kernel, j:j+N_kernel] += img[i,j] * h
        
        if frame_callback is not None:
            t_elapsed = time.perf_counter() - t_start
            # Crop to output size before saving
            partial = blurred_buffer[half:half+img.shape[0], half:half+img.shape[1]].copy()
            frame_callback(i, t_elapsed, partial)
    
    return blurred_buffer[half:half+img.shape[0], half:half+img.shape[1]]
    
#Computing PSF at each pixel and new image is a weighted
def render_imagev2(Z, Z_s, f, img, N_kernel, N_img, sensor_r, sensor_R, R_max, R, dx, mode = "cf"):
    """Render a defocus-blurred image with one PSF evaluated per pixel.

    Identical rendering logic to scripts/render_image.py's render_imagev2
    (per-pixel thin-lens defocus + brightness-weighted PSF splatting), with
    extra diagnostic plots of the Cd/sigma maps along the way. See that
    function's docstring for the full parameter/return description.
    """
    # z_i ideal object focal plane
    # z_s sensor plane
    f_num = f / (2*R_max)
    obj_focusing_dist = 1 / ((1/f) - (1/Z))
    Z_o_ideal = 1 / (1/f - 1/Z_s)
    
    delta_Z_i = np.abs(Z_s - obj_focusing_dist)
    Cd = np.pi * R_max**2 * np.abs(Z_o_ideal - Z) / (2 * wavelength * Z * Z_o_ideal)
    #Cd = (np.pi * delta_Z_i) / (8 * wavelength * f_num **2) 
    #Cd = (np.pi * delta_Z_i * R_max**2) / (wavelength * Z_s**2)
    Cd = np.clip(Cd, 0.01, Cd.max())

    sigma = np.abs(R_max - ((1/f) - (1/Z)) * R_max * Z_s)
    sigma_diameter = 2*R_max * (np.abs(Z - Z_o_ideal) / Z) * (f / (Z_o_ideal - f))
    
    fig, ax = plt.subplots(1,2, dpi=300)
    ax[0].plot(Cd[50,:])
    ax[0].set_title("Cd")

    ax[1].plot(Z[50, :], sigma[50,:])
    ax[1].set_title("Sigma")
    plt.show()
    
    TLC = 50 + 10 * Cd
    
    # Buffer must be large enough to hold the 'spillover' of the PSF from edge pixels
    half = N_kernel // 2
    padded_rows = img.shape[0] + 2 * half
    padded_cols = img.shape[1] + 2 * half
    blurred_buffer = np.zeros((padded_rows, padded_cols))
    
    for i in range(img.shape[0]):
        #print(i)
        for j in range(img.shape[1]):
            if (mode == 'cf'):
                h = generate_psf_cf(N_kernel, sensor_r, sensor_R, Cd[i][j], R_max, TLC[i][j])
            
            elif (mode == "fft"):
                h = generate_psf_fft(dx, R, R_max, Cd[i][j], 0)
                
            elif (mode == "direct"):
                h = generate_psf_hankel(N_kernel, sensor_r, sensor_R, Cd[i][j], R_max)
                
            elif (mode == "pillbox"):
                h = generate_psf_pillbox(sigma[i][j], N_kernel, sensor_R)

            r_s, r_e = i, i + N_kernel
            c_s, c_e = j, j + N_kernel
            
            blurred_buffer[r_s : r_e, c_s : c_e] += img[i, j] * h
        if (i == 100):
            plt.imshow(h)
            plt.show()
    # Crop back to the original image dimensions
    # The center of the (0,0) PSF is at [half, half]
    return blurred_buffer[half : half + img.shape[0], half : half + img.shape[1]], Cd
    
def main():
    """Entry point: regenerate the high-DPI comparison figure and preview a
    real DoF scene's depth map.

    Sets up the pupil/sensor sampling grid, calls `compare_spherical` to
    reproduce poster_figures/spherical_comparsion.pdf, then loads a sample RGB-D
    scene (test_imgs/01421_*.png) and displays its depth map as a quick
    sanity check of the "real time DoF synthesizer" inputs. The
    defocus-only comparison (`compare_defocus`) is defined above but left
    commented out below -- uncomment to run it. To actually render that
    scene through the PSF simulators and produce a comparison video, see
    `render_rendering_race_video` (not called automatically -- it's slow).
    """
    # PLANE SETTINGS
    N = 129 # kernel size
    num_partitions = 5
    wavelength = 550 * 10**(-9)
    pupil_plane_dim = 5
    pupil_dx =  pupil_plane_dim / N
    pupil_x = np.linspace(-N/2, N/2, N) * pupil_dx
    pupil_X, pupil_Y = np.meshgrid(pupil_x, pupil_x)
    pupil_R = np.sqrt(pupil_X**2 + pupil_Y**2)
    pupil_r = pupil_R[N//2 ][:] # slice
    R_max = 1
    
    sensor_dx = 1 / (pupil_dx * N)
    sensor_plane_dim = sensor_dx * N
    sensor_x = np.linspace(-N/2, N/2, N) * sensor_dx
    sensor_X, sensor_Y = np.meshgrid(sensor_x, sensor_x)
    sensor_R = np.sqrt(sensor_X**2 + sensor_Y**2)
    sensor_r = sensor_R[N//2 ][:] # slice
    
    #-------------CORE COMPARISONS-------------------
    #compare_test(N, R_max, sensor_r, sensor_R, pupil_R, pupil_dx, two_dim_compare = True) # compares solely defocus
    compare_spherical(N, R_max, sensor_r, sensor_R, pupil_R, pupil_dx)
    
    #-------------Real Time DoF Synthesizer----------#
    f = 0.05
    Z_s = 0.0515
    Z_o_ideal = 1 / (1/f - 1/Z_s)
    print("Ideal Obj Distance: ", Z_o_ideal)

    N_img = 256
    sharp_image_raw = Image.open("test_imgs/01421_colors.png").convert("L")
    depth_array_raw = Image.open("test_imgs/01421_depth.png")
    sharp_image_raw = np.asarray(sharp_image_raw)
    depth_array_raw = np.asarray(depth_array_raw)
    N_img_raw = sharp_image_raw.shape
    if (N_img != N_img_raw[0] or N_img != N_img_raw[1]):
        sharp_image = sharp_image_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2, -100 + (N_img_raw[1]-N_img) // 2 : -100 + (N_img_raw[1]+N_img) // 2]
        depth_array = depth_array_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2,  -100 + (N_img_raw[1]-N_img) // 2 : -100 + (N_img_raw[1]+N_img) // 2]
    Z = scale_range(depth_array, 1.6, 1.8) # m

    plt.imshow(Z, cmap="gray")
    plt.colorbar()
    ax = plt.gca()
    fig = plt.gcf()
    fig.set_dpi(300)
    ax.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
    
def render_rendering_race_video(output_path="poster_figures/media/rendering_race.avi"):
    """Render a DoF scene with the closed-form, FFT, and direct-quadrature PSF
    generators side by side, recording each method's progress over time, and
    encode the result as a "rendering race" comparison video.

    Self-contained: builds its own pupil/sensor grid and loads the same
    sample RGB-D scene used by `main()` (test_imgs/01421_*.png), then runs
    `render_imagev2_instrumented` once per method, recording a timestamped
    snapshot of the partially-rendered image after every row. Those
    snapshots are then scrubbed along a shared timeline and encoded frame
    by frame into an MJPG-codec .avi video via OpenCV.

    Not called automatically by `main()` or the `__main__` guard -- this
    does three full per-pixel renders of a 256x256 scene plus video
    encoding, which can take several minutes. Call it explicitly to
    regenerate poster_figures/media/rendering_race.avi. Requires opencv-python (`cv2`),
    which nothing else in this repo depends on.

    Parameters
    ----------
    output_path : str, optional
        Where to save the encoded .avi video.
    """
    import cv2  # local import: only this function needs opencv-python

    # --- Scene / grid setup (mirrors main()'s PLANE SETTINGS + Real Time
    # DoF Synthesizer block) ---
    N = 129  # kernel size
    pupil_plane_dim = 5
    pupil_dx = pupil_plane_dim / N
    pupil_x = np.linspace(-N/2, N/2, N) * pupil_dx
    pupil_X, pupil_Y = np.meshgrid(pupil_x, pupil_x)
    pupil_R = np.sqrt(pupil_X**2 + pupil_Y**2)
    R_max = 1

    sensor_dx = 1 / (pupil_dx * N)
    sensor_x = np.linspace(-N/2, N/2, N) * sensor_dx
    sensor_X, sensor_Y = np.meshgrid(sensor_x, sensor_x)
    sensor_R = np.sqrt(sensor_X**2 + sensor_Y**2)
    sensor_r = sensor_R[N//2][:]

    f = 0.05
    Z_s = 0.0515
    Z_o_ideal = 1 / (1/f - 1/Z_s)
    print("Ideal Obj Distance: ", Z_o_ideal)

    N_img = 256
    sharp_image_raw = Image.open("test_imgs/01421_colors.png").convert("L")
    depth_array_raw = Image.open("test_imgs/01421_depth.png")
    sharp_image_raw = np.asarray(sharp_image_raw)
    depth_array_raw = np.asarray(depth_array_raw)
    N_img_raw = sharp_image_raw.shape
    if (N_img != N_img_raw[0] or N_img != N_img_raw[1]):
        sharp_image = sharp_image_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2, -100 + (N_img_raw[1]-N_img) // 2 : -100 + (N_img_raw[1]+N_img) // 2]
        depth_array = depth_array_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2,  -100 + (N_img_raw[1]-N_img) // 2 : -100 + (N_img_raw[1]+N_img) // 2]
    Z = scale_range(depth_array, 1.6, 1.8)  # m

    # --- Instrumented triple render: record each method's progress over time ---
    recordings = {method: [] for method in ['Our Method', 'fft', 'direct']}

    def make_callback(method_name):
        def callback(row, t, partial_img):
            recordings[method_name].append((row, t, partial_img))
        return callback

    # Run sequentially — clean, reproducible
    render_imagev2_instrumented(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode='cf',     frame_callback=make_callback('Our Method'))
    print('cf complete')
    render_imagev2_instrumented(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode='fft',    frame_callback=make_callback('fft'))
    print('fft complete')
    render_imagev2_instrumented(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode='direct', frame_callback=make_callback('direct'))
    print('direct complete')

    T_max = max(recordings[m][-1][1] for m in recordings)
    n_anim_frames = 200  # animation frames, NOT render rows
    timeline = np.linspace(0, T_max * 1.05, n_anim_frames)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    axes[0].imshow(sharp_image, cmap='gray')
    axes[0].set_title("Reference\n(Sharp)")
    axes[0].axis('off')
    im_handles = {}
    title_handles = {}

    for ax, method in zip(axes[1:], ['Our Method', 'fft', 'direct']):
        im_handles[method] = ax.imshow(np.zeros_like(sharp_image, dtype=float), cmap='gray', vmin=0, vmax=float(sharp_image.max()))
        title_handles[method] = ax.set_title(method)
        ax.axis('off')

    def get_frame_at_time(method, t):
        """Binary search or linear scan to find latest recorded frame <= t"""
        times = [r[1] for r in recordings[method]]
        idx = np.searchsorted(times, t, side='right') - 1
        if idx < 0:
            return np.zeros_like(sharp_image, dtype=float)
        return recordings[method][idx][2]

    # --- Encode the recorded race to video frame-by-frame via OpenCV ---
    fig.set_dpi(200)
    fig.canvas.draw()
    h_px, w_px = fig.canvas.get_width_height()[::-1]  # (height, width)

    w_px = w_px if w_px % 2 == 0 else w_px - 1
    h_px = h_px if h_px % 2 == 0 else h_px - 1

    out = cv2.VideoWriter(output_path,
                           cv2.VideoWriter_fourcc(*'MJPG'),
                           24, (w_px, h_px))

    for frame_idx in range(n_anim_frames):
        t = timeline[frame_idx]
        for method in ['Our Method', 'fft', 'direct']:
            img_frame = get_frame_at_time(method, t)
            im_handles[method].set_data(img_frame)
            t_method_end = recordings[method][-1][1]
            status = f"✓ {t_method_end:.2f}s" if t >= t_method_end else f"{t:.2f}s..."
            title_handles[method].set_text(f"{method.upper()}\n{status}")

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]
        bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    print(f"Saved to {output_path}")