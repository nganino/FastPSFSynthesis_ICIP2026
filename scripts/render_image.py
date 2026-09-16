"""Per-pixel depth-of-field (DoF) rendering demo (paper Fig. 1b / Fig. 5).

Given a sharp RGB(-D) image and a per-pixel depth map, computes a
spatially-varying defocus PSF at every pixel (from the closed-form
solution, FFT, direct quadrature, or a geometric pillbox model) and
composites a blurred output image from the per-pixel-weighted PSFs. Used
to compare the different simulators' visual quality and runtime on a
realistic DoF-rendering workload.

This file is written in a notebook-like style: the function definitions
below are followed by a top-level "ENTRY GATEWAY" script section (marked
with `# %%` cell separators) that sets up a sample scene and runs the
comparison directly at import/run time -- it is not gated behind
`if __name__ == '__main__':`. Run it with `python -m scripts.render_image`
from the repository root, or execute the `# %%` cells interactively
(e.g. in Spyder / VS Code / Jupyter).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from src.closed_form import closed_form_defocus
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from scripts.main import interpolate2D, fft, create_pupil
from src.direct_quadrature import direct_hankel_h
import time

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


def create_test_texture(size=100, square_size = 10):
    """Generate a noisy checkerboard test texture.

    Builds a binary checkerboard pattern (tiles of `square_size` pixels)
    cropped to `size` x `size`, then OR's in sparse salt noise. Useful as a
    high-frequency synthetic target for visually comparing PSF blur kernels
    (e.g. in render_imagev1/v3).

    Parameters
    ----------
    size : int, optional
        Output image side length in pixels.
    square_size : int, optional
        Checkerboard tile size in pixels.

    Returns
    -------
    np.ndarray, shape (size, size)
        Values in {0, 1}.
    """
    num_squares = int(np.ceil(size / (2 * square_size)))
    base_pattern = np.array([[1, 0] * num_squares, [0, 1] * num_squares] * num_squares)
    
    # Expand to pixel grid
    checker = np.kron(base_pattern, np.ones((square_size, square_size)))
    
    # Crop to exact requested size (in case num_squares * square_size > size)
    checker = checker[:size, :size]
    
    # Generate noise with the same shape as the cropped checkerboard
    noise = np.zeros((size,size))
    noise = np.random.choice([0, 1], size=(size, size), p=[0.9, 0.1])
    
    return np.clip(checker + noise, 0, 1)
    
    return np.clip(checker + noise, 0, 1)


def generate_psf_cf(N_kernel, k, K, Cd, R_max, TLC):
    """Generate a normalized 2D defocus-only PSF kernel via the closed-form solution.

    Parameters
    ----------
    N_kernel : int
        Side length of the output PSF kernel.
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids used to
        evaluate and interpolate the closed-form solution.
    Cd : float
        Defocus coefficient. If 0, returns a delta (identity) kernel.
    R_max : float
        Aperture (pupil) radius.
    TLC : float
        Bessel-approximation operating-point hyperparameter.

    Returns
    -------
    np.ndarray, shape (N_kernel, N_kernel)
        PSF intensity kernel, normalized to sum to 1.
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
    """Generate a normalized 2D PSF intensity kernel via brute-force FFT.

    Parameters
    ----------
    dx : float
        Pupil-plane grid spacing.
    R : np.ndarray
        2D pupil-plane radial coordinate grid.
    R_max : float
        Aperture (pupil) radius.
    Cd, Cs : float
        Defocus and spherical aberration coefficients.

    Returns
    -------
    np.ndarray, same shape as R
        PSF intensity kernel, normalized to sum to 1.
    """
    pupil = create_pupil(R, R_max, Cd, Cs)
    H = fft(dx, pupil)
    H = np.abs(H)**2
    
    return H / np.sum(H)

def generate_psf_pillbox(sigma, N_kernel, R):
    """Generate a normalized geometric (pillbox / circle-of-confusion) PSF kernel.

    Approximates the PSF as a uniform disk of radius `sigma`, the classic
    geometric-optics blur-circle model (no diffraction). Falls back to a
    delta (identity) kernel when the blur circle is too small to resolve
    on the grid. Relies on module-level `wavelength` and `Z_s` (sensor
    distance) globals set in the ENTRY GATEWAY section below.

    Parameters
    ----------
    sigma : float
        Blur-circle radius.
    N_kernel : int
        Side length of the output PSF kernel.
    R : np.ndarray
        2D sensor-plane radial coordinate grid.

    Returns
    -------
    np.ndarray, shape (N_kernel, N_kernel)
        PSF intensity kernel, normalized to sum to 1.
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
    """Generate a normalized 2D defocus-only PSF kernel via direct Hankel quadrature.

    Same interface as generate_psf_cf but using the brute-force numerical
    (direct_hankel_h) reference instead of the closed-form solution.

    Parameters
    ----------
    N_kernel : int
        Side length of the output PSF kernel.
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids.
    Cd : float
        Defocus coefficient. If 0, returns a delta (identity) kernel.
    R_max : float
        Aperture (pupil) radius.

    Returns
    -------
    np.ndarray, shape (N_kernel, N_kernel)
        PSF intensity kernel, normalized to sum to 1.
    """
    if Cd == 0:
        kernel = np.zeros((N_kernel, N_kernel))
        kernel[N_kernel//2, N_kernel//2] = 1
        return kernel

    psf = direct_hankel_h(k, R_max, Cd, 0, N_kernel)
    psf2d = interpolate2D(k, psf, K)
    h = np.abs(psf2d) **2
    
    return h / np.sum(h)
    

# Bluerring Rows / Columns at a time
def render_imagev1(Z, z_i, img, N_kernel, N_img, k, K, R_max, R, dx, mode = "cf"):
    """Render a defocus-blurred image assuming one depth (and thus one PSF) per column.

    A coarse/fast approximation of per-pixel DoF rendering: each image
    column i is assigned a single defocus coefficient Cd[i] (from Z[:, 0]
    treated as a per-column depth profile), blurred with the corresponding
    PSF via FFT convolution, and only that column is kept from the result.

    Parameters
    ----------
    Z : np.ndarray
        Per-row/column object distance (only column 0 is used per row of Cd).
    z_i : float
        Ideal in-focus object distance.
    img : np.ndarray
        Sharp input image to blur.
    N_kernel : int
        Side length of the PSF kernel.
    N_img : int
        Image size (unused directly; kept for interface parity).
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids, passed
        through to generate_psf_cf.
    R_max : float
        Aperture (pupil) radius.
    R : np.ndarray
        2D pupil/sensor-plane radial coordinate grid.
    dx : float
        Grid spacing, used by the "fft" mode.
    mode : {'cf', 'fft', 'pillbox'}, optional
        Which PSF generator to use per column.

    Returns
    -------
    blurred_image : np.ndarray, same shape as img
    sigma : np.ndarray
        Per-row/column geometric blur-circle radius (pillbox model).
    """
    wavelength = 500 * 10 ** (-6) # millimeters
    f = 6
    f_num = f / (R_max)
    
    delta_Z = np.abs(Z - z_i)

    Cd = (np.pi * delta_Z) / (8 * wavelength * f_num **2) 
    
    sigma = np.abs(R_max - ((1/f) - (1/z_i)) * R_max * Z)
    
    fig, ax = plt.subplots(1,2, dpi=300)
    ax[0].plot(Cd[:,0])
    ax[0].set_title("Cd")

    ax[1].plot(sigma[:,0])
    ax[1].set_title("Sigma")
    plt.show()
    
    TLC = 50 + 10 * Cd
    
    blurred_image = np.zeros_like(img)

    
    for i in range(delta_Z.shape[0]):
        if (mode == 'cf'):
            h = generate_psf_cf(N_kernel, k, K, Cd[i][0], R_max, TLC[i][0])
        
        elif (mode == "fft"):
            h = generate_psf_fft(dx, R, R_max, Cd[i][0], 0)
        
        elif (mode == "pillbox"):
            h = generate_psf_pillbox(sigma[i][0], N_kernel, R, R_max)
        
        
        full_blurred_row = fftconvolve(img, h, mode='same')
        blurred_image[:, i] = full_blurred_row[:, i] #[:, i] is by col [i, :] is by row
        
        if (i % 20 == 0 and mode == "peillbox"):
            plt.figure()
            plt.imshow(h)
            plt.title(f"PSF at Row {i}, ModeL {mode}")
        
    
    return blurred_image, sigma

#Computing PSF at each pixel and new image is a weighted
def render_imagev2(Z, Z_s, f, img, N_kernel, N_img, sensor_r, sensor_R, R_max, R, dx, mode = "cf"):
    """Render a defocus-blurred image with one PSF evaluated per pixel.

    The full per-pixel DoF rendering pipeline (paper Fig. 5): converts the
    per-pixel depth map Z into a per-pixel defocus coefficient Cd (thin-lens
    model), generates a PSF kernel for every pixel via the selected method,
    and accumulates each pixel's brightness-weighted PSF into an
    output buffer (splat-style, rather than per-pixel convolution).

    Parameters
    ----------
    Z : np.ndarray
        Per-pixel object distance (depth map), same shape as `img`.
    Z_s : float
        Sensor plane distance.
    f : float
        Lens focal length.
    img : np.ndarray
        Sharp input image to blur.
    N_kernel : int
        Side length of the PSF kernel.
    N_img : int
        Image size (unused directly; kept for interface parity).
    sensor_r, sensor_R : np.ndarray
        Radial (1D) and 2D sensor-plane coordinate grids, passed to the
        'cf'/'direct' PSF generators.
    R_max : float
        Aperture (pupil) radius.
    R : np.ndarray
        2D pupil-plane radial coordinate grid, used by the 'fft' mode.
    dx : float
        Pupil-plane grid spacing, used by the 'fft' mode.
    mode : {'cf', 'fft', 'direct', 'pillbox'}, optional
        Which PSF generator to use per pixel.

    Returns
    -------
    blurred_image : np.ndarray, same shape as img
        Cropped back to the input image's dimensions.
    Cd : np.ndarray, same shape as img
        Per-pixel defocus coefficient map (diagnostic output).
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

def render_imagev3(N_kernel, img, k, K, Cd_list, R_max):
    """Animate "focus breathing": sweep Cd from 0 up and back down, blurring
    a fixed image at each step, and save the result as a GIF.

    Cd ramps from 0.1 up to 5.1 over the first 50 frames, then back down
    over the remaining 50, giving a defocus-in/defocus-out breathing effect.
    Saves the animation to poster_figures/media/focus_breathing.gif.

    Parameters
    ----------
    N_kernel : int
        Side length of the PSF kernel.
    img : np.ndarray
        Sharp input image to blur at each frame.
    k, K : np.ndarray
        Radial (1D) and 2D sensor-plane spatial-frequency grids, passed to
        generate_psf_cf.
    Cd_list : list
        Unused (kept for interface parity) -- the actual per-frame Cd
        schedule is computed inside `animate`.
    R_max : float
        Aperture (pupil) radius.
    """
    fig, ax = plt.subplots(dpi = 300)
    im = ax.imshow(img, cmap="gray")
    ax.set_title("Blurred Image Cd = 0")
    
    def animate(i):
        if i <= 50:
            Cd = 0.1 + 0.1 * i 
        if i > 50:
            Cd = 5.1 - 0.1*(i - 50)
        TLC = 50 + 10 * Cd
        h = generate_psf_cf(N_kernel, k, K, Cd, R_max, TLC)
        blurred_image = fftconvolve(img, h, mode='same')
        im.set_data(blurred_image)
        ax.set_title(f"Blurred Image Cd = {Cd:.1f}")
        return [im]
    
    ani = animation.FuncAnimation(
    fig,                  # The figure to animate
    animate,              # The function to call each frame
    frames=100,           # Total number of frames
    interval=10,          # Delay between frames in milliseconds
    blit=True             # Optimize drawing
    )  
    
    ani.save('poster_figures/media/focus_breathing.gif', writer='pillow', fps=10)
    
'''
-------------------------------ENTRY GATEWAY----------------------------------
'''

# --- Parameters in m ---
N = 257
wavelength = 550 * (10 ** (-9))
num_partitions = 5 
f = 0.05
Z_s = 0.0515
Z_o_ideal = 1 / (1/f - 1/Z_s)
print("Ideal Obj Distance: ", Z_o_ideal)
pupil_plane_dim = 0.01
pupil_dx =  pupil_plane_dim / N
pupil_x = np.linspace(-N/2, N/2, N) * pupil_dx
pupil_X, pupil_Y = np.meshgrid(pupil_x, pupil_x)
pupil_R = np.sqrt(pupil_X**2 + pupil_Y**2)
pupil_r = pupil_R[N//2 ][:] # slice
R_max = 0.005

sensor_dx = 1 / (pupil_dx * N)
sensor_plane_dim = sensor_dx * N
sensor_x = np.linspace(-N/2, N/2, N) * sensor_dx
sensor_X, sensor_Y = np.meshgrid(sensor_x, sensor_x)
sensor_R = np.sqrt(sensor_X**2 + sensor_Y**2)
sensor_r = sensor_R[N//2 ][:] # slice

## TEST OF RENDERING IMAGE V2 (VARIABLE PLANE OF FOCUS ON A TRUE SCENE)
# %%
N_img = 256
sharp_image_raw = Image.open("test_imgs/01421_colors.png").convert("L")
depth_array_raw = Image.open("test_imgs/01421_depth.png")
sharp_image_raw = np.asarray(sharp_image_raw)
depth_array_raw = np.asarray(depth_array_raw)
N_img_raw = sharp_image_raw.shape
if (N_img != N_img_raw[0] or N_img != N_img_raw[1]):
    sharp_image = sharp_image_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2,  (N_img_raw[1]-N_img) // 2 : (N_img_raw[1]+N_img) // 2] 
    depth_array = depth_array_raw[(N_img_raw[0]-N_img) // 2 : (N_img_raw[0]+N_img) // 2,  (N_img_raw[1]-N_img) // 2 : (N_img_raw[1]+N_img) // 2] 
Z = scale_range(depth_array, 1.45, 2.0) # m

plt.imshow(Z, cmap="gray")
plt.colorbar()
ax = plt.gca()
fig = plt.gcf()
fig.set_dpi(300)
ax.axis("off")
plt.show()

# %%


pb_start = time.time()
blurred_img_pb, Cd = render_imagev2(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode= "pillbox")
pb_end = time.time()
pb_slice = blurred_img_pb[:, N_img // 2]

plt.imshow(blurred_img_pb, cmap="gray")
plt.show()

fft_start = time.time()
blurred_img_fft, Cd = render_imagev2(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode= "fft")
fft_slice = blurred_img_fft[:, N_img // 2]
fft_end = time.time()

plt.imshow(blurred_img_fft, cmap="gray")
plt.show()

cf_start = time.time()
blurred_img_cf, Cd = render_imagev2(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode= "cf")
cf_end = time.time()
cf_slice = blurred_img_cf[:, N_img // 2]

plt.imshow(blurred_img_cf, cmap="gray")
plt.show()

direct_start = time.time()
blurred_img_direct, Cd = render_imagev2(Z, Z_s, f, sharp_image, N, N_img, sensor_r, sensor_R, R_max, pupil_R, pupil_dx, mode= "direct")
direct_slice = blurred_img_direct[:, N_img // 2]
direct_end = time.time()


times = {
    "CF" : cf_end-cf_start,
    "FFT": fft_end-fft_start,
    "Direct": direct_end-direct_start, 
    "Pillbox": pb_end-pb_start }

# %%
print(f"CF: {cf_end-cf_start:.4}, FFT {fft_end-fft_start:.2f}, Direct {direct_end-direct_start:.2f}, Pillbox {pb_end-pb_start:.2f}")

ftsize = 10
fig, ax = plt.subplots(2,2, dpi=300, layout='constrained')

ax[0,1].imshow(blurred_img_cf, cmap='gray')
ax[0,1].set_title(f"Closed Form Kernel.", fontsize=9)
ax[0,1].axis("off")
print()

ax[1,0].imshow(blurred_img_fft, cmap='gray')
ax[1,0].set_title(f"FFT Kernel.", fontsize=9)
ax[1,0].axis("off")

ax[0,0].imshow(blurred_img_direct, cmap='gray')
ax[0,0].set_title(f"Direct Hankel Kernel.", fontsize=9)
ax[0,0].axis("off")

ax[1,1].imshow(blurred_img_pb, cmap='gray')
ax[1,1].set_title(f"Pillbox Kernel.", fontsize=9)
ax[1,1].axis("off")
plt.tight_layout()
plt.savefig("poster_figures/variable_blur_comparison.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()


plt.plot(cf_slice, "b", label="closed form kernel")
plt.plot(fft_slice, "k", label="fft kernel")
plt.plot(pb_slice, "r", label="pillbox kernel")
plt.plot(direct_slice, 'y', label="direct hankel")
fig = plt.gcf()
fig.set_dpi(300)
plt.legend()
plt.xlabel("x")
plt.title("Slice Across Checkerboard Pattern for Various Blur Kernels")
plt.savefig("poster_figures/variable_blur_slice.pdf")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300, constrained_layout=True)

# Mapping images to their respective axes and metadata
plot_data = [
    (axes[0, 0], blurred_img_direct, f"Direct Hankel Kernel\n({times['Direct']:.2f}s)"),
    (axes[0, 1], blurred_img_cf,     f"Closed Form Kernel\n({times['CF']:.2f}s)"),
    (axes[1, 0], blurred_img_fft,    f"FFT Kernel\n({times['FFT']:.2f}s)"),
    (axes[1, 1], blurred_img_pb,     f"Pillbox Kernel\n({times['Pillbox']:.2f}s)")
]

for ax, img, title in plot_data:
    ax.imshow(img, cmap='gray', interpolation='bicubic')
    ax.set_title(title, pad=10)
    ax.axis("off") # Removes axes and labels
    # Optional: adds a subtle frame if preferred, otherwise leave off
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(1)

# Save as PDF for vector-quality text and lossless image compression
plt.savefig("poster_figures/variable_blur_comparison_final.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()

# Focus-breathing animation: sweep defocus up then back down and save the
# result to poster_figures/media/focus_breathing.gif.
Cd_list = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3]
render_imagev3(N, sharp_image, sensor_r, sensor_R, Cd_list, R_max)
