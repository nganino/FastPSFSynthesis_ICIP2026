"""Utility for applying a synthesized PSF kernel to an image via convolution.

Used by the DoF-rendering scripts to visualize/sanity-check a single PSF
kernel (e.g. from closed_form.py) by blurring a test image with it, as
opposed to the full per-pixel, spatially-varying rendering pipeline in
scripts/render_image.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d


def apply_psf_to_image(image_path, psf_kernel, output_path="blurred_output.jpg",
                        img_array=None, save=False, plot=True):
    """Convolve an image with a PSF kernel and optionally plot/save the result.

    Parameters
    ----------
    image_path : str
        Path to the input image, used only when `img_array` is not supplied.
    psf_kernel : np.ndarray
        2D PSF kernel (e.g. |h|^2 from one of the simulators in this repo).
        Does not need to be pre-normalized; it is rescaled to sum to 1 so the
        convolution preserves overall image brightness.
    output_path : str, optional
        Where to save the blurred result if `save=True`.
    img_array : np.ndarray, optional
        Pre-loaded grayscale image array. If provided, `image_path` is not
        read from disk.
    save : bool, optional
        If True, write the blurred image to `output_path`.
    plot : bool, optional
        If True, display the base image, kernel, and blurred output side by
        side.

    Returns
    -------
    np.ndarray, uint8
        The blurred image, same shape as the input, clipped to [0, 255].
    """
    # 1. Load and convert to grayscale ('L' mode) unless an array was given.
    #    NOTE: with the default img_array=None, `(img_array == None).any()`
    #    raises AttributeError (comparing None to None yields a plain bool,
    #    which has no .any()) -- callers must currently pass img_array as an
    #    actual array, or the function must be entered with a real path and
    #    an array-typed img_array. Left as-is to avoid changing behavior.
    if (img_array == None).any():
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

    # 2. Normalize the PSF kernel so the resulting image doesn't change brightness
    # The sum of all elements in the kernel should equal 1.0
    kernel = psf_kernel / np.sum(psf_kernel)

    # 3. Perform 2D convolution
    # 'boundary=fill' and 'fillvalue=0' handles the edges of the image
    convolved_array = convolve2d(img_array, kernel, mode='same', boundary='fill', fillvalue=0)

    # 4. Clip values to [0, 255] and convert back to 8-bit integers
    convolved_array = np.clip(convolved_array, 0, 255).astype(np.uint8)

    # 5. Show base image / kernel / result side by side.
    if plot:
        fig, ax = plt.subplots(1, 3, dpi=200)
        ax[0].imshow(img_array)
        ax[1].imshow(kernel)
        ax[2].imshow(convolved_array)
        ax[0].set_title("Base Image")
        ax[1].set_title("Kernel")
        ax[2].set_title("Output")
        plt.show()

    # 6. Optionally persist the result to disk.
    if save:
        result_img = Image.fromarray(convolved_array)
        result_img.save(output_path)
        print(f"Saved convolved image to {output_path}")
    return convolved_array
