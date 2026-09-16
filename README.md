# Fast PSF Synthesis with Defocus and Spherical Aberration

Reference implementation for **"Fast PSF Synthesis with Defocused and Spherical Aberration"** by Nicholas Ganino and Qi Guo (Elmore Family School of Electrical and Computer Engineering, Purdue University), an IEEE ICIP paper.

Accurately simulating a lens's point spread function (PSF) requires evaluating a diffraction (Hankel) integral that has no closed-form solution, and is normally computed with FFT or Hankel-transform numerics. This repo implements an **approximate closed-form solution** to that integral for defocus and spherical (Seidel) aberration, obtained by combining a piecewise Bessel-function approximation with Gaussian-type integrals. The result is a PSF simulator with `O(N)` complexity in radial resolution that is **~2x faster than Hankel-based integration and ~4x faster than FFT**, while closely matching full wave-optical PSFs — enabling large-scale, per-pixel depth-of-field (DoF) image synthesis.

An interactive version of the simulator is available at **https://hankel.qiguo.org**.

## Theory

For a radially symmetric pupil, the free-space diffraction integral reduces to a zeroth-order Hankel transform (a Fourier-Bessel transform) of the pupil function $P(r)$:

$$
h(k) = \left| \, 2\pi \int_0^{R} P(r)\, J_0(2\pi k r)\, r \, dr \, \right|^2
$$

where $r$ and $k$ are the radial coordinates in the aperture and sensor planes, $R$ is the aperture radius, and $J_0$ is the zeroth-order Bessel function of the first kind. Under defocus, the pupil phase is quadratic in $r$:

$$
P(r) = \exp\!\left(j\, 2 C_d \, r^2 / R^2\right)
$$

with $C_d$ the defocus coefficient. This integral has no closed form in general because of $J_0$ — so the key idea behind this repo is to replace $J_0$ with a piecewise closed-form approximation that is accurate over its whole domain:

$$
\tilde{J}_0(a) =
\begin{cases}
1 - \dfrac{a^2}{4} + \dfrac{a^4}{64}, & a \le 1 \\[6pt]
\sqrt{\dfrac{2}{\pi}}\left(\dfrac{3}{2}\alpha^{-1/2} - \dfrac{a^2}{2}\alpha^{-3/2} + \dfrac{1}{a}\right)\cos\!\left(a - \dfrac{\pi}{4}\right), & a > 1
\end{cases}
$$

where $\alpha$ is a tunable operating-point hyperparameter (the `TLC` argument threaded through `src/closed_form.py`). Substituting $\tilde{J}_0$ into the Hankel transform above turns it into a sum of Gaussian-type integrals with closed-form solutions in terms of the error function — this is the derivation implemented in `src/closed_form.py` and detailed further in the "How it works" section below.

## How it works

1. **Defocus only** (`src/closed_form.py`): The pupil phase is purely quadratic (`P(r) = exp(j·2·Cd·r²/R²)`). Substituting a piecewise rational/cosine approximation of the zeroth-order Bessel function `J0` into the Hankel integral reduces it to six Gaussian-type integrals with closed-form solutions (error function / imaginary error function).
2. **Defocus + spherical aberration** (`src/spherical.py` + `closed_form_spherical` in `src/closed_form.py`): The quartic spherical-aberration phase term (`Cs·r⁴/R⁴`) has no closed form on its own, so the pupil is partitioned into `M` radial rings, each locally fit with a quadratic phase (`spherical_approx`). Every ring then reduces to the defocus-only closed form (`closed_form_variable`), and the PSF is the coherent sum over rings.
3. **Baselines** for validation and benchmarking: brute-force 2D FFT of the pupil (`fft`/`create_pupil` in `scripts/main.py`), a dense-grid direct quadrature of the Hankel integral (`src/direct_quadrature.py`), and geometric (pillbox) PSF approximation.
4. **Per-ring quadratic coefficients (`alpha`)**: the quadratic-fit coefficient for each ring depends on `(Cd, Cs)` and is not itself closed-form; it was found by sweeping alpha per ring for a grid of `(Cd, Cs)` pairs. The results of that sweep live in `data/alpha_coefficients.csv` and are loaded via `src/alpha_table.py`.

## Repository layout

```
src/                      Core numerical library (no plotting / I/O side effects)
  closed_form.py            Closed-form defocus & spherical-aberration PSF solutions (Appendix derivation)
  closed_form_accelerated.py  Reduced-term "accelerated" variant of the closed-form solution
  direct_quadrature.py      Ground-truth Hankel integral via direct numerical quadrature
  spherical.py               Piecewise-quadratic phase approximation for spherical aberration
  alpha_table.py              Loads the pretrained per-ring alpha coefficients from
                              data/alpha_coefficients.csv
  apply_psf.py               Convolve an image with a PSF kernel and visualize/save the result

data/
  alpha_coefficients.csv    Pretrained per-ring alpha coefficients from an alpha sweep,
                              one row per (Cd, Cs) combination -- see src/alpha_table.py

scripts/                  Entry-point experiments (reproduce paper figures / run demos)
  main.py                    Core defocus & spherical comparisons vs. FFT/Hankel baselines,
                              runtime benchmarking, cross-correlation sweeps (Figs. 2-4)
  render_image.py            Per-pixel depth-of-field rendering over an RGB-D scene,
                              comparing closed-form / FFT / direct / pillbox kernels (Fig. 5),
                              plus the focus-breathing animation
                              (poster_figures/media/focus_breathing.gif); contains cell
                              markers (`# %%`) for interactive/notebook-style use
  website_figure_generators.py  Higher-DPI variants of the comparison figures (adds SSIM metric,
                              requires torch/torchmetrics), plus render_rendering_race_video()
                              which regenerates poster_figures/media/rendering_race.avi
                              (opt-in, requires opencv-python, not run automatically -- see Usage)

poster_figures/           Generated comparison plots (defocus, spherical, speed) and demo
                           media (poster_figures/media/: focus-breathing GIF, DoF-rendering
                           race video) -- the output of running the scripts/ above
paper_figures/            Curated, camera-ready figures as used in the paper
test_imgs/                Sample RGB-D (color + depth) frames used by scripts/render_image.py
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and NumPy 2.0+ (`direct_quadrature.py` uses `np.trapezoid`). `torch`/`torchmetrics` are only needed for `scripts/website_figure_generators.py`'s SSIM metric.

## Usage

All scripts import from `src/` and each other, so run them as modules **from the repository root**:

```bash
python -m scripts.main                        # Defocus/spherical accuracy comparisons
python -m scripts.render_image                 # DoF rendering demo + focus-breathing GIF over a sample RGB-D scene
python -m scripts.website_figure_generators     # High-DPI paper/website figure variants (needs torch)
```

`scripts/main.py`'s `__main__` block runs `main()`, which calls `compare_defocus` and `compare_spherical` by default. `compare_speed` (runtime benchmarking, Fig. 3a) and `compare_correlation` (cross-correlation sweep) are defined in the same file but not called from `main()` — invoke them directly, e.g.:

```bash
python -c "from scripts.main import compare_speed; compare_speed(plane_dim=20, R_max=1)"
```

To regenerate `poster_figures/media/rendering_race.avi` (a side-by-side race between the closed-form, FFT, and direct-quadrature renderers on a real scene), call the opt-in function explicitly — it is not run automatically since it takes several minutes and requires `opencv-python`:

```bash
python -c "from scripts.website_figure_generators import render_rendering_race_video; render_rendering_race_video()"
```

## Citation

If you use this code, please cite:

> Nicholas Ganino and Qi Guo, "Fast PSF Synthesis with Defocused and Spherical Aberration," IEEE International Conference on Image Processing (ICIP).

(Update with full proceedings/DOI details once available.)
