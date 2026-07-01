"""
Strong-lens image simulation for the survey-invariant embedding project.

Physics / pipeline conventions
------------------------------
- Field of view is fixed in arcsec (default 20"). The lens center is the
  coordinate origin (0,0) and is the center of both the image and the PSF stamp.
- "Convolve-then-resample" is the physically correct order:
    1. render a SHARP lens image on a fine pixel grid,
    2. convolve with the (Gaussian) PSF on that fine grid,
    3. block/area-resample to the survey's native pixel scale s.
- Outputs per sample (all at the NATIVE pixel scale s, FOV x FOV arcsec):
    f_gt : sharp image           = resample(sharp_fine)              [deconv target]
    g    : blurred input image    = resample(conv(sharp_fine, psf))   [model input]
    h    : PSF stamp              = Gaussian(FWHM) sampled at scale s  [model input + reblur kernel]
    s    : pixel scale (arcsec/px)                                    [model input scalar]
    theta_E : Einstein radius (arcsec)                                [regression label]
  plus the full parameter dict.

Priors (from the project spec)
------------------------------
    theta_E   ~ U(0.5, 2.5)  arcsec
    e1,e2 (mass)   : ellipticity, |e| <~ 0.3
    gamma1,gamma2  : external shear, ~ N(0, 0.05)
    source: Sersic, random position near center, random ellipticity
    PSF FWHM  ~ logU(0.1, 0.9)  arcsec
    pixel s   ~ logU(0.05, 0.30) arcsec/px
"""
import numpy as np

# lenstronomy imports kept local to functions where practical, but the core
# image model is built once per call.
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(rng):
    """Draw one set of lens+source+PSF+scale parameters from the priors."""
    theta_E = float(rng.uniform(0.5, 2.5))
    # mass ellipticity (mild)
    phi = rng.uniform(0, np.pi)
    q = rng.uniform(0.7, 0.95)
    e1 = (1 - q) / (1 + q) * np.cos(2 * phi)
    e2 = (1 - q) / (1 + q) * np.sin(2 * phi)
    gamma1 = float(rng.normal(0, 0.05))
    gamma2 = float(rng.normal(0, 0.05))
    # source
    src_x = float(rng.normal(0, 0.15))
    src_y = float(rng.normal(0, 0.15))
    src_R = float(rng.uniform(0.1, 0.35))      # effective radius arcsec
    src_n = float(rng.uniform(1.0, 3.0))       # Sersic index
    sphi = rng.uniform(0, np.pi)
    sq = rng.uniform(0.6, 0.95)
    se1 = (1 - sq) / (1 + sq) * np.cos(2 * sphi)
    se2 = (1 - sq) / (1 + sq) * np.sin(2 * sphi)
    src_amp = float(rng.uniform(2.0, 6.0))
    # observational
    fwhm = _loguniform(rng, 0.1, 0.9)
    scale = _loguniform(rng, 0.05, 0.30)
    return dict(theta_E=theta_E, e1=e1, e2=e2, gamma1=gamma1, gamma2=gamma2,
                src_x=src_x, src_y=src_y, src_R=src_R, src_n=src_n,
                se1=se1, se2=se2, src_amp=src_amp, fwhm=fwhm, scale=scale)


def gaussian_psf_stamp(fwhm, scale, npix):
    """Analytic normalized Gaussian PSF on an npix x npix grid at given scale."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # arcsec
    sigma_pix = sigma / scale
    c = (npix - 1) / 2.0
    y, x = np.mgrid[0:npix, 0:npix].astype(float)
    r2 = (x - c) ** 2 + (y - c) ** 2
    psf = np.exp(-0.5 * r2 / sigma_pix ** 2)
    psf /= psf.sum()
    return psf


def _render_sharp(params, scale, npix):
    """Render the SHARP (un-convolved) lens image at a given pixel scale/size."""
    # coordinate grid centered at (0,0)
    ra_at_xy0 = -(npix - 1) / 2.0 * scale
    dec_at_xy0 = -(npix - 1) / 2.0 * scale
    transform = np.array([[scale, 0], [0, scale]])
    pixel_grid = PixelGrid(nx=npix, ny=npix,
                           transform_pix2angle=transform,
                           ra_at_xy_0=ra_at_xy0, dec_at_xy_0=dec_at_xy0)

    lens_model = LensModel(['EPL', 'SHEAR'])
    kwargs_lens = [
        {'theta_E': params['theta_E'], 'gamma': 2.0,
         'e1': params['e1'], 'e2': params['e2'], 'center_x': 0.0, 'center_y': 0.0},
        {'gamma1': params['gamma1'], 'gamma2': params['gamma2'], 'ra_0': 0, 'dec_0': 0},
    ]
    source_model = LightModel(['SERSIC_ELLIPSE'])
    kwargs_source = [
        {'amp': params['src_amp'], 'R_sersic': params['src_R'], 'n_sersic': params['src_n'],
         'e1': params['se1'], 'e2': params['se2'],
         'center_x': params['src_x'], 'center_y': params['src_y']},
    ]
    # No PSF here (sharp). Use a trivial PSF model.
    psf_none = PSF(psf_type='NONE')
    # supersampling handled by caller via npix/scale; here render directly.
    image_model = ImageModel(pixel_grid, psf_none, lens_model_class=lens_model,
                             source_model_class=source_model)
    img = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source,
                            kwargs_lens_light=None, kwargs_ps=None)
    return img


def _area_resample(img_fine, factor):
    """Average-pool a fine image by integer factor (area resampling / flux conserving mean)."""
    n = img_fine.shape[0]
    m = n // factor
    img = img_fine[:m * factor, :m * factor]
    return img.reshape(m, factor, m, factor).mean(axis=(1, 3))


def simulate_one(params, fov=20.0, fine_scale=0.025, return_fine=False):
    """
    Produce (f_gt, g, h, s, theta_E, params) at the native pixel scale.

    Convolve-then-resample: render sharp at `fine_scale`, convolve with Gaussian
    PSF at fine scale, then area-resample by the integer factor closest to
    native scale s. The realized native scale is fine_scale * factor.
    """
    from scipy.ndimage import gaussian_filter

    s_target = params['scale']
    factor = max(1, int(round(s_target / fine_scale)))
    s = fine_scale * factor                     # realized native scale
    npix_native = int(round(fov / s))
    npix_fine = npix_native * factor

    sharp_fine = _render_sharp(params, fine_scale, npix_fine)

    # PSF convolution at fine scale
    sigma_arcsec = params['fwhm'] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_fine_pix = sigma_arcsec / fine_scale
    blurred_fine = gaussian_filter(sharp_fine, sigma_fine_pix, mode='constant')

    # resample to native
    f_gt = _area_resample(sharp_fine, factor)
    g = _area_resample(blurred_fine, factor)

    # PSF stamp at native scale, same FOV/size
    h = gaussian_psf_stamp(params['fwhm'], s, npix_native)

    out = dict(f_gt=f_gt.astype(np.float32), g=g.astype(np.float32),
               h=h.astype(np.float32), s=np.float32(s),
               theta_E=np.float32(params['theta_E']), params=params,
               npix=npix_native)
    if return_fine:
        out['sharp_fine'] = sharp_fine
        out['blurred_fine'] = blurred_fine
    return out


def add_noise(g, bkg_rms, gain, rng):
    """Apply realistic per-pixel noise to a blurred image `g` (Test Set 2).

    Physically-structured noise = Poisson shot noise on the source signal plus
    a Gaussian background floor (sky + read noise):

        var(pixel) = max(signal, 0) / gain   +   bkg_rms**2

    - `gain` is counts per unit surface brightness; the shot variance of a
      pixel with brightness b is b/gain (deeper data -> larger gain -> less
      shot noise). We add it in the Gaussian approximation to Poisson, which is
      accurate in the counts regime relevant here and keeps everything
      differentiable-friendly and fast.
    - `bkg_rms` is the homogeneous background RMS per native pixel.

    Noise is signal-dependent, so it is applied at the NATIVE pixel scale,
    AFTER convolve-then-resample (correlated-noise-from-resampling is a second
    order effect we omit deliberately; the dominant per-survey difference is
    captured by gain/bkg_rms). Only the model INPUT is noised: the ground-truth
    sharp image f_gt and the PSF stamp h stay clean.

    Returns a new noised float32 array (does not modify `g`).
    """
    g = np.asarray(g, dtype=np.float64)
    var = np.maximum(g, 0.0) / float(gain) + float(bkg_rms) ** 2
    noisy = g + rng.normal(0.0, 1.0, size=g.shape) * np.sqrt(var)
    return noisy.astype(np.float32)


def simulate_one_noisy(params, fov=20.0, fine_scale=0.025, rng=None):
    """simulate_one + add_noise on the input, using params['bkg_rms','gain'].

    Convenience wrapper for Test Set 2. Requires `bkg_rms` and `gain` in params
    (e.g. from surveys.SURVEYS). Returns the same dict as simulate_one with:
      g        -> the NOISED input (what the model sees)
      g_clean  -> the noise-free blurred image (for reference / SNR diagnostics)
    f_gt and h are unchanged (clean).
    """
    if rng is None:
        rng = np.random.default_rng()
    o = simulate_one(params, fov=fov, fine_scale=fine_scale)
    o["g_clean"] = o["g"].copy()
    o["g"] = add_noise(o["g"], params["bkg_rms"], params["gain"], rng)
    o["bkg_rms"] = np.float32(params["bkg_rms"])
    o["gain"] = np.float32(params["gain"])
    return o


def simulate_dataset(n, rng, fov=20.0, fine_scale=0.025):
    """Generator-style list of samples (variable image sizes per sample)."""
    samples = []
    for _ in range(n):
        p = sample_params(rng)
        samples.append(simulate_one(p, fov=fov, fine_scale=fine_scale))
    return samples
