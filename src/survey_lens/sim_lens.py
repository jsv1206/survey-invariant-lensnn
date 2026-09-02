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
    # observational.
    # Bounds are deliberately WIDER than the survey configs in surveys.py so that
    # every target survey is INTERIOR to the training prior, not on its edge:
    #   HST  (fwhm 0.10, s 0.05 ) was on the lower edge of logU(0.1 , 0.9 )/logU(0.05, 0.30)
    #   DES  (fwhm 0.90, s 0.263) was on the upper edge of logU(0.1 , 0.9 )
    # One-sided training support at the extremes makes a prior-coverage artifact
    # look like a cross-survey generalization failure. See the project notes (bug 2).
    fwhm = _loguniform(rng, 0.08, 1.1)
    scale = _loguniform(rng, 0.04, 0.32)
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
    """Sum-pool a fine image by integer factor (flux-conserving area resampling).

    lenstronomy's ImageModel.image() returns flux PER PIXEL, not surface
    brightness, so collapsing `factor**2` fine pixels into one native pixel
    must SUM their flux, not average it -- averaging divides brightness by
    `factor**2` and, since the supersampling factor is chosen adaptively per
    sample (see choose_supersample_factor), that turns into a per-sample,
    scale-correlated amplitude error. Do NOT apply this to the PSF stamp `h`, which must stay
    sum-normalized to 1.
    """
    n = img_fine.shape[0]
    m = n // factor
    img = img_fine[:m * factor, :m * factor]
    return img.reshape(m, factor, m, factor).sum(axis=(1, 3))


def choose_supersample_factor(s_target, fwhm, factor_min=4, sigma_px_target=3.0):
    """Integer supersampling factor for the ADAPTIVE fine grid (fine = s/factor).

    Chosen per sample so that
      (a) the sharp lens image is rendered on a grid at least `factor_min` times
          finer than native (sub-pixel structure of the arc is resolved), and
      (b) the PSF is resolved on the fine grid: sigma/fine >= `sigma_px_target`,
          i.e. factor >= sigma_px_target * s / sigma. A narrow PSF at a coarse
          native scale therefore gets a deeper supersampling.

    Because `fine = s / factor` exactly divides the requested scale, the realized
    native scale equals the requested one with NO quantization error -- unlike a
    single global `fine_scale`, which rounds s to a multiple of itself (that is
    dataset bug 1: it collapsed Roman 0.11" onto Euclid's 0.10" grid).
    """
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return int(max(factor_min, np.ceil(sigma_px_target * float(s_target) / sigma)))


def simulate_one(params, fov=20.0, fine_scale=None, return_fine=False,
                 factor_min=4, sigma_px_target=3.0, contam=None):
    """
    Produce (f_gt, g, h, s, theta_E, params) at the native pixel scale.

    Convolve-then-resample: render sharp on a fine grid, convolve with the
    Gaussian PSF at fine scale, then area-resample to the native scale.

    `fine_scale=None` (DEFAULT, recommended) selects the fine grid ADAPTIVELY per
    sample as `s / choose_supersample_factor(s, fwhm)`. This renders the
    requested pixel scale EXACTLY (no quantization) and holds the render/PSF
    sampling quality roughly constant across surveys, at ~2x the cost of the old
    global 0.025" grid. Passing an explicit float reproduces the legacy
    behaviour, where the realized scale is `fine_scale * round(s/fine_scale)`
    and can differ from the requested `s`.
    """
    from scipy.ndimage import gaussian_filter

    s_target = params['scale']
    if fine_scale is None:                      # adaptive: exact scale fidelity
        factor = choose_supersample_factor(s_target, params['fwhm'],
                                           factor_min, sigma_px_target)
        fine_scale = s_target / factor
        s = float(s_target)
    else:                                       # legacy: global fine grid
        factor = max(1, int(round(s_target / fine_scale)))
        s = fine_scale * factor                 # realized native scale
    npix_native = int(round(fov / s))
    npix_fine = npix_native * factor

    sharp_fine = _render_sharp(params, fine_scale, npix_fine)

    # Unmodelled astrophysics (domain-shift test sets), added BEFORE the PSF
    # convolution so it is blurred and resampled by exactly the same path as the
    # arc -- and so it lands in BOTH f_gt and g. Injecting into g alone would be
    # degenerate: `model.py` computes recon = g + delta, a residual on the input,
    # so a g-only contaminant appears in the output by construction and scoring
    # it against a clean f_gt would measure the deflector's brightness rather
    # than anything about the reconstruction.
    if contam is not None:
        sharp_fine = sharp_fine + render_contaminants_scaled(
            contam, fine_scale, npix_fine, float(sharp_fine.max()))

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


# ---------------------------------------------------------------------------
# Realistic (correlated) noise for TRAINING, and unmodelled astrophysical
# components for the domain-shift evaluation.
#
# Noise parameters are drawn by `sample_noise` from a SEPARATE generator, never
# from the one `sample_params` uses. That is not a style choice: `rng` is
# sequential and `build_test_set_1` draws 200 lenses from ONE generator in a
# loop, so adding draws inside `sample_params` -- even appended at the very end
# -- shifts every sample after the first and silently changes the lens
# population for a fixed seed. Measured: all 14 lens fields moved. Every
# published Run 3 number would have stopped being reproducible from its seed
# with no file changed. `tests/test_param_draw_order.py` pins this.
# ---------------------------------------------------------------------------

def sample_noise(rng):
    """Per-sample noise realism parameters, drawn from their OWN generator.

    Bounds bracket all five surveys so each sits INTERIOR rather than on an
    edge (bkg_rms 18.7-78.2 pctile, gain 17.6-87.5) -- the same rule that fixed
    dataset bug 2, where HST and DES on the fwhm/scale edges made a
    prior-coverage artifact look like a cross-survey generalization failure.
    """
    return dict(bkg_rms=_loguniform(rng, 6.0e-4, 2.8e-3),
                gain=_loguniform(rng, 4.0e3, 4.0e4),
                corr_px=float(rng.uniform(0.5, 1.5)))


def _corr_kernel_1d(corr_px):
    """Normalized 1D Gaussian; the 2D kernel is its outer product with itself."""
    r = max(1, int(np.ceil(4.0 * corr_px)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / corr_px) ** 2)
    return k / k.sum()


def add_correlated_noise(g, bkg_rms, gain, rng, corr_px=1.0):
    """Noise with the SAME per-pixel variance as `add_noise` but a non-zero
    correlation length.

    `add_noise`'s docstring records that correlated-noise-from-resampling is
    "a second order effect we omit deliberately". Real detector noise is
    correlated by drizzling and coaddition, so training can include it: that closes
    the realism gap rather than leaving it as a test-time surprise, and removes
    the obvious referee objection ("you can simulate that trivially -- why not
    train on it?").

    Per-pixel variance is PRESERVED exactly. Convolving unit-variance white
    noise with a sum-1 kernel `k` reduces the variance to `sum(k**2)`, so the
    result is divided by `sqrt(sum(k**2))` to restore it. Without that step a
    correlation-length change would also change the noise AMPLITUDE, and any
    measured effect would confound structure with depth.

    Approximation, stated: the correlation is imposed on unit-variance noise
    which is then scaled by the per-pixel sigma, so for signal-dependent noise
    the correlation is applied before rather than after scaling. Exact for the
    background-dominated regime that sets the floor in these images.
    """
    from scipy.ndimage import convolve1d
    var = np.maximum(g, 0.0) / gain + bkg_rms ** 2
    n = rng.normal(0.0, 1.0, size=g.shape)
    if corr_px and corr_px > 0:
        k1 = _corr_kernel_1d(corr_px)
        n = convolve1d(n, k1, axis=0, mode="reflect")
        n = convolve1d(n, k1, axis=1, mode="reflect")
        # separable => sum(k_2d**2) == sum(k1**2)**2, so sqrt of it is sum(k1**2)
        n = n / (k1 ** 2).sum()
    # float32 to match `add_noise`'s contract. numpy promotes float32 `g` to
    # float64 through the arithmetic above, and `data._to_tensor` does not cast,
    # so returning float64 reaches the model as a Double tensor and raises
    # "mat1 and mat2 must have the same dtype". The training path happened to
    # cast at its own call site and was unaffected; the test-set builder was not.
    return (g + n * np.sqrt(var)).astype(np.float32)


def sample_contaminants(params, rng, n_companion_max=3):
    """Unmodelled astrophysical components, drawn from their OWN generator.

    These are the DOMAIN SHIFT test sets, and they are chosen because the
    simulator genuinely omits them rather than because they are hard to add:
    `_render_sharp` passes `kwargs_lens_light=None`, so every training image is
    a lensed source on EMPTY SKY. Real strong lenses have the deflector galaxy
    at the centre, usually brighter than the arc, sitting directly on the ring
    theta_E is measured from.

    Amplitudes are set RELATIVE to the source amplitude, so the contrast is
    controlled rather than left to depend on the arbitrary flux normalization.
    The deflector-to-arc brightness ratio is the biggest judgement call here and
    must be stated in the paper: real ratios vary enormously and logU(1, 10)
    spans "comparable" to "an order of magnitude brighter".
    """
    # `peak_ratio` is the target PEAK brightness relative to the clean arc's
    # peak, resolved at render time. It is NOT an `amp` multiplier: lenstronomy's
    # Sersic `amp` is a surface-brightness normalization whose relation to the
    # peak depends steeply on n_sersic and R_sersic, so a shared `amp` between an
    # n=4 deflector at R~theta_E and an n~2 source at R~0.2" differs by ~1e5 in
    # peak. Measured that the wrong way round first.
    deflector = dict(peak_ratio=float(_loguniform(rng, 1.0, 10.0)), amp=1.0,
                     # de Vaucouleurs: ellipticals are n~4, and the effective
                     # radius of a deflector is comparable to its Einstein radius
                     R_sersic=float(params["theta_E"] * rng.uniform(0.7, 1.5)),
                     n_sersic=4.0,
                     e1=float(params["e1"] * rng.uniform(0.5, 1.2)),
                     e2=float(params["e2"] * rng.uniform(0.5, 1.2)),
                     center_x=float(rng.normal(0.0, 0.05)),
                     center_y=float(rng.normal(0.0, 0.05)))
    companions = []
    for _ in range(int(rng.integers(1, n_companion_max + 1))):
        companions.append(dict(peak_ratio=float(rng.uniform(0.05, 0.30)), amp=1.0,
                               R_sersic=float(rng.uniform(0.15, 0.6)),
                               n_sersic=1.0,          # disk-like
                               e1=float(rng.normal(0, 0.15)),
                               e2=float(rng.normal(0, 0.15)),
                               center_x=float(rng.uniform(-4.0, 4.0)),
                               center_y=float(rng.uniform(-4.0, 4.0))))
    return dict(deflector=deflector, companions=companions)


def render_contaminants_scaled(contam, scale, npix, arc_peak):
    """Contaminant light on the same grid geometry as `_render_sharp`, scaled so
    each component's PEAK is `peak_ratio * arc_peak`.

    Rendered on the FINE grid and added to `sharp_fine` BEFORE the PSF
    convolution, so the contaminant is blurred and resampled by exactly the same
    path as the arc, and lands in BOTH f_gt and g. Adding it after resampling
    would give f_gt a sharp contaminant and g an unblurred one -- physically
    wrong and trivially detectable.

    Scaling is on the rendered PEAK, not on lenstronomy's `amp`: see
    `sample_contaminants`.
    """
    ra0 = dec0 = -(npix - 1) / 2.0 * scale
    yy, xx = np.mgrid[0:npix, 0:npix].astype(float)
    x = ra0 + xx * scale
    y = dec0 + yy * scale
    model = LightModel(['SERSIC_ELLIPSE'])
    out = np.zeros((npix, npix), dtype=float)
    for kw in [contam["deflector"]] + list(contam["companions"]):
        target = kw["peak_ratio"] * arc_peak
        kwl = {k: v for k, v in kw.items() if k != "peak_ratio"}
        comp = model.surface_brightness(x, y, [kwl])
        pk = float(comp.max())
        if pk > 0:
            comp = comp * (target / pk)
        out += comp
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


def simulate_one_noisy(params, fov=20.0, fine_scale=None, rng=None):
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


def simulate_dataset(n, rng, fov=20.0, fine_scale=None):
    """Generator-style list of samples (variable image sizes per sample)."""
    samples = []
    for _ in range(n):
        p = sample_params(rng)
        samples.append(simulate_one(p, fov=fov, fine_scale=fine_scale))
    return samples
