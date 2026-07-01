"""Fixed survey-specific configurations for Test Set 1 / Test Set 2.

Approximate optical/instrumental characteristics (single representative band):
  scale   pixel scale (arcsec/px)
  fwhm    Gaussian PSF FWHM (arcsec)
These drive Test Set 1 (cross-survey generalization, no noise).

Test Set 2 additionally applies realistic per-survey noise. Two more numbers
per survey, expressed in the image's surface-brightness units (the same units
as `g`/`f_gt` from the simulator, whose characteristic arc brightness is
~4e-3 ... 1.3e-2 -- see sim_lens):
  bkg_rms gaussian background RMS per native pixel (sky + read noise floor)
  gain    counts per unit surface brightness; sets the Poisson shot term
          (variance = signal / gain). Larger gain -> deeper data -> less shot
          noise.
The numbers encode the usual ordering: space surveys (HST/Roman/Euclid) are
deep and low-background; ground surveys (Rubin/DES) are shallower with a
brighter sky, so their per-pixel SNR is lower -- on top of the heavier PSF
blur that already spreads the arc light. These are nominal, calibrated to give
realistic per-pixel SNR in the simulator's units, not exhaustive instrument
noise models. Tune per survey here; nothing else needs to change.
"""
SURVEYS = {
    "HST":    dict(scale=0.05,  fwhm=0.10, bkg_rms=0.0008, gain=3.0e4),  # ACS/WFC drizzled, sharp, deep
    "Euclid": dict(scale=0.10,  fwhm=0.16, bkg_rms=0.0011, gain=2.0e4),  # VIS
    "Roman":  dict(scale=0.11,  fwhm=0.12, bkg_rms=0.0009, gain=2.5e4),  # WFI
    "Rubin":  dict(scale=0.20,  fwhm=0.70, bkg_rms=0.0016, gain=8.0e3),  # LSST, seeing-limited, single-epoch-ish
    "DES":    dict(scale=0.263, fwhm=0.90, bkg_rms=0.0020, gain=6.0e3),  # DECam, seeing-limited, shallow
}

# Keys consumed as observational config when rendering (so simulate_one ignores
# the noise keys -- they are applied AFTER rendering by add_noise()).
RENDER_KEYS = ("scale", "fwhm")
NOISE_KEYS = ("bkg_rms", "gain")
