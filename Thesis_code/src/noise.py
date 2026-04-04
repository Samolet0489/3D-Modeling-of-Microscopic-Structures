import numpy as np


def add_poisson_noise(image, peak_photons=5000, seed=None):
    """
    Legacy helper:
    Add Poisson (shot) noise and return float image in [0, 1].

    image: 2D float array, arbitrary positive values.
    peak_photons: max photons at brightest pixel.
    """
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32)
    img = img - img.min()

    if img.max() > 0:
        img = img / img.max()

    img = img * peak_photons
    noisy_photons = rng.poisson(img).astype(np.float32)
    noisy = noisy_photons / float(peak_photons)
    return noisy


def add_gaussian_noise(image, sigma=0.03, seed=None):
    """
    Legacy helper:
    Add zero-mean Gaussian noise and clip to [0,1].

    sigma: standard deviation relative to [0,1] range.
    """
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0)


def _supervisor_style_noise(
    image,
    rng,
    apply_gaussian=False,
    std=15.0,
    offset_range=(40, 70),
    max_signal_range=(200, 240),
    return_uint8=False,
):
    """
    Supervisor-style noise:
    1) random offset / max signal normalization
    2) Poisson noise
    3) optional Gaussian noise (disabled by default)

    By default returns float32 in [0,1] to avoid breaking the rest of the project.
    Set return_uint8=True if you want the exact uint8-style behavior.
    """
    img = image.astype(np.float32)

    # make sure the input is non-negative before scaling
    img = img - img.min()
    peak = img.max()

    offset = int(rng.integers(offset_range[0], offset_range[1] + 1))
    max_sig = int(rng.integers(max_signal_range[0], max_signal_range[1] + 1))

    # avoid division by zero for empty / all-zero images
    if peak <= 0:
        img_normalized = np.full_like(img, fill_value=offset, dtype=np.float32)
    else:
        img_normalized = (max_sig - offset) * (img / peak) + offset

    img_noisy = rng.poisson(img_normalized).astype(np.float32)

    # Supervisor said to keep this off by default
    if apply_gaussian:
        img_noisy = rng.normal(loc=img_noisy, scale=std, size=img_noisy.shape).astype(np.float32)

    img_noisy = np.clip(img_noisy, 0, 255)

    if return_uint8:
        return img_noisy.astype(np.uint8)

    return (img_noisy / 255.0).astype(np.float32)


def add_microscopy_noise(
    image,
    peak_photons=5000,
    read_sigma=0.03,
    seed=None,
    apply_gaussian=False,
    std=15.0,
    return_uint8=False,
):
    """
    Main microscopy noise convenience function.

    Safe default:
    - uses supervisor-style Poisson noise
    - Gaussian is OFF by default
    - returns float [0,1] by default

    Notes:
    - peak_photons and read_sigma are kept mainly for backward compatibility
    - if apply_gaussian=True and std is left as default, read_sigma can still be used
    """
    rng = np.random.default_rng(seed)

    # backward-compatible mapping if someone explicitly wants to use old sigma style
    if apply_gaussian and std == 15.0 and read_sigma is not None:
        if read_sigma <= 1.0:
            std = float(read_sigma) * 255.0
        else:
            std = float(read_sigma)

    return _supervisor_style_noise(
        image=image,
        rng=rng,
        apply_gaussian=apply_gaussian,
        std=std,
        return_uint8=return_uint8,
    )


def add_noise(
    image,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    seed=None,
    apply_gaussian=False,
    std=15.0,
    return_uint8=False,
):
    """
    Helper matching the API used by demo_pipeline.

    This now uses the supervisor's preferred behavior:
    - random offset + random max signal normalization
    - Poisson noise
    - Gaussian OFF by default

    Compatibility behavior:
    - still returns float [0,1] by default
    - keeps old parameters so existing calls do not break
    - if photon_level is in [1,255], it is used as a fixed max signal
    - if apply_gaussian=True, read_noise_sigma can still influence std
    """
    rng = np.random.default_rng(seed)

    offset_range = (40, 70)
    max_signal_range = (200, 240)

    # Optional compatibility:
    # if caller passes something like photon_level=220, use it as fixed max signal
    if photon_level is not None and 1 <= photon_level <= 255:
        fixed_level = int(photon_level)
        max_signal_range = (fixed_level, fixed_level)

    # Optional compatibility for Gaussian if someone explicitly enables it
    if apply_gaussian and std == 15.0 and read_noise_sigma is not None:
        if read_noise_sigma <= 1.0:
            std = float(read_noise_sigma) * 255.0
        else:
            std = float(read_noise_sigma)

    return _supervisor_style_noise(
        image=image,
        rng=rng,
        apply_gaussian=apply_gaussian,
        std=std,
        offset_range=offset_range,
        max_signal_range=max_signal_range,
        return_uint8=return_uint8,
    )