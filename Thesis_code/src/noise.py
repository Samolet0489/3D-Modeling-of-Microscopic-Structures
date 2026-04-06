import numpy as np
from scipy.ndimage import gaussian_filter


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


def _make_structured_background(
    shape,
    rng,
    background_strength=0.08,
    band_strength=0.025,
    gradient_strength=0.03,
):
    """
    Create a weak, structured microscopy-like background.

    Components:
    - low-frequency cloudy background
    - column-wise banding / striping
    - weak global illumination gradient
    """
    h, w = shape

    # low-frequency cloudy field
    field = rng.normal(loc=0.0, scale=1.0, size=(h, w)).astype(np.float32)
    field_sigma = float(rng.uniform(6.0, 14.0))
    field = gaussian_filter(field, sigma=field_sigma)
    field -= field.mean()
    field /= max(field.std(), 1e-6)

    # soft column-wise banding
    cols = rng.normal(loc=0.0, scale=1.0, size=w).astype(np.float32)
    cols = gaussian_filter(cols, sigma=max(3.0, w / 24.0))
    cols -= cols.mean()
    cols /= max(cols.std(), 1e-6)
    band_img = np.tile(cols[None, :], (h, 1))

    # weak gradient with random direction
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / max(w - 1, 1)) - 0.5
    yy = (yy / max(h - 1, 1)) - 0.5
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    grad = np.cos(angle) * xx + np.sin(angle) * yy
    grad /= max(np.abs(grad).max(), 1e-6)

    bg = (
        float(background_strength) * field
        + float(band_strength) * band_img
        + float(gradient_strength) * grad
    )
    return bg.astype(np.float32)


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
    3) optional Gaussian noise
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
    """
    rng = np.random.default_rng(seed)

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
    background_strength=0.08,
    band_strength=0.025,
    gradient_strength=0.03,
    return_uint8=False,
):
    """
    Helper matching the API used by demo_pipeline.

    This version keeps the supervisor-style Poisson noise but can also add a
    structured background before noise, which helps make the background look
    more like real microscopy.

    Compatibility behavior:
    - still returns float [0,1] by default
    - old calls still work
    - if photon_level is in [1,255], it is used as a fixed max signal
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

    img = image.astype(np.float32).copy()

    # Add structured background before Poisson / Gaussian noise.
    if background_strength > 0 or band_strength > 0 or gradient_strength > 0:
        bg = _make_structured_background(
            img.shape,
            rng,
            background_strength=background_strength,
            band_strength=band_strength,
            gradient_strength=gradient_strength,
        )
        img = np.clip(img + bg, 0.0, None)

    return _supervisor_style_noise(
        image=img,
        rng=rng,
        apply_gaussian=apply_gaussian,
        std=std,
        offset_range=offset_range,
        max_signal_range=max_signal_range,
        return_uint8=return_uint8,
    )
