import numpy as np


def add_poisson_noise(image, peak_photons=5000):
    """
    Add Poisson (shot) noise.

    image: 2D float array, arbitrary positive values.
    peak_photons: max photons at brightest pixel.
    """
    img = image.astype(np.float32)

    # normalize to [0, 1]
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    # scale to photon counts
    img = img * peak_photons

    noisy_photons = np.random.poisson(img).astype(np.float32)
    noisy = noisy_photons / peak_photons
    return noisy


def add_gaussian_noise(image, sigma=0.03):
    """
    Add zero-mean Gaussian noise and clip to [0,1].
    sigma: standard deviation relative to [0,1] range.
    """
    img = image.astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0)


def add_microscopy_noise(image, peak_photons=5000, read_sigma=0.03):
    """
    Convenience: Poisson shot noise + Gaussian read noise.
    """
    img = add_poisson_noise(image, peak_photons=peak_photons)
    img = add_gaussian_noise(img, sigma=read_sigma)
    return img


def add_noise(image, photon_level=2000.0, read_noise_sigma=0.03, seed=None):
    """Helper matching the API used by demo_pipeline.

    This is roughly equivalent to a Poisson shot noise with a given peak
    photon count followed by additive Gaussian noise.  The implementation
    uses :func:`add_poisson_noise` and :func:`add_gaussian_noise`.

    Parameters
    ----------
    image : ndarray
        2D float image assumed to be in an arbitrary positive range.
    photon_level : float
        Peak photon count for the Poisson step.
    read_noise_sigma : float
        Standard deviation for the Gaussian read noise (relative to [0,1]).
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    noisy : ndarray
        Noisy image clipped to [0,1].
    """
    if seed is not None:
        np.random.seed(seed)

    img = image.astype(np.float32)
    img = add_poisson_noise(img, peak_photons=photon_level)
    img = add_gaussian_noise(img, sigma=read_noise_sigma)
    return img