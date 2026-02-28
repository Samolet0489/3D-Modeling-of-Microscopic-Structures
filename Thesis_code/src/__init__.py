"""Top-level package for synthetic microscopy dataset generation.

The modules exposed here are:

* ``shapes`` – volume and mask creation
* ``psf`` – point-spread function utilities
* ``render`` – rendering and projection routines
* ``noise`` – noise models
* ``dataset_generator`` – dataset building and CLI
* ``demo_pipeline`` – small demonstration script
"""

# expose frequently used symbols for convenience
from .shapes import generate_mito_volume, create_ellipsoid_volume
from .psf import make_gaussian_psf, gaussian_psf_kernel
from .render import render_image, project_volume, get_middle_slice
from .noise import add_noise, add_microscopy_noise

__all__ = [
    "generate_mito_volume",
    "create_ellipsoid_volume",
    "make_gaussian_psf",
    "gaussian_psf_kernel",
    "render_image",
    "project_volume",
    "get_middle_slice",
    "add_noise",
    "add_microscopy_noise",
    "dataset_generator",
    "demo_pipeline",
]
