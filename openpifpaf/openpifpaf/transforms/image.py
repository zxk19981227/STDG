import io
import logging

import numpy as np
import PIL
import scipy
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class ImageTransform(Preprocess):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta,depth):
        image = self.image_transform(image)
        depth=self.image_transform(depth)
        return image, anns, meta,depth


class JpegCompression(Preprocess):
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta,depth):
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        f2 = io.BytesIO()
        image.save(f2, 'jpeg', quality=self.quality)
        return PIL.Image.open(f), anns, meta,PIL.Image.open(f2)


class Blur(Preprocess):
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta,depth):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        depth_np= np.asarray(depth)
        depth_np = scipy.ndimage.filters.gaussian_filter(depth_np, sigma=(sigma, sigma, 0))
        return PIL.Image.fromarray(im_np), anns, meta,PIL.Image.fromarray(depth_np)
