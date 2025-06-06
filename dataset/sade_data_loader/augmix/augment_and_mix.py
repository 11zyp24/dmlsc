# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""

from ..augmix import augmentations as augmentations, augmentations_224

import numpy as np
from PIL import Image
import torch

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.


def augment_and_mix_pil(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * normalize(image) + m * mix
    return mixed


def aug_mix_torch(image, preprocess, aug_severity=3, mixture_width=3, mixture_depth=-1, alpha=1, all_ops=False):
    """Perform AugMix augmentations and compute mixture. Return Tensor.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
      aug_severity: Severity of underlying augmentation operators (between 1 to 10).
      mixture_width: Width of augmentation chain
      mixture_depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
      all_ops: Weather use all ops

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([alpha] * mixture_width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = np.random.randint(1, mixture_depth if mixture_depth > 0 else 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

def aug_mix_torch224(image, preprocess, aug_severity=3, mixture_width=3, mixture_depth=-1, alpha=1, all_ops=False):
    """Perform AugMix augmentations and compute mixture. Return Tensor.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
      aug_severity: Severity of underlying augmentation operators (between 1 to 10).
      mixture_width: Width of augmentation chain
      mixture_depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
      all_ops: Weather use all ops

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations_224.augmentations
    if all_ops:
        aug_list = augmentations_224.augmentations_all

    ws = np.float32(np.random.dirichlet([alpha] * mixture_width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = np.random.randint(1, mixture_depth if mixture_depth > 0 else 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


def aug_mix_cuda(image, preprocess, m, n, alpha=1, all_ops=False):
    """
    Use `mixture_depth` as the number of augmentations and `aug_severity` as the severity of augmentations,
    embedded into CUDA.

    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
        m: Augmentation severity
        n: Number of augmentations
        alpha: Probability coefficient for Beta and Dirichlet distributions.
        all_ops: Whether to use all augmentation operators

    Returns:
        mixed: Augmented and mixed image.
    """
    return aug_mix_torch(image, preprocess, aug_severity=m, mixture_width=3, mixture_depth=-1, alpha=alpha, all_ops=all_ops)
