import random
from typing import Any, Dict, Optional, Tuple, cast

from kornia import augmentation as kna
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound
from kornia.enhance.adjust import adjust_contrast
from torch import Tensor

from .augment_base import Augment


# copied from https://github.com/kornia/kornia/blob/53808e5fd039a4de97fcb207b7643fd72c30f032/kornia/augmentation/_2d/intensity/contrast.py
# because still not on pypi
class RandomContrast(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the contrast of a tensor image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.

    .. image:: _static/img/RandomContrast.png

    Args:
        p: probability of applying the transformation.
        contrast: the contrast factor to apply
        clip_output: if true clip output
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_contrast

        Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomContrast(contrast = (0.5,2.),p=1.)
        >>> aug(inputs)
        tensor([[[[0.2750, 0.4258, 0.0490],
                  [0.0732, 0.1704, 0.3514],
                  [0.2716, 0.4969, 0.2525]],
        <BLANKLINE>
                 [[0.3505, 0.1934, 0.2227],
                  [0.0124, 0.0936, 0.1629],
                  [0.2874, 0.3867, 0.4434]],
        <BLANKLINE>
                 [[0.0893, 0.1564, 0.3778],
                  [0.5072, 0.2201, 0.4845],
                  [0.2325, 0.3064, 0.5281]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomContrast((0.8,1.2), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        contrast: Tuple[float, float] = (1.0, 1.0),
        clip_output: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self.contrast: Tensor = _range_bound(contrast, 'contrast', center=1.0)
        self._param_generator = cast(
            rg.PlainUniformGenerator, rg.PlainUniformGenerator((self.contrast, "contrast_factor", None, None))
        )
        self.clip_output = clip_output

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        contrast_factor = params["contrast_factor"].to(input)
        return adjust_contrast(input, contrast_factor, self.clip_output)


class AugmentPreTraining(Augment):
    def __init__(self, imgH, imgW) -> None:
        super().__init__()
        self.transforms = kna.AugmentationSequential(
            RandomContrast(contrast=(.7, 1.),p=1.),
            kna.RandomGaussianBlur((11, 11), (0.5, 1.5), p=1.),
            kna.AugmentationSequential(
                kna.RandomCrop(size=(int(imgH*random.uniform(.6, 1.)),
                                     imgW),
                               p=1.,),
                kna.Resize((imgH, imgW))
                ),
            kna.AugmentationSequential(
                kna.RandomCrop(size=(imgH,
                                     int(imgW*random.uniform(.96, 1.))),
                               p=1.,),
                kna.Resize((imgH, imgW))
                ),
            kna.RandomSharpness((4, 4.5), p=1.),
            kna.RandomThinPlateSpline(scale=random.uniform(0.02, 0.03),
                                      p=1.),
            kna.RandomPerspective(random.uniform(0.01, 0.02),
                                  p=1., keepdim=True),
            random_apply=(1, 5),
            same_on_batch=False,
        )


class AugmentTraining(Augment):
    def __init__(self, imgH, imgW) -> None:
        super().__init__()
        self.transforms = kna.AugmentationSequential(
            utils.augment.RandomContrast(contrast=(.7, 1.),p=1.),
            kna.RandomGaussianBlur((11, 11), (0.5, 1.5), p=1.),
            kna.AugmentationSequential(
                kna.RandomCrop(size=(int(imgH*random.uniform(.8, 1.)),
                                     imgW),
                               p=1.,),
                kna.Resize((imgH, imgW))
                ),
            kna.AugmentationSequential(
                kna.RandomCrop(size=(imgH,
                                     int(imgW*random.uniform(.98, 1.))),
                               p=1.,),
                kna.Resize((imgH, imgW))
                ),
            random_apply=True,
            same_on_batch=False,
        )
