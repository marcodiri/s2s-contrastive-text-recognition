import torch
from kornia import image_to_tensor, tensor_to_image
from torch import Tensor, nn


class Augment(nn.Module):
    """
    This module applies `self.transforms` function to input `torch.Tensor` 
    and returns it.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.transforms = lambda img: img

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, img: torch.Tensor):
        img_ = img.detach().clone()
        img_.mul_(0.5).add_(0.5)
        img_aug = self.transforms(img_)
        img_aug.sub_(0.5).div_(0.5)
        assert img_aug.shape == img.shape, \
        "Augmented images should have the same shape as input images"
        return img_aug
        