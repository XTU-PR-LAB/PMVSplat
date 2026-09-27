from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor
import torch.nn.functional as F
from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        upscale_factor: int,
        global_step: int,
    ) -> Float[Tensor, ""]:
        _, _, _, h, w = batch["target"]["image"].shape
        target_gt = batch["target"]["image"]
        scale = upscale_factor / 4 
        new_h = int(h*scale)
        new_w = int(w*scale)
        new_target_gt = F.interpolate(rearrange(target_gt, "b v c h w -> (b v) c h w"), size=(new_h, new_w), mode="bilinear", align_corners=False)
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=new_target_gt.device)

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            new_target_gt,
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
