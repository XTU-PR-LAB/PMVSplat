from dataclasses import dataclass
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        upscale_factor: int,
        global_step: int,
    ) -> Float[Tensor, ""]:
        b, v, _, h, w = batch["target"]["image"].shape
        target_gt = batch["target"]["image"]
        scale = upscale_factor / 4 
        new_h = int(h*scale)
        new_w = int(w*scale)
        new_target_gt = F.interpolate(rearrange(target_gt, "b v c h w -> (b v) c h w"), size=(new_h, new_w), mode="bilinear", align_corners=False)
        new_target_gt = rearrange(new_target_gt, "(b v) c h w -> b v c h w",b=b,v=v)
        delta = prediction.color - new_target_gt
        return self.cfg.weight * (delta**2).mean()
