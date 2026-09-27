from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
import torch.nn.functional as F
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .backbone import (
    FeatureNet,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
from typing import Tuple

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_cto_FR:bool
    wo_cvfm:bool
    wo_KD:bool
    use_epipolar_trans: bool


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=False,
            use_epipolar_trans=False,
        )
        self.feature_net = FeatureNet()
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not False
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=cfg.costvolume_unet_channel_mult,
            costvolume_unet_attn_res=cfg.costvolume_unet_attn_res,
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_cto_FR=cfg.wo_cto_FR,
            wo_cvfm=cfg.wo_cvfm,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Tuple[List[Gaussians],torch.Tensor]:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )
        feats = self.forward_feat(context["image"]) 
        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        # kl loss
        features_0 = feats['level_0'] 
        features_1 = feats['level_1'] 
        features_2 = feats['level_2'] 
        loss_kl = kl_loss_ms(features_0, features_1, features_2)
        depths_list, densities_list, raw_gaussians_list = self.depth_predictor(
            in_feats,
            feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )
        j = 0
        gaussians_list = []
        for upscale_factor in ([4] if self.cfg.wo_cto_FR else [1, 2, 4]):
            scale = upscale_factor / 4 
            new_h = int(h*scale)
            new_w = int(w*scale)
            # Convert the features and depths into Gaussians.
            xy_ray, _ = sample_image_grid((new_h, new_w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians_list[j],
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((new_h, new_w), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.cfg.gaussians_per_pixel
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths_list[j],
                self.map_pdf_to_opacity(densities_list[j], global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (new_h, new_w),
            )
            gaussians_list.append(gaussians)
            j = j + 1


        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1
        gaussians_list_out = [
                Gaussians(
                    means=rearrange(
                        gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    covariances=rearrange(
                        gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    harmonics=rearrange(
                        gaussians.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    opacities=rearrange(
                        opacity_multiplier * gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                    )
                )
                for gaussians in gaussians_list
        ]
        return gaussians_list_out,loss_kl

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
def kl_loss_ms(feat64, feat128, feat256, eps=1e-8): 
    def to_prob(feat): 
        B, V, C, H, W = feat.shape 
        pooled = F.adaptive_avg_pool2d(feat.view(B*V, C, H, W), 1) 
        pooled = pooled.view(B, V, C) 
        return F.log_softmax(pooled, dim=2)
        
    p64 = to_prob(feat64) 
    p128 = to_prob(feat128) 
    p256 = to_prob(feat256) 
    
    loss = ( F.kl_div(p128, p64.detach().exp(), reduction="batchmean") + 
            F.kl_div(p256, p128.detach().exp(), reduction="batchmean") ) 
    
    return loss
