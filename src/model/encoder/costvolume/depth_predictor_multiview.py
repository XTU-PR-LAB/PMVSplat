import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..backbone.unimatch.vit_fpn import ViTFeaturePyramid
from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
import matplotlib.pyplot as plt

def warp_with_pose_depth_candidates(
    feature1,
    intrinsics_ref,
    intrinsics_src,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics_src.size(1) == intrinsics_src.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics_ref).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics_src, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]
        _, _, H_feat, W_feat = feature1.shape
        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (W_feat - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (H_feat - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr

class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=[1, 1, 1],
        costvolume_unet_attn_res=[],
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_cto_FR=False,
        wo_cvfm=False,
        wo_KD=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.costvolume_unet_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        self.wo_cto_FR = wo_cto_FR
        self.wo_cvfm = wo_cvfm
        self.mv_pyramid = ViTFeaturePyramid(
            in_channels=128, scale_factors=[2**i for i in range(3)]
        )
        self.corr_refine_net = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head_lowres = nn.ModuleList()
        for i in [0, 1, 2]:
            curr_num_depth_candidates = num_depth_candidates // (2**i)
            feature_mv_channels = 128 // (2**i)
            input_channels = curr_num_depth_candidates + feature_mv_channels
            channels =  self.costvolume_unet_feat_dim // (2**i)
            if upscale_factor > 1:
                costvolume_unet_channel_mult = costvolume_unet_channel_mult + [1]
                costvolume_unet_attn_res = [x * 2 for x in costvolume_unet_attn_res]
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]
            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )
            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))
            self.corr_refine_net.append(nn.Sequential(*modules))
            self.regressor_residual.append(nn.Conv2d(
                input_channels, channels, 1, 1, 0
            ))

            self.depth_head_lowres.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, channels, 3, 1, 1),
            ))

        proj_in_channels = feature_channels  + feature_channels  
        upsample_out_channels = feature_channels 
        self.upsampler_4 = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.upsampler_2 = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.upsampler_1 = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=1,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1 
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1, 
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels 
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        
        channels = depth_unet_feat_dim
        disps_models = [
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        ]
        self.to_disparity = nn.Sequential(*disps_models)

    def forward(
        self,
        features,
        feats,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        b, v, c, h, w = features.shape
        
        feat_comb_lists, _, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        features_mv = rearrange(
            features, "b v c h w -> (b v) c h w"
        ) 
        features_list_mv = self.mv_pyramid(features_mv)
        features_0 = feats['level_0'] 
        features_1 = feats['level_1'] 
        features_2 = feats['level_2'] 
        feat_comb_lists_64, intr_curr64, _, _ = (
            prepare_feat_proj_data_lists(
                features_0,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=32,
            )
        )
        feat_comb_lists_128, intr_curr128, _, _ = (
            prepare_feat_proj_data_lists(
                features_1,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=64,
            )
        )
        feat_comb_lists_256, intr_curr256, _, _ = (
            prepare_feat_proj_data_lists(
                features_2,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")
        depths_list = []
        densities_list = [] 
        raw_gaussians_list = []
        for scale_idx in [0, 1, 2]:
            scale = (2**scale_idx) / 4 
            new_h = int(256*scale)
            new_w = int(256*scale)
            new_extra_info = F.interpolate(extra_info['images'], size=(new_h, new_w), mode="bilinear", align_corners=False)
            if scale_idx == 2:
                intr_curr = intr_curr256
                feat01 = feat_comb_lists_256[0]
                upsampler = self.upsampler_4
            elif scale_idx == 1:
                intr_curr = intr_curr128
                feat01 = feat_comb_lists_128[0]
                upsampler = self.upsampler_2
            elif scale_idx == 0:
                intr_curr = intr_curr64
                feat01 = feat_comb_lists_64[0]
                upsampler = self.upsampler_1
            if scale_idx > 0:
                num_depth_candidates = self.num_depth_candidates // (2**scale_idx)
                assert fine_disps is not None
                fine_disps = F.interpolate(
                    fine_disps, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()
                min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
                max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  
                depth_interval = depth_interval.view(-1, 1, 1, 1)
                depth_range_min = (
                    fine_disps - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    fine_disps + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))
                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features)
                    .view(1, num_depth_candidates, 1, 1)
                )  
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  
                depth_candidates_curr = depth_candidates
            else:
                depth_candidates_curr = disp_candi_curr.repeat([1, 1, *feat01.shape[-2:]])
            feat = feat_comb_lists[0]
            raw_correlation_in_lists = []
            for feat10_64, feat10_128, feat10_256, pose_curr in zip(feat_comb_lists_64[1:], feat_comb_lists_128[1:], feat_comb_lists_256[1:], pose_curr_lists):
                feat01_warped_64 = warp_with_pose_depth_candidates(
                    feat10_64,
                    intr_curr,
                    intr_curr64,
                    pose_curr,
                    1.0 / depth_candidates_curr,
                    warp_padding_mode="zeros",
                )  
                feat01_warped_128 = warp_with_pose_depth_candidates(
                    feat10_128,
                    intr_curr,
                    intr_curr128,
                    pose_curr,
                    1.0 / depth_candidates_curr,
                    warp_padding_mode="zeros",
                )  
                feat01_warped_256 = warp_with_pose_depth_candidates(
                    feat10_256,
                    intr_curr,
                    intr_curr256,
                    pose_curr,
                    1.0 / depth_candidates_curr, 
                    warp_padding_mode="zeros",
                )  
                raw_correlation_in_64 = (feat01.unsqueeze(2) * feat01_warped_64).sum(
                    1
                ) / (
                    feat01.shape[1]**0.5
                )  
                raw_correlation_in_128 = (feat01.unsqueeze(2) * feat01_warped_128).sum(
                    1
                ) / (
                    feat01.shape[1]**0.5
                )
                raw_correlation_in_256 = (feat01.unsqueeze(2) * feat01_warped_256).sum(
                    1
                ) / (
                    feat01.shape[1]**0.5
                )
                del feat01_warped_64,feat01_warped_128,feat01_warped_256
                raw_correlation_in = torch.stack(
                    [raw_correlation_in_64, raw_correlation_in_128, raw_correlation_in_256], dim=0
                ) 
                raw_correlation_in = raw_correlation_in.max(dim=0)[0]  
                raw_correlation_in_lists.append(raw_correlation_in)
            raw_correlation_in = torch.stack(raw_correlation_in_lists, dim=0).max(dim=0)[0] 
            
            raw_correlation_in = torch.cat((raw_correlation_in, features_list_mv[scale_idx]), dim=1)
            raw_correlation = self.corr_refine_net[scale_idx](raw_correlation_in)  
            raw_correlation = raw_correlation + self.regressor_residual[scale_idx](
                raw_correlation_in
            )
            pdf = F.softmax(
                self.depth_head_lowres[scale_idx](raw_correlation), dim=1
            )  
            if scale_idx == 0 or self.wo_cto_FR or self.wo_cvfm:
                depth_candidates = disp_candi_curr
            coarse_disps = (depth_candidates * pdf).sum(
                dim=1, keepdim=True
            )  
            pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  
            
            proj_feat_in_fullres = upsampler(torch.cat((feat, cnn_features), dim=1))
            proj_feature = self.proj_feature(proj_feat_in_fullres)
            refine_out = self.refine_unet(torch.cat(
                (new_extra_info, proj_feature, coarse_disps, pdf_max), dim=1
            ))
            raw_gaussians_in = [refine_out,
                                new_extra_info, proj_feat_in_fullres]
            raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
            raw_gaussians = self.to_gaussians(raw_gaussians_in)
            raw_gaussians = rearrange(
                raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
            )
            raw_gaussians_list.append(raw_gaussians)
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            densities_list.append(densities)
            fine_disps = (coarse_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            )
            depths = 1.0 / fine_disps 
            depths_out = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            depths_list.append(depths_out)
        
        return depths_list, densities_list, raw_gaussians_list
