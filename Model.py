
from __future__ import annotations

import json
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial import cKDTree
from typing import Any


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_rgb01(rgb: torch.Tensor) -> torch.Tensor:
    rgb_min = float(rgb.min().detach().cpu())
    rgb_max = float(rgb.max().detach().cpu())

    if 0.0 <= rgb_min and rgb_max <= 1.0:
        return rgb

    if -1.0 <= rgb_min and rgb_max <= 1.0:
        return ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)

    mean = _IMAGENET_MEAN.to(rgb.device)
    std = _IMAGENET_STD.to(rgb.device)
    return (rgb * std + mean).clamp(0.0, 1.0)


def _ransac_point_inliers(
    pts: torch.Tensor,
    num_iters: int = 64,
    sample_size: int = 8,
    min_points: int = 12,
) -> torch.Tensor:
    N = pts.shape[0]
    if N < min_points:
        return torch.ones(N, dtype=torch.bool, device=pts.device)

    median_pt = pts.median(dim=0).values
    med_dist = (pts - median_pt).norm(dim=-1).median()
    radius = torch.clamp(med_dist * 2.0, min=1e-4)

    k = min(sample_size, N)
    best_count = -1
    best_center = median_pt

    for _ in range(num_iters):
        idx = torch.randint(0, N, (k,), device=pts.device)
        c = pts[idx].mean(dim=0)
        dists = (pts - c).norm(dim=-1)
        inliers = dists <= radius
        cnt = int(inliers.sum().item())
        if cnt > best_count:
            best_count = cnt
            best_center = c

    dists = (pts - best_center).norm(dim=-1)
    final = dists <= radius
    if int(final.sum().item()) < max(min_points, N // 10):
        return torch.ones(N, dtype=torch.bool, device=pts.device)
    return final



class SAM2Segmenter(nn.Module):

    def __init__(self, sam2_checkpoint: str, sam2_config: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.sam2 = self._load_sam2(sam2_checkpoint, sam2_config)
        self._mask_generator = None

    def _load_sam2(self, checkpoint: str, config: str):
        from sam2.build_sam import build_sam2
        model = build_sam2(config, checkpoint, device=self.device)
        return model

    @torch.no_grad()
    def _segment_single(self, rgbd: torch.Tensor):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        rgb = _to_rgb01(rgbd[:, :3, :, :])

        rgb_np = (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        all_masks, all_boxes = [], []
        if self._mask_generator is None:
            self._mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
        generator = self._mask_generator

        for img in rgb_np:
            result = generator.generate(img)
            if len(result) == 0:
                H, W = img.shape[:2]
                all_masks.append(np.zeros((0, H, W), dtype=bool))
                all_boxes.append(np.zeros((0, 4),    dtype=np.float32))
                continue

            masks = np.stack([r["segmentation"] for r in result])
            boxes = np.array([r["bbox"] for r in result], dtype=np.float32)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            keep = masks.reshape(masks.shape[0], -1).any(axis=1)
            masks = masks[keep]
            boxes = boxes[keep]

            if masks.shape[0] > 1:
                conf = np.array(
                    [float(r.get("predicted_iou", r.get("stability_score", 0.0))) for r in result],
                    dtype=np.float32,
                )[keep]
                keep_nms = self._mask_nms(masks, scores=conf, iou_thresh=0.8)
                masks = masks[keep_nms]
                boxes = boxes[keep_nms]

            all_masks.append(masks)
            all_boxes.append(boxes)

        return all_masks, all_boxes

    @staticmethod
    def _mask_nms(
        masks: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float = 0.8,
    ) -> np.ndarray:
        if masks.ndim != 3:
            raise ValueError(f"Expected masks (M,H,W), got {masks.shape}")
        if scores.ndim != 1 or scores.shape[0] != masks.shape[0]:
            raise ValueError(
                f"Expected scores shape ({masks.shape[0]},), got {scores.shape}"
            )

        order = np.argsort(-scores)
        flat = masks.reshape(masks.shape[0], -1).astype(np.bool_)
        areas = flat.sum(axis=1).astype(np.float32)

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            inter = np.logical_and(flat[i], flat[rest]).sum(axis=1).astype(np.float32)
            union = areas[i] + areas[rest] - inter
            iou = np.divide(inter, np.maximum(union, 1e-6), dtype=np.float32)
            order = rest[iou <= float(iou_thresh)]

        return np.array(keep, dtype=np.int64)

    def forward(
        self,
        scene_rgbd:  torch.Tensor,
        object_rgbd: torch.Tensor,
    ) -> dict[str, Any]:
        scene_masks,  scene_boxes  = self._segment_single(scene_rgbd)
        object_masks, object_boxes = self._segment_single(object_rgbd)

        return {
            "scene_masks"  : scene_masks,
            "scene_boxes"  : scene_boxes,
            "object_masks" : object_masks,
            "object_boxes" : object_boxes,
        }



_DINO_PATCH_SIZES = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "dinov2_vitg14": 14,
}

_DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

class DINOv2ObjectEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        out_dim: int | None = None,
    ):
        super().__init__()
        assert model_name in _DINO_DIMS, f"Unknown DINOv2 variant: {model_name}"

        self.model_name  = model_name
        self.patch_size  = _DINO_PATCH_SIZES[model_name]
        self.dino_dim    = _DINO_DIMS[model_name]
        self.out_dim     = out_dim or self.dino_dim

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            verbose=False,
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.proj = (
            nn.Linear(self.dino_dim, self.out_dim)
            if self.out_dim != self.dino_dim
            else nn.Identity()
        )


    def _normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        mean = _IMAGENET_MEAN.to(rgb.device)
        std  = _IMAGENET_STD.to(rgb.device)
        return (rgb - mean) / std

    def _get_patch_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        import contextlib
        B, _, H, W = rgb.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        ctx = torch.no_grad() if backbone_frozen else contextlib.nullcontext()

        with ctx:
            out = self.backbone.forward_features(self._normalize(rgb))

        patch_tokens = out["x_norm_patchtokens"]
        patch_tokens = patch_tokens.view(B, h_p, w_p, -1)
        return patch_tokens

    @staticmethod
    def _downsample_mask(
        mask: np.ndarray,
        h_p: int,
        w_p: int,
        pad_h: int = 0,
        pad_w: int = 0,
    ) -> torch.Tensor:
        if pad_h or pad_w:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h_p, w_p), mode="area")
        return t.squeeze(0).squeeze(0)


    def forward(
        self,
        rgbd:  torch.Tensor,
        masks: list,
        boxes: list,
    ) -> list[torch.Tensor]:
        rgb = _to_rgb01(rgbd[:, :3, :, :])
        B, _, H, W = rgb.shape

        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        H_pad = H + pad_h
        W_pad = W + pad_w
        h_p = H_pad // self.patch_size
        w_p = W_pad // self.patch_size

        patch_tokens = self._get_patch_tokens(rgb)

        result: list[torch.Tensor] = []

        for b in range(B):
            tokens_b  = patch_tokens[b]
            masks_b   = masks[b]
            M = len(masks_b)

            if M == 0:
                result.append(torch.zeros(0, self.out_dim, device=rgbd.device))
                continue

            obj_feats = []
            for m_idx in range(M):
                mask_fp = self._downsample_mask(masks_b[m_idx], h_p, w_p, pad_h=pad_h, pad_w=pad_w)
                mask_fp = mask_fp.to(rgbd.device)

                weight  = mask_fp.sum().clamp(min=1e-6)
                feat = (tokens_b * mask_fp.unsqueeze(-1)).sum(dim=(0, 1)) / weight
                obj_feats.append(feat)

            obj_feats_t = torch.stack(obj_feats, dim=0)
            obj_feats_t = self.proj(obj_feats_t)
            result.append(obj_feats_t)

        return result




def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src * src, dim=-1).view(B, N, 1)
    dist += torch.sum(dst * dst, dim=-1).view(B, 1, M)
    return dist


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def _query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    radius2 = radius * radius

    dist = _square_distance(new_xyz, xyz)
    dist_masked = dist.clone()
    dist_masked[dist_masked > radius2] = 1e10

    group_idx = dist_masked.topk(k=nsample, dim=-1, largest=False)[1]
    group_dist = dist_masked.gather(-1, group_idx)
    group_first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    invalid = group_dist >= 1e9
    group_idx = torch.where(invalid, group_first, group_idx)
    return group_idx


class _PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        S = self.npoint

        fps_idx = _farthest_point_sample(xyz, S)
        new_xyz = _index_points(xyz, fps_idx)

        idx = _query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = _index_points(xyz, idx)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = _index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 1, 2).contiguous()

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class _PointNetPPBackbone(nn.Module):
    def __init__(self, out_dim: int = 256, in_feat_dim: int = 3):
        super().__init__()
        self.sa1 = _PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=3 + in_feat_dim,
            mlp=[64, 64, 128],
        )
        self.sa2 = _PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=3 + 128,
            mlp=[128, 128, 256],
        )
        self.head = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(xyz, feat)
        _l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        global_feat = torch.max(l2_points, dim=1)[0]
        return self.head(global_feat)

class PointNetPPObjectEncoder(nn.Module):

    def __init__(
        self,
        out_dim: int = 256,
        n_points: int = 1024,
        use_rgb: bool = True,
        dense_n_points: int = 2048,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_points = n_points
        self.use_rgb = use_rgb
        self.dense_n_points = dense_n_points if dense_n_points > 0 else n_points
        in_feat_dim = 3 if use_rgb else 0
        self.backbone = _PointNetPPBackbone(out_dim=out_dim, in_feat_dim=in_feat_dim)

    def _points_from_mask(
        self,
        rgb01: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        dense: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        H, W = depth.shape
        valid = mask & (depth > 0) & torch.isfinite(depth)
        all_idx = valid.nonzero(as_tuple=False)
        if all_idx.numel() == 0:
            return None

        K = all_idx.shape[0]

        if K >= self.n_points:
            choice = torch.randperm(K, device=depth.device)[: self.n_points]
        else:
            choice = torch.randint(0, K, (self.n_points,), device=depth.device)
        idx = all_idx[choice]
        y = idx[:, 0].long()
        x = idx[:, 1].long()

        z = depth[y, x].float()

        if intrinsics is not None and intrinsics[0] > 0:
            fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        else:
            cx = 0.5 * (W - 1)
            cy = 0.5 * (H - 1)
            fx = float(max(W, 1))
            fy = float(max(H, 1))

        X = (x.float() - cx) * z / fx
        Y = (y.float() - cy) * z / fy
        xyz_raw = torch.stack([X, Y, z], dim=-1)

        xyz = xyz_raw - xyz_raw.mean(dim=0, keepdim=True)
        scale = torch.linalg.vector_norm(xyz, dim=-1).max().clamp(min=1e-6)
        xyz = xyz / scale

        if self.use_rgb:
            feat = rgb01[:, y, x].permute(1, 0).contiguous()
        else:
            feat = torch.empty((self.n_points, 0), device=depth.device, dtype=depth.dtype)

        if dense and self.dense_n_points != self.n_points:
            if K >= self.dense_n_points:
                choice_d = torch.randperm(K, device=depth.device)[: self.dense_n_points]
            else:
                choice_d = torch.randint(0, K, (self.dense_n_points,), device=depth.device)
            idx_d = all_idx[choice_d]
            y_d = idx_d[:, 0].long()
            x_d = idx_d[:, 1].long()
            z_d = depth[y_d, x_d].float()
            X_d = (x_d.float() - cx) * z_d / fx
            Y_d = (y_d.float() - cy) * z_d / fy
            xyz_raw_dense = torch.stack([X_d, Y_d, z_d], dim=-1)
        else:
            xyz_raw_dense = xyz_raw

        return xyz, feat, xyz_raw, xyz_raw_dense

    def forward(
        self,
        rgbd: torch.Tensor,
        masks: list[np.ndarray],
        boxes: list[np.ndarray] | None = None,
        return_points: bool = False,
        camera_intrinsics: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        del boxes
        B, C, H, W = rgbd.shape
        assert C >= 4, f"Expected rgbd with 4 channels, got {C}"
        assert len(masks) == B, f"masks list length {len(masks)} != batch {B}"

        rgb01 = _to_rgb01(rgbd[:, :3, :, :])
        depth = rgbd[:, 3, :, :]
        device = rgbd.device

        result: list[torch.Tensor] = []
        points_result: list[torch.Tensor] = []
        for b in range(B):
            masks_b = masks[b]
            intrinsics_b = camera_intrinsics[b] if camera_intrinsics is not None else None

            if masks_b.size == 0:
                result.append(torch.zeros((0, self.out_dim), device=device))
                points_result.append(torch.zeros((0, self.n_points, 3), device=device))
                continue

            masks_t = torch.from_numpy(masks_b).to(device=device)
            if masks_t.dtype != torch.bool:
                masks_t = masks_t.bool()
            M = masks_t.shape[0]

            xyz_list: list[torch.Tensor] = []
            feat_list: list[torch.Tensor] = []
            raw_pts_list: list[torch.Tensor] = []
            valid_idx: list[int] = []

            for i in range(M):
                pts = self._points_from_mask(rgb01[b], depth[b], masks_t[i], intrinsics=intrinsics_b, dense=False)
                if pts is None:
                    continue
                xyz_i, feat_i, raw_pts_i, _ = pts
                xyz_list.append(xyz_i)
                feat_list.append(feat_i)
                raw_pts_list.append(raw_pts_i)
                valid_idx.append(i)

            feats_b = torch.zeros((M, self.out_dim), device=device)
            pts_b = torch.zeros((M, self.n_points, 3), device=device)

            if len(xyz_list) > 0:
                xyz_batch = torch.stack(xyz_list, dim=0)
                feat_batch = torch.stack(feat_list, dim=0)
                raw_pts_batch = torch.stack(raw_pts_list, dim=0)

                pts_b[torch.tensor(valid_idx, device=device, dtype=torch.long)] = raw_pts_batch

                emb = self.backbone(xyz_batch, feat_batch)
                feats_b[torch.tensor(valid_idx, device=device, dtype=torch.long)] = emb.to(feats_b.dtype)

            result.append(feats_b)
            points_result.append(pts_b)

        if return_points:
            return result, points_result
        return result





class T5TextEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "t5-base",
        max_len: int = 77,
        freeze: bool = True,
        load_encoder: bool = True,
        qwen_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except Exception as e:
            raise ImportError(
                "T5TextEncoder requires HuggingFace transformers (+ sentencepiece). "
                "Install with: pip install transformers sentencepiece"
            ) from e

        self.model_name = model_name
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if load_encoder:
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()
            self.d_model = int(self.encoder.config.d_model)
        else:
            self.encoder = None
            from transformers import AutoConfig
            self.d_model = int(AutoConfig.from_pretrained(model_name).d_model)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, padding_side="left")
        if self.qwen_tokenizer.pad_token_id is None:
            self.qwen_tokenizer.pad_token_id = self.qwen_tokenizer.eos_token_id
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.qwen_model.eval()


    def forward(self, text_prompts: list[str]) -> dict[str, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        enc = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        tokens = out.last_hidden_state
        return {
            "tokens": tokens,
            "attention_mask": attention_mask.bool(),
        }

    def _extract_noun_phrases_qwen(self, text_prompts: list[str]) -> list[dict]:
        prompt = (
            "You are a precise entity extractor for robotic placement commands.\n"
            "Extract two things from the command:\n"
            "1. 'source': the FULL noun phrase of the object being moved — copy it exactly, including the determiner AND all adjectives.\n"
            "2. 'targets': a JSON array of FULL noun phrases for all destination/reference objects in order — include ALL adjectives.\n\n"
            "Rules:\n"
            "- Every string MUST be an EXACT substring of the command — copy character-for-character.\n"
            "- NEVER drop adjectives or colors. 'the red cup' not 'the cup'. 'the black tray' not 'the tray'.\n"
            "- 'source' is the object directly after the action verb (place/put/move/set/etc.). It MUST be the moved object only — NEVER include any spatial preposition or anything that comes after it (no 'on ...', 'to the ...', 'in ...', 'behind ...', 'next to ...').\n"
            "- 'targets' MUST be non-empty whenever the command contains a spatial preposition (on, in, under, above, behind, in front of, next to, between, near, to the left/right/front/back of, on top of). EVERY noun phrase that follows such a preposition is a separate target, in left-to-right order.\n"
            "- A target is ALWAYS a separate noun phrase from 'source' — do NOT merge 'source' and 'targets' into one phrase.\n"
            "- Output exactly one JSON object, no markdown, no extra text.\n\n"
            "Examples:\n"
            'Command: Place the red cup to the left of the blue bottle.\n'
            '{{"source": "the red cup", "targets": ["the blue bottle"]}}\n\n'
            'Command: Move the small wooden block on top of the metal table and in front of the white shelf.\n'
            '{{"source": "the small wooden block", "targets": ["the metal table", "the white shelf"]}}\n\n'
            'Command: Put the purple mug on the black tray.\n'
            '{{"source": "the purple mug", "targets": ["the black tray"]}}\n\n'
            'Command: Place the green bowl to the left of the white plate on the wooden shelf.\n'
            '{{"source": "the green bowl", "targets": ["the white plate", "the wooden shelf"]}}\n\n'
            'Command: Set the book behind the lamp.\n'
            '{{"source": "the book", "targets": ["the lamp"]}}\n\n'
            "Command: {t}\n"
        )
        chat_inputs = [
            self.qwen_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a precise JSON extractor. Output only valid JSON."},
                    {"role": "user",   "content": prompt.format(t=t)},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            for t in text_prompts
        ]
        inputs = self.qwen_tokenizer(
            chat_inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.qwen_model.device)
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )
        results = []
        for i, output_seq in enumerate(outputs):
            gen_tokens = output_seq[len(inputs.input_ids[i]):]
            raw = self.qwen_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = {}
            for candidate in [raw, None]:
                try:
                    if candidate is None:
                        m = re.search(r"\{.*?\}", raw, re.DOTALL)
                        if m is None:
                            break
                        candidate = m.group()
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and ("source" in parsed or "targets" in parsed):
                        break
                    parsed = {}
                except Exception:
                    parsed = {}
            if isinstance(parsed.get("targets"), str):
                parsed["targets"] = [parsed["targets"]]
            if not parsed:
                print(f"[Qwen] Failed to parse noun phrases from: {raw!r}")
            results.append(parsed)
        return results

    def extract_noun_phrases(
        self, text_prompts: list[str]
    ) -> tuple[
        list[list[tuple[int, int]]],
        list[list[tuple[int, int] | None]],
        list[dict] | None,
    ]:
        enc = self.tokenizer(
            text_prompts,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        if "offset_mapping" not in enc:
            return [[] for _ in text_prompts], [[] for _ in text_prompts], None

        offsets = enc["offset_mapping"]
        phrase_results: list[list[tuple[int, int]]] = []
        relation_results: list[list[tuple[int, int] | None]] = []

        qwen_extracted_dicts = self._extract_noun_phrases_qwen(text_prompts)

        for i, text in enumerate(text_prompts):
            toks_off = offsets[i]
            data = qwen_extracted_dicts[i]
            spans: list[tuple[int, int]] = []

            def _char_to_tok_span(phrase: str) -> tuple[int, int] | None:
                start_char = text.lower().find(phrase.lower())
                if start_char == -1:
                    return None
                end_char = start_char + len(phrase)
                tok_idxs = [
                    j for j, (a, b) in enumerate(toks_off)
                    if b > 0 and not (b <= start_char or a >= end_char)
                ]
                return (min(tok_idxs), max(tok_idxs) + 1) if tok_idxs else None

            src_span = _char_to_tok_span(data.get("source", ""))
            if src_span:
                spans.append(src_span)

            for tgt in data.get("targets", [])[:MAX_ANCHORS]:
                tgt_span = _char_to_tok_span(tgt)
                if tgt_span:
                    spans.append(tgt_span)

            rel_spans: list[tuple[int, int] | None] = []
            for k in range(len(spans) - 1):
                left_end = spans[k][1]
                right_start = spans[k + 1][0]
                rel_spans.append((left_end, right_start) if right_start > left_end else None)
            

            phrase_results.append(spans)
            relation_results.append(rel_spans)
        return phrase_results, relation_results, qwen_extracted_dicts



class ObjectTokenBuilder(nn.Module):

    N_MODALITIES = 3

    def __init__(
        self,
        d_dino: int,
        d_pn: int,
        d_spatial: int = 7,
        d_model: int = 512,
    ):
        super().__init__()

        self.d_model = int(d_model)

        self.proj_dino = nn.Sequential(nn.Linear(int(d_dino), self.d_model), nn.LayerNorm(self.d_model))
        self.proj_pn = nn.Sequential(nn.Linear(int(d_pn), self.d_model), nn.LayerNorm(self.d_model))
        self.proj_spatial = nn.Sequential(
            nn.Linear(int(d_spatial), 64),
            nn.GELU(),
            nn.Linear(64, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.scene_embedding = nn.Embedding(2, self.d_model)

        init_gate = 1.0 / (self.N_MODALITIES ** 0.5)
        self.gate_dino = nn.Parameter(torch.tensor(init_gate))
        self.gate_pn = nn.Parameter(torch.tensor(init_gate))
        self.gate_spatial = nn.Parameter(torch.tensor(init_gate))

        self.out_norm = nn.LayerNorm(self.d_model)

    @staticmethod
    def _pad_to_dense(
        feats_list: list[torch.Tensor],
        d: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(feats_list)
        if B == 0:
            dense = torch.zeros((0, 0, int(d)), dtype=torch.float32)
            mask = torch.zeros((0, 0), dtype=torch.bool)
            return dense, mask

        M_max = max(int(f.shape[0]) for f in feats_list) if feats_list else 0
        device = feats_list[0].device
        dtype = feats_list[0].dtype

        dense = torch.zeros((B, M_max, int(d)), device=device, dtype=dtype)
        mask = torch.zeros((B, M_max), device=device, dtype=torch.bool)

        for b, f in enumerate(feats_list):
            if f.numel() == 0:
                continue
            if f.ndim != 2 or int(f.shape[1]) != int(d):
                raise ValueError(f"Expected (M, {d}) tensor, got {tuple(f.shape)}")
            M_i = int(f.shape[0])
            dense[b, :M_i] = f
            mask[b, :M_i] = True

        return dense, mask

    def forward(
        self,
        dino_feats_list: list[torch.Tensor],
        pointnet_feats_list: list[torch.Tensor],
        spatial_feats_list: list[torch.Tensor],
        scene_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (len(dino_feats_list) == len(pointnet_feats_list) == len(spatial_feats_list)):
            raise ValueError("All modality lists must have the same batch length")

        B = len(dino_feats_list)
        for b in range(B):
            M = int(dino_feats_list[b].shape[0])
            if int(pointnet_feats_list[b].shape[0]) != M or int(spatial_feats_list[b].shape[0]) != M:
                raise ValueError(
                    f"Mismatched object counts at batch index {b}: "
                    f"dino={int(dino_feats_list[b].shape[0])}, "
                    f"pointnet={int(pointnet_feats_list[b].shape[0])}, "
                    f"spatial={int(spatial_feats_list[b].shape[0])}"
                )

        dino_dense, obj_mask = self._pad_to_dense(dino_feats_list, self.proj_dino[0].in_features)
        pn_dense, _ = self._pad_to_dense(pointnet_feats_list, self.proj_pn[0].in_features)
        sp_dense, _ = self._pad_to_dense(spatial_feats_list, self.proj_spatial[0].in_features)

        z_dino = self.proj_dino(dino_dense)
        z_pn = self.proj_pn(pn_dense)
        z_spatial = self.proj_spatial(sp_dense)

        fused = (
            self.gate_dino * z_dino
            + self.gate_pn * z_pn
            + self.gate_spatial * z_spatial
        )

        scene_id_idx = torch.full(
            (fused.shape[0], fused.shape[1]), fill_value=scene_id,
            dtype=torch.long, device=fused.device,
        )
        fused = fused + self.scene_embedding(scene_id_idx)

        fused = fused * obj_mask.unsqueeze(-1).to(dtype=fused.dtype)
        tokens = self.out_norm(fused)
        tokens = tokens * obj_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, obj_mask

    @staticmethod
    def compute_spatial_feats(
        masks: list[np.ndarray],
        depth: torch.Tensor,
        camera_intrinsics: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        
        if depth.ndim != 3:
            raise ValueError(f"Expected depth (B,H,W), got {tuple(depth.shape)}")

        B, H, W = depth.shape
        device = depth.device
        result: list[torch.Tensor] = []

        if len(masks) != B:
            raise ValueError(f"masks list length {len(masks)} != batch {B}")

        for b in range(B):
            masks_b = masks[b]
            M = int(len(masks_b))
            if M == 0:
                result.append(torch.zeros((0, 7), device=device, dtype=depth.dtype))
                continue

            spatial = torch.zeros((M, 7), device=device, dtype=depth.dtype)
            depth_b = depth[b]
            
            intrinsics_b = camera_intrinsics[b] if camera_intrinsics is not None else None
            if intrinsics_b is not None and intrinsics_b[0] > 0:
                fx, fy, cx, cy = intrinsics_b[0], intrinsics_b[1], intrinsics_b[2], intrinsics_b[3]
            else:
                cx = 0.5 * float(W - 1)
                cy = 0.5 * float(H - 1)
                fx = float(max(W, 1))
                fy = float(max(H, 1))

            for i, m_np in enumerate(masks_b):
                m = torch.from_numpy(m_np).to(device=device)
                if m.dtype != torch.bool:
                    m = m.bool()

                valid = m & (depth_b > 0) & torch.isfinite(depth_b)
                idx = valid.nonzero(as_tuple=False)
                if idx.numel() == 0:
                    continue
                
                y = idx[:, 0].float()
                x = idx[:, 1].float()
                z = depth_b[y.long(), x.long()]

                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                pts = torch.stack([X, Y, z], dim=-1)

                centroid = pts.mean(dim=0)
                extent = pts.max(dim=0).values - pts.min(dim=0).values

                xy = pts[:, :2] - pts[:, :2].mean(dim=0, keepdim=True)
                if xy.shape[0] >= 2:
                    cov = (xy.transpose(0, 1) @ xy) / max(1, xy.shape[0] - 1)
                    original_dtype = cov.dtype
                    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))
                    eigvals = torch.nan_to_num(eigvals.to(original_dtype), nan=0.0)
                    eigvecs = torch.nan_to_num(eigvecs.to(original_dtype), nan=0.0)
                    v = eigvecs[:, torch.argmax(eigvals)]
                    orient = torch.atan2(v[1], v[0])
                else:
                    orient = torch.zeros((), device=device, dtype=depth.dtype)

                spatial[i] = torch.cat([centroid, extent, orient.unsqueeze(0)]).to(dtype=depth.dtype)

            result.append(spatial)

        return result



class SceneSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int | None = None,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_dim = None if ff_dim is None else int(ff_dim)
        self.dropout = float(dropout)

        self._built_d_model: int | None = None
        self.encoder: nn.TransformerEncoder | None = None
        self.final_norm: nn.LayerNorm | None = None

        if d_model is not None:
            self._build(int(d_model), device=None, dtype=None)

    def _build(
        self,
        d_model: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_model % self.n_heads != 0:
            raise ValueError(
                f"Transformer d_model={d_model} must be divisible by n_heads={self.n_heads}. "
                "If Stage 6 concatenates modality embeddings, choose per-modality projection dims "
                "so the concatenated dim is divisible by n_heads."
            )

        ff_dim = self.ff_dim if self.ff_dim is not None else 4 * d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(self.n_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(self.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(self.n_layers),
        )
        self.final_norm = nn.LayerNorm(int(d_model))

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

        self._built_d_model = int(d_model)

    def forward(self, object_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None):
        if object_tokens.ndim != 3:
            raise ValueError(f"Expected object_tokens (B,M,D), got {tuple(object_tokens.shape)}")

        B, M, D = object_tokens.shape
        if M == 0:
            return object_tokens

        if self.encoder is None or self.final_norm is None:
            self._build(D, device=object_tokens.device, dtype=object_tokens.dtype)
        elif self._built_d_model is not None and D != self._built_d_model:
            raise ValueError(
                f"SceneSelfAttention was built for d_model={self._built_d_model} but got input with d_model={D}."
            )

        src_key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != (B, M):
                raise ValueError(
                    f"Expected padding_mask (B,M)=({B},{M}), got {tuple(padding_mask.shape)}"
                )
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.bool()
            src_key_padding_mask = ~padding_mask

        assert self.encoder is not None
        assert self.final_norm is not None
        x = self.encoder(object_tokens, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x






class InstructionCrossAttention(nn.Module):

    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.d_model  = int(d_model)
        self.n_heads  = int(n_heads)
        self.n_layers = int(n_layers)

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
            for _ in range(self.n_layers)
        ])
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
            for _ in range(self.n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Linear(self.d_model * 4, self.d_model),
            )
            for _ in range(self.n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
        self.norms3 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])

    def forward(
        self,
        text_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        object_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = text_tokens
        for cross_attn, self_attn, ffn, norm1, norm2, norm3 in zip(
            self.cross_attns, self.self_attns, self.ffns,
            self.norms1, self.norms2, self.norms3,
        ):
            attn_out, _ = cross_attn(
                x, object_tokens, object_tokens,
                key_padding_mask=object_padding_mask,
            )
            x = norm1(x + attn_out)

            self_out, _ = self_attn(x, x, x)
            x = norm2(x + self_out)

            x = norm3(x + ffn(x))

        return x



RELATION_CLASSES = [
    "ontop", "left", "right", "front", "back"
]
N_RELATION_CLASSES = len(RELATION_CLASSES)
MAX_ANCHORS = 5


class RelationClassifier(nn.Module):

    def __init__(self, d_t5: int = 768, d_rel: int = 512):
        super().__init__()
        self.d_t5  = int(d_t5)
        self.d_rel = int(d_rel)

        self.classifier = nn.Linear(self.d_t5, N_RELATION_CLASSES)

        self.embedding_table = nn.Embedding(N_RELATION_CLASSES, self.d_rel)

    def forward(
        self,
        cls_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self.classifier(cls_token)
        pred_class = logits.argmax(dim=-1)

        weights = logits.softmax(dim=-1)
        relation_emb = weights @ self.embedding_table.weight

        return {
            "logits":       logits,
            "relation_emb": relation_emb,
            "pred_class":   pred_class,
        }


class SceneQueryBottleneck(nn.Module):

    def __init__(
        self,
        n_queries:    int = 32,
        d_model:      int = 512,
        relation_dim: int = 512,
        n_heads:      int = 8,
    ):
        super().__init__()
        self.n_queries    = int(n_queries)
        self.d_model      = int(d_model)
        self.relation_dim = int(relation_dim)
        self.n_heads      = int(n_heads)

        self.queries = nn.Parameter(torch.randn(self.n_queries, self.d_model))

        self.relation_mlp = nn.Sequential(
            nn.Linear(self.relation_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.vis_cross = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
        self.vis_self  = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)

        self.lang_cross = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
        self.lang_self  = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)

        self.ffn_vis = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.ffn_lang = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        self.norm_cross1 = nn.LayerNorm(self.d_model)
        self.norm_self1  = nn.LayerNorm(self.d_model)
        self.norm_ffn1   = nn.LayerNorm(self.d_model)

        self.norm_cross2 = nn.LayerNorm(self.d_model)
        self.norm_self2  = nn.LayerNorm(self.d_model)
        self.norm_ffn2   = nn.LayerNorm(self.d_model)

    def forward(
        self,
        o_ref_tokens:        torch.Tensor,
        o_anchor_tokens:     torch.Tensor,
        scene_tokens:        torch.Tensor,
        relation_emb:        torch.Tensor,
        text_tokens:         torch.Tensor | None = None,
        scene_padding_mask:  torch.Tensor | None = None,
        text_padding_mask:   torch.Tensor | None = None,
        anchor_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = o_ref_tokens.shape[0]

        q = self.queries.unsqueeze(0).expand(B, -1, -1).clone()
        cond = self.relation_mlp(relation_emb).unsqueeze(1)
        q = q + cond

        kv_vis = torch.cat([o_ref_tokens, o_anchor_tokens, scene_tokens], dim=1)

        P = o_ref_tokens.shape[1]
        A = o_anchor_tokens.shape[1]
        oref_mask_pt = torch.zeros(B, P, dtype=torch.bool, device=scene_tokens.device)

        if anchor_padding_mask is not None:
            anchor_mask_pt = anchor_padding_mask
        else:
            anchor_mask_pt = torch.zeros(B, A, dtype=torch.bool, device=scene_tokens.device)

        if scene_padding_mask is not None:
            scene_mask_pt = ~scene_padding_mask
            kv_vis_mask = torch.cat([oref_mask_pt, anchor_mask_pt, scene_mask_pt], dim=1)
        else:
            kv_vis_mask = torch.cat([oref_mask_pt, anchor_mask_pt,
                                     torch.zeros(B, scene_tokens.shape[1],
                                                 dtype=torch.bool, device=scene_tokens.device)], dim=1)

        cross_v, _ = self.vis_cross(
            q, kv_vis, kv_vis,
            key_padding_mask=kv_vis_mask,
        )
        q = self.norm_cross1(q + cross_v)

        self_v, _ = self.vis_self(q, q, q)
        q = self.norm_self1(q + self_v)

        q = self.norm_ffn1(q + self.ffn_vis(q))

        if text_tokens is not None:
            cross_l, _ = self.lang_cross(
                q, text_tokens, text_tokens,
                key_padding_mask=text_padding_mask,
            )
            q = self.norm_cross2(q + cross_l)

            self_l, _ = self.lang_self(q, q, q)
            q = self.norm_self2(q + self_l)

            q = self.norm_ffn2(q + self.ffn_lang(q))

        return q



class ActionLatentBottleneck(nn.Module):

    def __init__(self, d_model: int = 512, d_latent: int = 256, n_heads: int = 8):
        super().__init__()
        self.d_latent = d_latent

        self.latent_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.attn_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.GELU(),
            nn.LayerNorm(d_latent),
        )

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        B = query_tokens.shape[0]
        q_lat = self.latent_token.expand(B, -1, -1)

        pooled, _ = self.attn_pool(q_lat, query_tokens, query_tokens)
        pooled = self.norm(pooled.squeeze(1))

        return self.proj(pooled)




class PlacementPredictionHead(nn.Module):

    def __init__(self, d_z: int = 512, d_hidden: int = 256):
        super().__init__()
        self.d_z      = int(d_z)
        self.d_hidden = int(d_hidden)

        self.trunk = nn.Sequential(
            nn.Linear(self.d_z, self.d_hidden),
            nn.GELU(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.GELU(),
        )

        self.head_trans = nn.Linear(self.d_hidden, 3)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        delta_trans = self.head_trans(h)
        B = z.shape[0]
        rot_matrix = torch.eye(3, device=z.device, dtype=z.dtype).expand(B, 3, 3)
        return {
            "delta_trans": delta_trans,
            "rot_matrix":  rot_matrix,
        }



def reconstruct_scene(
    o_ref_points:      torch.Tensor,
    delta_trans:       torch.Tensor,
    rot_matrix:        torch.Tensor,
    o_ref_centroid:    torch.Tensor,
    o_anchor_centroid: torch.Tensor,
) -> torch.Tensor:
    centred = o_ref_points - o_ref_centroid.unsqueeze(1)

    rotated = torch.einsum("bij,bpj->bpi", rot_matrix, centred)

    origin = o_anchor_centroid + delta_trans
    transformed = rotated + origin.unsqueeze(1)

    return transformed




def _nn_dists(
    query: torch.Tensor,
    key:   torch.Tensor,
) -> torch.Tensor:
    diff = query.unsqueeze(2) - key.unsqueeze(1)
    return diff.pow(2).sum(-1).clamp(min=1e-6).sqrt().min(dim=-1).values


def loss_translation(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    return F.smooth_l1_loss(pred, gt, beta=beta)


def loss_collision(
    transformed_o_ref: torch.Tensor,
    scene_points:      torch.Tensor,
    tau_pen: float = 0.02,
) -> torch.Tensor:
    nn_dists = _nn_dists(transformed_o_ref, scene_points)
    return F.relu(tau_pen - nn_dists).mean()


def loss_collision_obb(
    transformed_o_ref: torch.Tensor,
    anchor_obb:        torch.Tensor,
    anchor_valid:      torch.Tensor,
    tau:     float = 0.005,
    inflate: float = 0.0,
) -> torch.Tensor:
    B, P, _ = transformed_o_ref.shape
    A = anchor_obb.shape[1]

    c = anchor_obb[..., :3]
    R = anchor_obb[..., 3:12].reshape(B, A, 3, 3)
    h = anchor_obb[..., 12:15] + inflate

    rel = transformed_o_ref[:, None, :, :] - c[:, :, None, :]
    local = torch.einsum("bapj,bajk->bapk", rel, R)

    face_gap = h[:, :, None, :] - local.abs()
    pen_depth = face_gap.clamp(min=0.0).min(dim=-1).values

    valid = anchor_valid[:, :, None]
    inside = (face_gap > 0).all(dim=-1) & valid
    pen = F.relu(pen_depth - tau) * inside.float()
    return pen.sum() / inside.float().sum().clamp(min=1.0)


def _surface_height_field(
    obj_pts:   torch.Tensor,
    scene_pts: torch.Tensor,
    xz_radius: float,
    k_surface: int,
    surface_band: float = 0.02,
    down: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, P, _ = obj_pts.shape
    M = scene_pts.shape[1]
    dtype = obj_pts.dtype
    big = torch.finfo(dtype).max / 4

    real = scene_pts.abs().sum(-1) > 1e-6

    if down is None:
        dx = obj_pts[:, :, None, 0] - scene_pts[:, None, :, 0]
        dz = obj_pts[:, :, None, 2] - scene_pts[:, None, :, 2]
        perp2 = dx * dx + dz * dz
        scene_h = scene_pts[:, None, :, 1].expand(B, P, M)
    else:
        down = F.normalize(down, dim=-1)                            
        obj_h   = torch.einsum("bpj,bj->bp", obj_pts, down)          
        scene_hm = torch.einsum("bmj,bj->bm", scene_pts, down)        
        diff = obj_pts[:, :, None, :] - scene_pts[:, None, :, :]    
        d2 = (diff * diff).sum(-1)                                    
        along = obj_h[:, :, None] - scene_hm[:, None, :]             
        perp2 = (d2 - along * along).clamp(min=0.0)
        scene_h = scene_hm[:, None, :].expand(B, P, M)

    within = (perp2 <= xz_radius * xz_radius) & real[:, None, :]

    masked_y = torch.where(within, scene_h, torch.full_like(scene_h, big))
    kref = min(2, M)
    y_ref = masked_y.topk(kref, dim=-1, largest=False).values[..., -1:]
    top_mask = within & (scene_h >= y_ref - surface_band) & (scene_h <= y_ref + surface_band)
    cnt = top_mask.sum(-1)
    surf = torch.where(top_mask, scene_h, torch.zeros_like(scene_h)).sum(-1) / cnt.clamp(min=1)
    return surf, within


def snap_to_support_surface(
    obj_pts:     torch.Tensor,
    scene_pts:   torch.Tensor,
    xz_radius:   float = 0.04,
    min_support: int   = 3,
    k_surface:   int   = 8,
    max_snap:    float = 5.35,
    bottom_band: float = 0.02,
    down:        torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = obj_pts.shape[0]
    if down is None:
        down_vec = torch.zeros(B, 3, device=obj_pts.device, dtype=obj_pts.dtype)
        down_vec[:, 1] = 1.0
    else:
        down_vec = F.normalize(down, dim=-1)

    surf, within = _surface_height_field(
        obj_pts, scene_pts, xz_radius, k_surface, down=down_vec
    )
    n_sup = within.sum(-1)
    supported = n_sup >= min_support

    obj_y = torch.einsum("bpj,bj->bp", obj_pts, down_vec)     
    y_bottom = obj_y.max(dim=1, keepdim=True).values
    is_bottom = obj_y >= y_bottom - bottom_band
    drive = supported & is_bottom
    drive = torch.where(drive.any(dim=1, keepdim=True), drive, supported)

    big = torch.finfo(obj_pts.dtype).max / 4
    gap = torch.where(drive, surf - obj_y,
                      torch.full_like(surf, big))                
    delta_y = gap.min(dim=1).values
    delta_y = torch.where(drive.any(dim=1), delta_y, torch.zeros_like(delta_y))
    delta_y = torch.where(delta_y.abs() <= max_snap, delta_y, torch.zeros_like(delta_y))

    snapped = obj_pts + delta_y[:, None, None] * down_vec[:, None, :]
    return snapped, delta_y


def ontop_snap_direction(
    o_anchor_obb:   torch.Tensor,   
    anchor_valid:   torch.Tensor, 
    relation_class: torch.Tensor,   
    max_tilt_deg:   float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = o_anchor_obb.shape[0]
    dev, dt = o_anchor_obb.device, o_anchor_obb.dtype
    down = torch.zeros(B, 3, device=dev, dtype=dt)
    down[:, 1] = 1.0                                             

    sel_slot = torch.full((B,), -1, dtype=torch.long, device=dev)

    ontop_idx = RELATION_CLASSES.index("ontop")
    cos_thr = math.cos(math.radians(max_tilt_deg))

    for b in range(B):
        is_ontop = anchor_valid[b] & (relation_class[b] == ontop_idx)
        slots = torch.nonzero(is_ontop, as_tuple=False).flatten()
        if slots.numel() == 0:
            continue
        slot = int(slots[0].item())
        sel_slot[b] = slot
        R = o_anchor_obb[b, slot, 3:12].reshape(3, 3)
        n = R[:, 1]                                               
        norm = torch.linalg.norm(n)
        if not torch.isfinite(n).all() or norm < 1e-6:
            continue
        n = n / norm
        if n[1] < 0:                                             
            n = -n
        if n[1] < cos_thr:                                       
            continue
        down[b] = n
    return down, sel_slot


def loss_surface_penetration(
    transformed_o_ref: torch.Tensor,
    scene_points:      torch.Tensor,
    xz_radius:   float = 0.04,
    tau:         float = 0.005,
    min_support: int   = 3,
    k_surface:   int   = 8,
) -> torch.Tensor:
    surf, within = _surface_height_field(
        transformed_o_ref, scene_points, xz_radius, k_surface
    )
    supported = within.sum(-1) >= min_support
    pen = F.relu((transformed_o_ref[:, :, 1] - surf) - tau)
    pen = pen * supported.float()
    return pen.sum() / supported.float().sum().clamp(min=1.0)


def loss_vertical_support(
    transformed_o_ref: torch.Tensor,
    scene_points:      torch.Tensor,
    bottom_fraction:   float = 0.15,
    n_sample:          int   = 64,
    tau_support:       float = 0.02,
    w_horiz:           float = 4.0,
    w_vert:            float = 1.0,
) -> torch.Tensor:
    B, P, _ = transformed_o_ref.shape
    dev = transformed_o_ref.device

    k = max(1, int(P * bottom_fraction))
    bot_idx_full = transformed_o_ref[:, :, 1].topk(k, dim=1, largest=True).indices

    n_s = min(n_sample, k)
    perm = torch.rand(B, k, device=dev).argsort(dim=1)[:, :n_s]
    bot_idx = bot_idx_full.gather(1, perm)
    bot_pts = transformed_o_ref.gather(
        1, bot_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  

    diff     = scene_points[:, None, :, :] - bot_pts[:, :, None, :]
    horiz_d2 = diff[..., 0].pow(2) + diff[..., 2].pow(2)
    vert_d   = diff[..., 1]

    real_pt = scene_points.abs().sum(-1) > 1e-6
    big = torch.finfo(diff.dtype).max / 4
    metric = w_horiz * horiz_d2 + w_vert * vert_d.pow(2)
    metric = metric.masked_fill(~real_pt[:, None, :], big)

    nn = metric.argmin(dim=-1, keepdim=True)
    horiz_nn = horiz_d2.gather(-1, nn).squeeze(-1).clamp_min(1e-12).sqrt()
    vert_nn  = vert_d.gather(-1, nn).squeeze(-1)

    return (w_horiz * horiz_nn + w_vert * F.relu(vert_nn - tau_support)).mean()


def loss_relation_class(
    logits:       torch.Tensor,
    gt_class_idx: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits, gt_class_idx)


def _statistical_outlier_mask(P: np.ndarray, k: int = 16, n_std: float = 2.0) -> np.ndarray:
    N = len(P)
    if N <= k + 1:
        return np.ones(N, dtype=bool)
    tree = cKDTree(P)
    d, _ = tree.query(P, k=k + 1)
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + n_std * md.std()
    return md <= thr


def _ransac_plane_normal(P: np.ndarray, thr: float, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    N = len(P)
    best_in, best_mask = -1, None
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        inl = np.abs((P - a) @ n) < thr
        s = int(inl.sum())
        if s > best_in:
            best_in, best_mask = s, inl
    if best_mask is None or best_in < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0
    Q = P[best_mask]
    Qc = Q - Q.mean(0)
    evecs = np.linalg.eigh(Qc.T @ Qc)[1]
    n = evecs[:, 0]
    return n / (np.linalg.norm(n) + 1e-12), best_in / N


def anchor_obb(
    pts: torch.Tensor, max_pts: int = 3000, denoise: bool = True,
    extents_over_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev, dt = pts.device, pts.dtype
    eye = torch.eye(3, device=dev, dtype=dt)
    if pts.shape[0] < 3:
        c = pts.mean(0) if pts.shape[0] else torch.zeros(3, device=dev, dtype=dt)
        return c, eye, torch.zeros(3, device=dev, dtype=dt)

    p = pts.detach()
    if p.shape[0] > max_pts:
        sel = torch.randperm(p.shape[0], device=p.device)[:max_pts]
        p = p[sel]
    P = np.ascontiguousarray(p.cpu().numpy(), dtype=np.float32)
    if not np.isfinite(P).all():
        return pts.mean(0), eye, torch.zeros(3, device=dev, dtype=dt)

    if denoise and P.shape[0] >= 32:
        keep = _statistical_outlier_mask(P)
        if keep.sum() >= 16:
            P = P[keep]

    rng = np.random.default_rng(0)
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    thr = float(np.clip(0.01 * diag, 0.004, 0.03))
    n, frac = _ransac_plane_normal(P, thr=thr, iters=100, rng=rng)
    if frac < 0.25:
        n = np.array([0.0, 1.0, 0.0], dtype=P.dtype)

    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - (seed @ n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    pts2 = np.ascontiguousarray(np.stack([P @ u, P @ v], axis=1), dtype=np.float32)
    try:
        (_, _), (_, _), ang = cv2.minAreaRect(pts2)
        corners = cv2.boxPoints(((0.0, 0.0), (1.0, 1.0), ang))
    except cv2.error:
        return pts.mean(0), eye, torch.zeros(3, device=dev, dtype=dt)
    e0 = corners[1] - corners[0]
    d0 = e0 / (np.linalg.norm(e0) + 1e-9)
    ax1 = d0[0] * u + d0[1] * v
    ax2 = np.cross(n, ax1)

    axes = np.stack([ax1, ax2, n], axis=1)
    A = np.abs(axes)
    perm = [0, 0, 0]
    used_cam: set[int] = set()
    used_ax: set[int] = set()
    for _ in range(3):
        m = A.copy()
        for c in used_cam:
            m[c, :] = -1.0
        for k in used_ax:
            m[:, k] = -1.0
        ix = int(np.argmax(m))
        c, k = ix // 3, ix % 3
        perm[c] = k
        used_cam.add(c)
        used_ax.add(k)

    cols = []
    for c in range(3):
        vv = axes[:, perm[c]].copy()
        if vv[c] < 0:
            vv = -vv
        cols.append(vv)
    Rn = np.stack(cols, axis=1)
    if np.linalg.det(Rn) < 0:
        dg = np.array([Rn[c, c] for c in range(3)])
        Rn[:, int(np.argmin(dg))] *= -1.0

    if extents_over_all:
        P_all = np.ascontiguousarray(pts.detach().cpu().numpy(), dtype=np.float32)
        P_all = P_all[np.isfinite(P_all).all(axis=1)]
        if denoise and P_all.shape[0] >= 32:
            keep_all = _statistical_outlier_mask(P_all)
            if keep_all.sum() >= 16:
                P_all = P_all[keep_all]
        ext_src = P_all if P_all.shape[0] >= 1 else P
    else:
        ext_src = P
    local = ext_src @ Rn
    lmin = local.min(0)
    lmax = local.max(0)
    half_np = (lmax - lmin) * 0.5
    center_np = Rn @ ((lmax + lmin) * 0.5)

    R = torch.as_tensor(Rn, device=dev, dtype=dt)
    center = torch.as_tensor(center_np, device=dev, dtype=dt)
    half = torch.as_tensor(half_np, device=dev, dtype=dt)
    return center, R, half


_OBB_CORNER_SIGNS = torch.tensor(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=torch.float32,
)


def object_obb_offsets(X: torch.Tensor) -> torch.Tensor:
    B = X.shape[0]
    dev, dt = X.device, X.dtype
    signs = _OBB_CORNER_SIGNS.to(device=dev, dtype=dt)
    offs = torch.zeros(B, 8, 3, device=dev, dtype=dt)
    for b in range(B):
        c, Rb, h = anchor_obb(X[b])
        mu = X[b].mean(0).detach()
        corners = c.unsqueeze(0) + (signs * h.unsqueeze(0)) @ Rb.T
        offs[b] = corners - mu.unsqueeze(0)
    return offs.detach()


def loss_relation_geometric(
    transformed_o_ref: torch.Tensor,
    anchor_obb: torch.Tensor | None,
    relation_probs: torch.Tensor,
    clearance: float = 0.05,
    obj_corner_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    B = transformed_o_ref.shape[0]
    device = transformed_o_ref.device

    if B == 0 or anchor_obb is None:
        return torch.zeros(1, device=device).squeeze()

    center = anchor_obb[:, :3]
    R      = anchor_obb[:, 3:12].reshape(B, 3, 3)
    half   = anchor_obb[:, 12:15]

    if obj_corner_offsets is not None:
        mu = transformed_o_ref.mean(dim=1)
        obj_pts = mu.unsqueeze(1) + obj_corner_offsets
    else:
        obj_pts = transformed_o_ref

    obj_local = torch.einsum(
        "bpj,bjk->bpk", obj_pts - center.unsqueeze(1), R
    )                                               
    ref_min = obj_local.min(dim=1).values
    ref_max = obj_local.max(dim=1).values
    anchor_min = -half
    anchor_max =  half

    total = torch.zeros(B, device=device)
    for r, rel in enumerate(RELATION_CLASSES):
        p = relation_probs[:, r]
        if rel == "ontop":
            g = F.relu(ref_max[:, 1] - anchor_min[:, 1])
        elif "left" in rel:
            g = F.relu(ref_max[:, 0] - (anchor_min[:, 0] - clearance))
        elif "right" in rel:
            g = F.relu((anchor_max[:, 0] + clearance) - ref_min[:, 0])
        elif "front" in rel:
            g = F.relu(ref_max[:, 2] - (anchor_min[:, 2] - clearance))
        elif "back" in rel:
            g = F.relu((anchor_max[:, 2] + clearance) - ref_min[:, 2])
        else:
            continue
        total = total + p * g

    return total.mean()


class PlacementLoss(nn.Module):

    def __init__(
        self,
        lambda_trans:    float = 3.0,
        lambda_col:      float = 8.0,
        lambda_col_obb:  float = 8.0,
        lambda_penetration: float = 8.0,
        lambda_support:  float = 4.0,
        lambda_rel:      float = 7.0,
        lambda_geom:     float = 2.0,
        tau_pen:         float = 0.005,
        tau_penetration: float = 0.005,
        xz_radius:       float = 0.04,
        tau_support:     float = 0.02,
        bottom_fraction: float = 0.15,
        clearance:       float = 0.05,
        beta:            float = 0.1,
    ):
        super().__init__()
        self.lam_trans       = lambda_trans
        self.lam_col         = lambda_col
        self.lam_col_obb     = lambda_col_obb
        self.lam_penetration = lambda_penetration
        self.lam_support     = lambda_support
        self.lam_rel         = lambda_rel
        self.lam_geom        = lambda_geom
        self.tau_pen         = tau_pen
        self.tau_penetration = tau_penetration
        self.xz_radius       = xz_radius
        self.tau_support     = tau_support
        self.bottom_fraction = bottom_fraction
        self.clearance       = max(clearance, xz_radius)
        self.beta            = beta

    def forward(
        self,
        pred_delta_trans:   torch.Tensor,
        transformed_o_ref:  torch.Tensor,
        pred_anchor_centroid: torch.Tensor,
        relation_logits:    torch.Tensor,
        gt_anchor_pos:      torch.Tensor,
        gt_delta_trans:     torch.Tensor,
        gt_anchor_valid:    torch.Tensor,
        gt_relation_class:  torch.Tensor | None = None,
        scene_points:       torch.Tensor | None = None,
        o_anchor_bboxes:    torch.Tensor | None = None,
        o_anchor_obb:       torch.Tensor | None = None,
        rot_matrix:         torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        B    = pred_delta_trans.shape[0]
        dev  = pred_delta_trans.device
        zero = torch.zeros(1, device=dev).squeeze()

        pred_pos     = pred_anchor_centroid + pred_delta_trans
        gt_final_pos = gt_anchor_pos + gt_delta_trans
        per_slot = F.smooth_l1_loss(
            pred_pos.unsqueeze(1).expand_as(gt_final_pos),
            gt_final_pos, reduction="none", beta=self.beta,
        ).sum(-1)                                                      
        valid_f = gt_anchor_valid.float()
        n_valid = valid_f.sum(dim=1).clamp(min=1)
        losses["trans"] = ((per_slot * valid_f).sum(dim=1) / n_valid).mean()

        if scene_points is not None:
            losses["collision"] = loss_collision(
                transformed_o_ref, scene_points, tau_pen=self.tau_pen,
            )
            losses["penetration"] = loss_surface_penetration(
                transformed_o_ref, scene_points,
                xz_radius=self.xz_radius,
                tau=self.tau_penetration,
            )
            losses["support"] = loss_vertical_support(
                transformed_o_ref, scene_points,
                bottom_fraction=self.bottom_fraction,
                tau_support=self.tau_support,
            )
        else:
            losses["collision"]   = zero
            losses["penetration"] = zero
            losses["support"]     = zero

        if o_anchor_obb is not None:
            losses["collision_obb"] = loss_collision_obb(
                transformed_o_ref, o_anchor_obb, gt_anchor_valid,
            )
        else:
            losses["collision_obb"] = zero


        if gt_relation_class is not None:
            B_r = relation_logits.shape[0]
            logits_flat = relation_logits.view(B_r * MAX_ANCHORS, N_RELATION_CLASSES)
            gt_flat     = gt_relation_class.view(B_r * MAX_ANCHORS)
            mask_flat   = valid_f.view(B_r * MAX_ANCHORS)
            per_ce      = F.cross_entropy(logits_flat, gt_flat, reduction='none')
            losses["relation"] = (per_ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        else:
            losses["relation"] = zero


        if gt_relation_class is not None and o_anchor_obb is not None:
            relation_probs_all = (
                F.one_hot(gt_relation_class, N_RELATION_CLASSES).float()
                * valid_f.unsqueeze(-1)
            )
        else:
            pred_class = relation_logits.argmax(dim=-1)
            relation_probs_all = F.one_hot(pred_class, N_RELATION_CLASSES).float()

        geom_loss = zero
        if o_anchor_obb is not None:
            obj_offsets = object_obb_offsets(transformed_o_ref)
            for slot in range(MAX_ANCHORS):
                if not gt_anchor_valid[:, slot].any():
                    continue
                geom_loss = geom_loss + loss_relation_geometric(
                    transformed_o_ref,
                    o_anchor_obb[:, slot],
                    relation_probs_all[:, slot],
                    clearance=self.clearance,
                    obj_corner_offsets=obj_offsets,
                )
        losses["geometric"] = geom_loss


        losses["total"] = (
              self.lam_trans   * losses["trans"]
            + self.lam_col     * losses["collision"]
            + self.lam_col_obb * losses["collision_obb"]
            + self.lam_penetration * losses["penetration"]
            + self.lam_support * losses["support"]
            + self.lam_rel     * losses["relation"]
            + self.lam_geom    * losses["geometric"]
        )

        return losses


def _match_anchors_to_gt(
    grounded_pos: torch.Tensor,
    gt_pos:       torch.Tensor,
    grounded_valid: torch.Tensor,
    gt_valid:       torch.Tensor,
) -> list[int]:
    A = grounded_pos.shape[0]
    perm = list(range(A))
    used = [False] * A

    for gt_slot in range(A):
        if not gt_valid[gt_slot]:
            continue
        best_slot, best_dist = -1, float("inf")
        for g_slot in range(A):
            if used[g_slot] or not grounded_valid[g_slot]:
                continue
            dist = (grounded_pos[g_slot] - gt_pos[gt_slot]).norm().item()
            if dist < best_dist:
                best_dist, best_slot = dist, g_slot
        if best_slot >= 0:
            perm[gt_slot] = best_slot
            used[best_slot] = True

    return perm


class SceneTransformModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self._use_cached_features = cfg.get("use_cached_features", False)
        self._max_scene_loss_points = int(cfg.get("max_scene_loss_points", 2048))

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.dino = DINOv2ObjectEncoder(
            model_name=cfg.get("dino_model", "dinov2_vitb14"),
            freeze=cfg.get("freeze_dino", True),
            out_dim=None,
        )

        if not self._use_cached_features:
            self.sam2 = SAM2Segmenter(cfg["sam2_checkpoint"], cfg["sam2_config"])
            self.t5 = T5TextEncoder(
                model_name=cfg.get("t5_model", "t5-base"),
                max_len=cfg.get("t5_max_len", 77),
                freeze=cfg.get("freeze_t5", True),
            )
        else:
            self.sam2 = None
            self.t5 = T5TextEncoder(
                model_name=cfg.get("t5_model", "t5-base"),
                max_len=cfg.get("t5_max_len", 77),
                freeze=True,
                load_encoder=False,
            )
            print("Skipping SAM2/T5-encoder loading (use_cached_features=True).")

        self.gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
        if cfg.get("freeze_grounding_dino", True):
            for p in self.gd_model.parameters():
                p.requires_grad_(False)
            self.gd_model.eval()

        self.pointnet = PointNetPPObjectEncoder(
            out_dim=cfg.get("pnet_out_dim", 256),
            n_points=cfg.get("pnet_n_points", 1024),
            use_rgb=cfg.get("pnet_use_rgb", True),
            dense_n_points=cfg.get("pnet_dense_n_points", 0),
        )

        d_model = int(cfg.get("d_model", 512))
        dino_model = cfg.get("dino_model", "dinov2_vitb14")

        _t5_d_model = int(self.t5.d_model) if self.t5 is not None else 768

        self.token_builder = ObjectTokenBuilder(
            d_dino=_DINO_DIMS[dino_model],
            d_pn=int(cfg.get("pnet_out_dim", 256)),
            d_spatial=7,
            d_model=d_model,
        )

        self.scene_sa = SceneSelfAttention(
            d_model=d_model, n_heads=int(cfg.get("n_heads", 8)), n_layers=4
        )
        self.relation_classifier = RelationClassifier(
            d_t5=_t5_d_model, d_rel=d_model
        )
        self.proj_t5 = nn.Sequential(
            nn.Linear(_t5_d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.inst_ca = InstructionCrossAttention(
            d_model=d_model, n_heads=int(cfg.get("n_heads", 8)), n_layers=2
        )
        self.qformer = SceneQueryBottleneck(
            n_queries=32, d_model=d_model, relation_dim=d_model, n_heads=int(cfg.get("n_heads", 8))
        )
        d_latent = int(cfg.get("d_latent", 256))
        self.latent_bottleneck = ActionLatentBottleneck(d_model=d_model, d_latent=d_latent)
        self.relation_proj = nn.Linear(d_model, d_model)
        self.rel_cond_proj = nn.Sequential(
            nn.Linear(_t5_d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.placement_head = PlacementPredictionHead(
            d_z=d_latent, d_hidden=256
        )

    def _extract_phrase_tokens(
        self,
        t5_tokens: torch.Tensor,
        phrase_spans: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        B, L, d_t5 = t5_tokens.shape
        result: list[torch.Tensor] = []

        for b in range(B):
            phrase_tokens_b: list[torch.Tensor] = []
            for start, end in phrase_spans[b]:
                if start < 0 or end > L or start >= end:
                    continue
                phrase_tokens_b.append(t5_tokens[b, start:end, :])

            if len(phrase_tokens_b) == 0:
                result.append(torch.zeros((0, d_t5), device=t5_tokens.device, dtype=t5_tokens.dtype))
            else:
                result.append(torch.cat(phrase_tokens_b, dim=0))

        return result

    def _ground_entity(
        self,
        rgbd: torch.Tensor,
        boxes_list: list[np.ndarray],
        phrase_tokens_list: list[torch.Tensor],
        text_prompts: list[str] | None = None,
        phrase_spans: list[list[tuple[int, int]]] | None = None,
    ) -> dict[str, Any]:
        from PIL import Image

        B = len(phrase_tokens_list)
        device = rgbd.device
        
        rgb01 = _to_rgb01(rgbd[:, :3, :, :])
        rgb_uint8 = (rgb01.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).byte().cpu().numpy()

        selected_idx_list: list[torch.Tensor] = []

        for b in range(B):
            P_i = phrase_tokens_list[b].shape[0]
            M_i = boxes_list[b].shape[0] if len(boxes_list) > b else 0

            if P_i == 0 or M_i == 0 or text_prompts is None or phrase_spans is None:
                selected_idx_list.append(torch.zeros(P_i, dtype=torch.long, device=device))
                continue

            text = text_prompts[b]
            spans = phrase_spans[b]
            
            tokenized = self.t5.tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True, max_length=self.t5.max_len
            )
            input_ids = tokenized["input_ids"][0].tolist()

            row_valid = torch.zeros(P_i, dtype=torch.bool, device=device)
            selected = torch.zeros(P_i, dtype=torch.long, device=device)

            pil_img = Image.fromarray(rgb_uint8[b])
            W_img, H_img = pil_img.size

            token_offset = 0
            for start, end in spans:
                span_len = end - start
                if span_len <= 0:
                    continue

                phrase_ids = input_ids[start:end]
                decoded = self.t5.tokenizer.decode(phrase_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                if not decoded:
                    token_offset += span_len
                    continue

                inputs = self.gd_processor(images=pil_img, text=decoded + ".", return_tensors="pt").to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
                    out = self.gd_model(**inputs)
                
                max_logit = torch.sigmoid(out.logits[0]).max(dim=-1)[0]
                best_idx = torch.argmax(max_logit).item()
                if max_logit[best_idx].item() < 0.05:
                    pass
                else:
                    box = out.pred_boxes[0][best_idx].cpu().numpy()
                    cx, cy, w, h = box
                    dino_box = np.array([
                        (cx - 0.5*w)*W_img, (cy - 0.5*h)*H_img,
                        (cx + 0.5*w)*W_img, (cy + 0.5*h)*H_img
                    ])

                    sam_boxes = boxes_list[b]
                    
                    x1 = np.maximum(dino_box[0], sam_boxes[:, 0])
                    y1 = np.maximum(dino_box[1], sam_boxes[:, 1])
                    x2 = np.minimum(dino_box[2], sam_boxes[:, 2])
                    y2 = np.minimum(dino_box[3], sam_boxes[:, 3])

                    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                    box1_area = (dino_box[2] - dino_box[0]) * (dino_box[3] - dino_box[1])
                    boxes2_area = (sam_boxes[:, 2] - sam_boxes[:, 0]) * (sam_boxes[:, 3] - sam_boxes[:, 1])
                    union_area = box1_area + boxes2_area - inter_area
                    iou = inter_area / np.maximum(union_area, 1e-6)
                    
                    best_sam_idx = np.argmax(iou)
                    if iou[best_sam_idx] > 0.0:
                        selected[token_offset:token_offset + span_len] = int(best_sam_idx)
                        row_valid[token_offset:token_offset + span_len] = True

                token_offset += span_len

            selected[~row_valid] = -1
            selected_idx_list.append(selected)

        return {
            "selected_obj_idx": selected_idx_list,
        }

    @staticmethod
    def _mask_to_metric_points(
        depth_b: torch.Tensor,
        mask_t: torch.Tensor,
        intrinsics_b: torch.Tensor | None,
    ) -> torch.Tensor | None:
        H, W = depth_b.shape
        valid = mask_t & (depth_b > 0) & torch.isfinite(depth_b)
        idx = valid.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None
        ys = idx[:, 0].float()
        xs = idx[:, 1].float()
        z = depth_b[idx[:, 0], idx[:, 1]].float()
        if intrinsics_b is not None and float(intrinsics_b[0]) > 0:
            fx, fy, cx, cy = (intrinsics_b[0], intrinsics_b[1],
                              intrinsics_b[2], intrinsics_b[3])
        else:
            cx = 0.5 * (W - 1)
            cy = 0.5 * (H - 1)
            fx = float(max(W, 1))
            fy = float(max(H, 1))
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        return torch.stack([X, Y, z], dim=-1)

    def _ground_anchors_from_masks(
        self,
        scene_masks: list[np.ndarray],
        anchor_masks: torch.Tensor,
        anchor_valid: torch.Tensor | None,
        n_slots: int,
    ) -> list[dict[str, Any]]:
        device = anchor_masks.device
        B = anchor_masks.shape[0]
        am_np = anchor_masks.detach().cpu().numpy().astype(bool)

        grounding: list[dict[str, Any]] = []
        for slot in range(n_slots):
            sel_list: list[torch.Tensor] = []
            for b in range(B):
                valid = bool(anchor_valid[b, slot].item()) if anchor_valid is not None else True
                sm = scene_masks[b] if b < len(scene_masks) else None
                gm = am_np[b, slot]
                if not valid or sm is None or len(sm) == 0 or not gm.any():
                    sel_list.append(torch.tensor([-1], dtype=torch.long, device=device))
                    continue

                sm_flat = np.asarray(sm).reshape(len(sm), -1).astype(bool)
                gm_flat = gm.reshape(-1)
                inter = np.logical_and(sm_flat, gm_flat).sum(axis=1).astype(np.float32)
                union = np.logical_or(sm_flat, gm_flat).sum(axis=1).astype(np.float32)
                iou = inter / np.maximum(union, 1e-6)
                best = int(np.argmax(iou))
                sel = best if iou[best] > 0.0 else -1
                sel_list.append(torch.tensor([sel], dtype=torch.long, device=device))
            grounding.append({"selected_obj_idx": sel_list})
        return grounding

    def _extract_o_ref_via_gd(
        self,
        object_rgbd: torch.Tensor,
        source_phrases: list[str],
        camera_intrinsics: torch.Tensor | None,
        object_masks_list: list[np.ndarray] | None = None,
        object_boxes_list: list[np.ndarray] | None = None,
        provided_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        from PIL import Image

        B, _, H, W = object_rgbd.shape
        device = object_rgbd.device
        d_dino = self.dino.dino_dim
        d_pn   = self.pointnet.out_dim

        rgb01 = _to_rgb01(object_rgbd[:, :3])
        depth = object_rgbd[:, 3]
        rgb_uint8 = (rgb01.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).byte().cpu().numpy()

        patch_size = self.dino.patch_size
        pad_h = (patch_size - (H % patch_size)) % patch_size
        pad_w = (patch_size - (W % patch_size)) % patch_size
        rgb_padded = F.pad(rgb01, (0, pad_w, 0, pad_h), mode="constant", value=0.0) if (pad_h or pad_w) else rgb01
        H_pad, W_pad = H + pad_h, W + pad_w
        h_p, w_p = H_pad // patch_size, W_pad // patch_size
        patch_tokens = self.dino._get_patch_tokens(rgb_padded)

        dino_feats: list[torch.Tensor] = []
        pn_feats:   list[torch.Tensor] = []
        spatial_feats: list[torch.Tensor] = []
        o_ref_points   = torch.zeros(B, self.pointnet.dense_n_points, 3, device=device)
        o_ref_centroid = torch.zeros(B, 3, device=device)
        o_ref_bbox     = torch.zeros(B, 4, device=device)
        o_ref_valid    = torch.zeros(B, dtype=torch.bool, device=device)
        o_ref_masks: list[np.ndarray | None] = []

        def _empty():
            dino_feats.append(torch.zeros(0, d_dino, device=device))
            pn_feats.append(torch.zeros(0, d_pn, device=device))
            spatial_feats.append(torch.zeros(0, 7, device=device))
            o_ref_masks.append(None)

        for b in range(B):
            sam2_mask_np: np.ndarray | None = None

            use_provided = (
                provided_mask is not None
                and b < provided_mask.shape[0]
                and bool(provided_mask[b].any())
            )
            if use_provided:
                box_mask_px = provided_mask[b].to(device=device, dtype=torch.bool)
                ys_px, xs_px = torch.where(box_mask_px)
                x1, y1 = int(xs_px.min().item()), int(ys_px.min().item())
                x2, y2 = int(xs_px.max().item()) + 1, int(ys_px.max().item()) + 1
                sam2_mask_np = box_mask_px.cpu().numpy()
            else:
                phrase = source_phrases[b].strip() if source_phrases[b] else ""
                if not phrase:
                    _empty()
                    continue

                pil_img = Image.fromarray(rgb_uint8[b])
                inputs = self.gd_processor(images=pil_img, text=phrase + ".", return_tensors="pt").to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
                    gd_out = self.gd_model(**inputs)

                max_logit = torch.sigmoid(gd_out.logits[0]).max(dim=-1)[0]
                best_idx = int(torch.argmax(max_logit).item())
                if max_logit[best_idx].item() < 0.05:
                    _empty()
                    continue

                cx, cy, w, h = gd_out.pred_boxes[0][best_idx].cpu().numpy()
                x1 = max(0, int((cx - 0.5 * w) * W))
                y1 = max(0, int((cy - 0.5 * h) * H))
                x2 = min(W, int((cx + 0.5 * w) * W))
                y2 = min(H, int((cy + 0.5 * h) * H))
                if x2 <= x1 or y2 <= y1:
                    _empty()
                    continue

                gd_box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                gd_area = float((x2 - x1) * (y2 - y1))
                size_tol = 4.0

                if (object_masks_list is not None and object_boxes_list is not None
                        and b < len(object_masks_list) and len(object_masks_list[b]) > 0):
                    o_boxes_np = object_boxes_list[b]
                    o_masks_np = object_masks_list[b]

                    ix1 = np.maximum(gd_box_np[0], o_boxes_np[:, 0])
                    iy1 = np.maximum(gd_box_np[1], o_boxes_np[:, 1])
                    ix2 = np.minimum(gd_box_np[2], o_boxes_np[:, 2])
                    iy2 = np.minimum(gd_box_np[3], o_boxes_np[:, 3])
                    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                    sam_areas = ((o_boxes_np[:, 2] - o_boxes_np[:, 0]) *
                                 (o_boxes_np[:, 3] - o_boxes_np[:, 1]))
                    union = gd_area + sam_areas - inter
                    iou = inter / np.maximum(union, 1e-6)

                    size_ratio = sam_areas / max(gd_area, 1e-6)
                    size_ok = (size_ratio >= 1.0 / size_tol) & (size_ratio <= size_tol)

                    overlap_ok = iou > 0.0

                    valid = size_ok & overlap_ok
                    if valid.any():
                        best = int(np.argmax(np.where(valid, iou, -1.0)))
                        sam2_mask_np = o_masks_np[best]

                if sam2_mask_np is not None:
                    box_mask_px = torch.from_numpy(sam2_mask_np).to(device=device, dtype=torch.bool)
                else:
                    box_mask_px = torch.zeros(H, W, dtype=torch.bool, device=device)
                    box_mask_px[y1:y2, x1:x2] = True

            box_f = box_mask_px.float().unsqueeze(0).unsqueeze(0)
            if pad_h or pad_w:
                box_f = F.pad(box_f, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            box_patches = F.interpolate(box_f, size=(h_p, w_p), mode="area").squeeze(0).squeeze(0)
            weight = box_patches.sum().clamp(min=1e-6)
            dino_feat = (patch_tokens[b] * box_patches.unsqueeze(-1)).sum(dim=(0, 1)) / weight

            intr_b = camera_intrinsics[b] if camera_intrinsics is not None else None
            pts_out = self.pointnet._points_from_mask(rgb01[b], depth[b], box_mask_px, intrinsics=intr_b, dense=True)
            if pts_out is None:
                _empty()
                continue
            xyz, feat, xyz_raw, xyz_raw_dense = pts_out
            pn_feat = self.pointnet.backbone(xyz.unsqueeze(0), feat.unsqueeze(0)).squeeze(0)

            inliers = _ransac_point_inliers(xyz_raw)
            xyz_in = xyz_raw[inliers] if int(inliers.sum().item()) >= 3 else xyz_raw
            center_for_dense = xyz_in.mean(dim=0)
            dense_dists = (xyz_raw_dense - center_for_dense).norm(dim=-1)
            dense_radius = (xyz_in - center_for_dense).norm(dim=-1).quantile(0.95).clamp(min=1e-4)
            dense_inliers = dense_dists <= dense_radius
            if int(dense_inliers.sum().item()) >= max(8, xyz_raw_dense.shape[0] // 10):
                kept = xyz_raw_dense[dense_inliers]
                pad_idx = torch.randint(0, kept.shape[0], (xyz_raw_dense.shape[0],), device=kept.device)
                xyz_raw_dense = kept[pad_idx]

            centroid = xyz_in.mean(dim=0)
            extent = xyz_in.max(dim=0).values - xyz_in.min(dim=0).values
            xy = xyz_in[:, :2] - xyz_in[:, :2].mean(dim=0, keepdim=True)
            if xy.shape[0] >= 2:
                cov = (xy.transpose(0, 1) @ xy) / max(1, xy.shape[0] - 1)
                eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))
                v = eigvecs[:, torch.argmax(eigvals)]
                orient = torch.atan2(v[1], v[0]).to(xyz_in.dtype)
            else:
                orient = torch.zeros((), device=device, dtype=xyz_in.dtype)
            spatial = torch.cat([centroid, extent, orient.unsqueeze(0)])

            dino_feats.append(dino_feat.unsqueeze(0))
            pn_feats.append(pn_feat.unsqueeze(0))
            spatial_feats.append(spatial.unsqueeze(0))
            o_ref_points[b]   = xyz_raw_dense
            o_ref_centroid[b] = centroid
            o_ref_bbox[b]     = torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32)
            o_ref_valid[b]    = True
            o_ref_masks.append(
                sam2_mask_np if sam2_mask_np is not None
                else box_mask_px.cpu().numpy()
            )

        return {
            "dino_feats": dino_feats,
            "pn_feats": pn_feats,
            "spatial_feats": spatial_feats,
            "o_ref_points": o_ref_points,
            "o_ref_centroid": o_ref_centroid,
            "o_ref_bbox": o_ref_bbox,
            "o_ref_valid": o_ref_valid,
            "o_ref_masks": o_ref_masks,
        }

    def forward(
        self,
        scene_rgbd:  torch.Tensor,
        object_rgbd: torch.Tensor,
        text_prompt: list[str],
        cached_features: dict[str, Any] | None = None,
        camera_intrinsics: torch.Tensor | None = None,
        gt_relation_class: torch.Tensor | None = None,
        gt_anchor_valid:   torch.Tensor | None = None,
        gt_anchor_pos:     torch.Tensor | None = None,
        snap_to_surface:   bool = True,
        provided_object_mask:  torch.Tensor | None = None,
        provided_anchor_masks: torch.Tensor | None = None,
        provided_anchor_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:

        if cached_features is not None:
            dev = scene_rgbd.device
            seg = cached_features["seg"]
            scene_masks = seg["scene_masks"]
            scene_boxes = seg["scene_boxes"]

            scene_dino  = [t.to(dev) for t in cached_features["scene_dino"]]

            text_out = {k: v.to(dev) for k, v in cached_features["text_out"].items()}
        else:
            seg = self.sam2(scene_rgbd, object_rgbd)
            scene_masks = seg["scene_masks"]
            scene_boxes = seg["scene_boxes"]

            scene_dino = self.dino(scene_rgbd, scene_masks, scene_boxes)

            text_out = self.t5(text_prompt)

        scene_pn, scene_points_list = self.pointnet(scene_rgbd, scene_masks, return_points=True, camera_intrinsics=camera_intrinsics)

        scene_depth = scene_rgbd[:, 3, :, :]
        scene_spatial = ObjectTokenBuilder.compute_spatial_feats(scene_masks, scene_depth, camera_intrinsics=camera_intrinsics)

        scene_tokens, scene_padding_mask = self.token_builder(
            scene_dino, scene_pn, scene_spatial, scene_id=1
        )

        scene_centroids_list = [sp[:, :3] for sp in scene_spatial]

        attn_mask_f = text_out["attention_mask"].unsqueeze(-1).float()
        t5_cls = (text_out["tokens"] * attn_mask_f).sum(1) / attn_mask_f.sum(1).clamp(min=1e-6)

        phrase_spans, relation_spans, qwen_dicts = self.t5.extract_noun_phrases(text_prompt)

        qwen_valid = torch.tensor(
            [
                bool(
                    isinstance(qd, dict)
                    and qd.get("source")
                    and qd.get("targets")
                )
                for qd in (qwen_dicts or [{}] * len(text_prompt))
            ],
            dtype=torch.bool,
            device=scene_tokens.device,
        )

        if not self.training:
            for b_idx, prompt in enumerate(text_prompt):
                qd = qwen_dicts[b_idx] if qwen_dicts and b_idx < len(qwen_dicts) else {}
                source = qd.get("source", "")
                targets = qd.get("targets", [])
                
                print(f"--- Inference Instance {b_idx} ---")
                print(f"Prompt: {prompt}")
                print(f"Source: {source}")
                
                p_lower = prompt.lower()
                last_end = 0
                if source:
                    idx = p_lower.find(source.lower())
                    if idx != -1:
                        last_end = idx + len(source)
                
                for t_idx, target in enumerate(targets):
                    t_idx_pos = p_lower.find(target.lower(), last_end)
                    if t_idx_pos != -1:
                        rel_text = prompt[last_end:t_idx_pos].strip()
                        print(f"Relation {t_idx}: {rel_text}")
                        last_end = t_idx_pos + len(target)
                    else:
                        print(f"Relation {t_idx}: [Not found]")
                    print(f"Target {t_idx}: {target}")
                print("-" * 30)

        phrase_tokens_list = self._extract_phrase_tokens(text_out["tokens"], phrase_spans)

        source_phrases = [
            (qd.get("source", "") if isinstance(qd, dict) else "")
            for qd in (qwen_dicts or [{}] * len(text_prompt))
        ]

        object_masks_for_gd = seg["object_masks"]
        object_boxes_for_gd = seg["object_boxes"]
        o_ref_extracted = self._extract_o_ref_via_gd(
            object_rgbd, source_phrases, camera_intrinsics,
            object_masks_list=object_masks_for_gd,
            object_boxes_list=object_boxes_for_gd,
            provided_mask=provided_object_mask,
        )
        o_ref_tokens_dense, o_ref_token_mask = self.token_builder(
            o_ref_extracted["dino_feats"],
            o_ref_extracted["pn_feats"],
            o_ref_extracted["spatial_feats"],
            scene_id=0,
        )
        B_ = object_rgbd.shape[0]
        D_ = self.token_builder.d_model
        if o_ref_tokens_dense.shape[1] == 0:
            o_ref_tokens = torch.zeros(B_, 1, D_, device=object_rgbd.device)
        else:
            o_ref_tokens = o_ref_tokens_dense
        o_ref_points   = o_ref_extracted["o_ref_points"]
        o_ref_centroid = o_ref_extracted["o_ref_centroid"]

        scene_tokens = self.scene_sa(scene_tokens, padding_mask=scene_padding_mask)

        text_padding_mask = (text_out["attention_mask"] == 0)
        proj_text_tokens = self.proj_t5(text_out["tokens"])
        grounded_t5_tokens = self.inst_ca(
            text_tokens=proj_text_tokens,
            object_tokens=scene_tokens,
            object_padding_mask=~scene_padding_mask,
        )

        B   = len(text_prompt)
        D   = scene_tokens.shape[-1]
        dev = scene_tokens.device

        d_t5 = t5_cls.shape[-1]
        per_slot_cls = t5_cls.unsqueeze(1).expand(B, MAX_ANCHORS, d_t5).clone()
        for b in range(B):
            for slot in range(MAX_ANCHORS):
                spans_b = relation_spans[b]
                if slot < len(spans_b) and spans_b[slot] is not None:
                    r_span = spans_b[slot]
                    if r_span[1] > r_span[0] and r_span[1] <= text_out["tokens"].shape[1]:
                        per_slot_cls[b, slot] = text_out["tokens"][b, r_span[0]:r_span[1]].mean(dim=0)

        slot_out_flat = self.relation_classifier(
            per_slot_cls.view(B * MAX_ANCHORS, d_t5)
        )
        per_slot_logits = slot_out_flat["logits"].view(B, MAX_ANCHORS, N_RELATION_CLASSES)
        per_slot_embs   = slot_out_flat["relation_emb"].view(B, MAX_ANCHORS, D)
        relation_out = {
            "logits":       per_slot_logits,
            "relation_emb": per_slot_embs,
            "pred_class":   slot_out_flat["pred_class"].view(B, MAX_ANCHORS),
        }

        anchor_spans_per_slot = [
            [[s[i + 1]] if len(s) > i + 1 else [] for s in phrase_spans]
            for i in range(MAX_ANCHORS)
        ]

        if provided_anchor_masks is not None:
            grounding_anchors = self._ground_anchors_from_masks(
                scene_masks, provided_anchor_masks, provided_anchor_valid, MAX_ANCHORS,
            )
        else:
            grounding_anchors = [
                self._ground_entity(
                    scene_rgbd, scene_boxes,
                    self._extract_phrase_tokens(text_out["tokens"], anchor_spans_per_slot[i]),
                    text_prompts=text_prompt, phrase_spans=anchor_spans_per_slot[i],
                )
                for i in range(MAX_ANCHORS)
            ]
        grounding_3d = grounding_anchors[0]
        grounding_3c = {
            "selected_obj_idx": [
                torch.tensor([0 if v else -1], dtype=torch.long, device=dev)
                for v in o_ref_extracted["o_ref_valid"].tolist()
            ],
            "o_ref_bbox": o_ref_extracted["o_ref_bbox"],
        }

        o_anchor_tokens    = torch.zeros(B, MAX_ANCHORS, D, device=dev)
        o_anchor_centroids = torch.zeros(B, MAX_ANCHORS, 3, device=dev)
        o_anchor_bboxes    = torch.zeros(B, MAX_ANCHORS, 6, device=dev)
        o_anchor_obb       = torch.zeros(B, MAX_ANCHORS, 15, device=dev)
        o_anchor_obb[..., 3:12] = torch.eye(3, device=dev).reshape(9)
        anchor_grounded    = torch.zeros(B, MAX_ANCHORS, dtype=torch.bool, device=dev)
        anchor_points_all: list[list[torch.Tensor | None]] = [
            [None] * MAX_ANCHORS for _ in range(B)
        ]

        scene_points_all: list[torch.Tensor] = []

        for b in range(B):
            for slot in range(MAX_ANCHORS):
                idx_list = grounding_anchors[slot]["selected_obj_idx"][b]
                idx = int(idx_list[0].item()) if (len(idx_list) > 0 and idx_list[0] >= 0) else -1

                if 0 <= idx < scene_tokens.shape[1]:
                    o_anchor_tokens[b, slot] = scene_tokens[b, idx]
                    anchor_grounded[b, slot] = True

                anchor_pts: torch.Tensor | None = None
                if provided_anchor_masks is not None:
                    m = provided_anchor_masks[b, slot].bool()
                    if bool(m.any()):
                        intr_b = camera_intrinsics[b] if camera_intrinsics is not None else None
                        anchor_pts = self._mask_to_metric_points(scene_depth[b], m, intr_b)

                if anchor_pts is not None and anchor_pts.shape[0] > 0:
                    anchor_grounded[b, slot] = True
                    o_anchor_centroids[b, slot] = anchor_pts.mean(dim=0)
                    o_anchor_bboxes[b, slot, :3] = anchor_pts.min(dim=0)[0]
                    o_anchor_bboxes[b, slot, 3:] = anchor_pts.max(dim=0)[0]
                    c_obb, R_obb, h_obb = anchor_obb(anchor_pts, extents_over_all=True)
                    o_anchor_obb[b, slot] = torch.cat([c_obb, R_obb.reshape(9), h_obb])
                    anchor_points_all[b][slot] = anchor_pts.detach()
                elif idx >= 0:
                    if idx < len(scene_centroids_list[b]):
                        o_anchor_centroids[b, slot] = scene_centroids_list[b][idx]
                    if idx < len(scene_points_list[b]):
                        pts = scene_points_list[b][idx]
                        if pts.shape[0] > 0:
                            o_anchor_bboxes[b, slot, :3] = pts.min(dim=0)[0]
                            o_anchor_bboxes[b, slot, 3:] = pts.max(dim=0)[0]
                            c_obb, R_obb, h_obb = anchor_obb(pts)
                            o_anchor_obb[b, slot] = torch.cat(
                                [c_obb, R_obb.reshape(9), h_obb]
                            )
                            anchor_points_all[b][slot] = pts.detach().view(-1, 3)

            if scene_points_list[b].shape[0] > 0:
                scene_points_all.append(scene_points_list[b].view(-1, 3))
            else:
                scene_points_all.append(torch.zeros(1, 3, device=dev))

        max_loss_pts = self._max_scene_loss_points
        if max_loss_pts > 0:
            for b in range(B):
                sp = scene_points_all[b]
                if sp.shape[0] > max_loss_pts:
                    idx = torch.randperm(sp.shape[0], device=sp.device)[:max_loss_pts]
                    scene_points_all[b] = sp[idx]

        max_scene_pts = max(sp.shape[0] for sp in scene_points_all)
        scene_points_pad = torch.zeros(B, max_scene_pts, 3, device=dev)
        for b, sp in enumerate(scene_points_all):
            scene_points_pad[b, :sp.shape[0]] = sp

        if gt_anchor_pos is not None and gt_anchor_valid is not None and provided_anchor_masks is None:
            for b in range(B):
                perm = _match_anchors_to_gt(
                    o_anchor_centroids[b], gt_anchor_pos[b],
                    anchor_grounded[b], gt_anchor_valid[b],
                )
                o_anchor_tokens[b]    = o_anchor_tokens[b][perm]
                o_anchor_centroids[b] = o_anchor_centroids[b][perm]
                o_anchor_bboxes[b]    = o_anchor_bboxes[b][perm]
                o_anchor_obb[b]       = o_anchor_obb[b][perm]
                anchor_grounded[b]    = anchor_grounded[b][perm]
                perm_list = perm.tolist()
                anchor_points_all[b]  = [anchor_points_all[b][p] for p in perm_list]

        if gt_anchor_valid is not None:
            anchor_valid = gt_anchor_valid
        else:
            target_present = torch.zeros(B, MAX_ANCHORS, dtype=torch.bool, device=dev)
            for b in range(B):
                qd = qwen_dicts[b] if qwen_dicts and b < len(qwen_dicts) else {}
                tgts = qd.get("targets", []) if isinstance(qd, dict) else []
                for slot in range(min(len(tgts), MAX_ANCHORS)):
                    if tgts[slot]: 
                        target_present[b, slot] = True
            anchor_valid = anchor_grounded & target_present

        relation_embs = torch.zeros(B, MAX_ANCHORS, D, device=dev)
        for slot in range(MAX_ANCHORS):
            for b in range(B):
                if anchor_valid[b, slot]:
                    if self.training and gt_relation_class is not None:
                        relation_embs[b, slot] = self.relation_classifier.embedding_table(
                            gt_relation_class[b, slot]
                        )
                    else:
                        hard_class = per_slot_logits[b, slot].argmax(dim=-1)
                        relation_embs[b, slot] = self.relation_classifier.embedding_table(hard_class)

        o_anchor_tokens = o_anchor_tokens + self.relation_proj(relation_embs)

        valid_f = anchor_valid.float().unsqueeze(-1)

        rel_proj = self.rel_cond_proj(per_slot_cls)
        rel_cond = (rel_proj * valid_f).sum(dim=1)

        anchor_padding_mask = ~anchor_valid

        o_anchor_centroid = o_anchor_centroids[:, 0]

        q_tokens = self.qformer(
            o_ref_tokens=o_ref_tokens,
            o_anchor_tokens=o_anchor_tokens,
            scene_tokens=scene_tokens,
            relation_emb=rel_cond,
            text_tokens=grounded_t5_tokens,
            scene_padding_mask=scene_padding_mask,
            text_padding_mask=text_padding_mask,
            anchor_padding_mask=anchor_padding_mask,
        )

        z = self.latent_bottleneck(q_tokens)

        placement = self.placement_head(z)
        
        transformed_o_ref = reconstruct_scene(
            o_ref_points=o_ref_points,
            delta_trans=placement["delta_trans"],
            rot_matrix=placement["rot_matrix"],
            o_ref_centroid=o_ref_centroid,
            o_anchor_centroid=o_anchor_centroid
        )

        if snap_to_surface and not self.training:
            with torch.no_grad():
                effective_rel = (
                    gt_relation_class
                    if gt_relation_class is not None
                    else per_slot_logits.argmax(dim=-1)
                )
                snap_down, snap_slot = ontop_snap_direction(
                    o_anchor_obb, anchor_valid, effective_rel
                )
                support_clouds: list[torch.Tensor] = []
                for b in range(B):
                    slot = int(snap_slot[b].item())
                    pts = anchor_points_all[b][slot] if slot >= 0 else None
                    if pts is not None and pts.shape[0] > 0:
                        support_clouds.append(pts.to(dev))
                    else:
                        sp = scene_points_pad[b]
                        support_clouds.append(sp[sp.abs().sum(-1) > 1e-6])
                max_sup = max(c.shape[0] for c in support_clouds)
                snap_support = torch.zeros(B, max_sup, 3, device=dev)
                for b, c in enumerate(support_clouds):
                    snap_support[b, :c.shape[0]] = c
                transformed_o_ref_snapped, snap_delta_y = snap_to_support_surface(
                    transformed_o_ref.detach(), snap_support, down=snap_down,
                )
        else:
            transformed_o_ref_snapped = transformed_o_ref.detach()
            snap_delta_y = torch.zeros(B, device=dev)
            snap_down = torch.zeros(B, 3, device=dev)
            snap_down[:, 1] = 1.0

        return {
            "seg": seg,
            "scene_object_tokens": scene_tokens,
            "scene_padding_mask": scene_padding_mask,
            "o_ref_tokens": o_ref_tokens,
            "o_ref_valid":  o_ref_extracted["o_ref_valid"],
            "o_ref_bbox":   o_ref_extracted["o_ref_bbox"],
            "o_ref_masks":  o_ref_extracted["o_ref_masks"],
            "text": text_out,
            "t5_cls": t5_cls,
            "phrase_spans": phrase_spans,
            "phrase_tokens": phrase_tokens_list,
            "qwen_dicts": qwen_dicts,
            "qwen_valid": qwen_valid,
            "grounding_3c": grounding_3c,
            "grounding_3d": grounding_3d,
            "grounding_anchors": grounding_anchors,
            "o_ref_centroid_extracted": o_ref_centroid,
            "scene_centroids":  scene_centroids_list,
            "relation_out": relation_out,
            "q_tokens": q_tokens,
            "action_latent": z,
            "placement": placement,
            "o_ref_points": o_ref_points,
            "transformed_o_ref": transformed_o_ref,
            "transformed_o_ref_snapped": transformed_o_ref_snapped,
            "snap_delta_y": snap_delta_y,
            "snap_down": snap_down,
            "scene_points": scene_points_pad,
            "o_anchor_centroid":  o_anchor_centroid,
            "o_anchor_centroids": o_anchor_centroids,
            "o_anchor_bboxes":    o_anchor_bboxes,
            "o_anchor_obb":       o_anchor_obb,
            "anchor_valid":       anchor_valid,
        }
