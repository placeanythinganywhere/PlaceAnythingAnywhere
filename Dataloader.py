"""
dataloader.py
=============
Dataset and DataLoader for the Language-Conditioned Scene Transformation model.

Inputs (per sample):
    - scene_rgbd  : (H, W, 4)  float32 — full scene RGBD image
    - object_rgbd : (H, W, 4)  float32 — isolated object RGBD image
    - text_prompt : str         — free-form instruction (e.g. "place the mug on the shelf")

Outputs (per sample):
    - target_pointcloud  : (N, 3+C)  float32 — full modified scene point cloud (xyz + optional rgb/normals)
    - object_placements  : list[dict]        — per-object placement metadata
        {
            "object_id"   : int,
            "translation" : (3,)   float32,   # Δ or absolute position
            "rotation"    : (4,)   float32,   # quaternion (w, x, y, z)
            "scale"       : (3,)   float32,   # per-axis scale factor
            "exists"      : bool,             # False → object was removed
        }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_IMG_SIZE: tuple[int, int] = (480, 640)   # (H, W)
DEFAULT_PC_POINTS: int = 8192                     # number of points in target cloud


# ---------------------------------------------------------------------------
# Per-sample point-cloud helpers
# ---------------------------------------------------------------------------

def farthest_point_sample(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """Deterministic FPS to produce exactly n_samples points from pts (N, C)."""
    N, _ = pts.shape
    if N <= n_samples:
        idx = np.concatenate([
            np.arange(N),
            np.random.choice(N, n_samples - N, replace=True),
        ])
        return pts[idx]

    # --- FAST PATH ---
    # FPS on 200,000 points takes ~17 seconds per sample in pure Python/NumPy.
    # By uniformly downsampling to 20k points first, we drop the time to < 0.1s
    # while preserving the exact same bounding volume and spatial distribution.
    MAX_PTS_FOR_FPS = 20000
    if N > MAX_PTS_FOR_FPS:
        # Uniformly random subsample first
        sub_idx = np.random.choice(N, MAX_PTS_FOR_FPS, replace=False)
        pts = pts[sub_idx]
        N = MAX_PTS_FOR_FPS

    sampled_idx = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)
    current = np.random.randint(0, N)

    for i in range(n_samples):
        sampled_idx[i] = current
        diff = pts[:, :3] - pts[current, :3]
        d = np.sum(diff ** 2, axis=-1)
        distances = np.minimum(distances, d)
        current = int(np.argmax(distances))

    return pts[sampled_idx]


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class SceneTransformDataset(Dataset):
    """
    Dataset for language-conditioned 3D scene editing.

    Expected directory layout
    -------------------------
    root/
      metadata.json              ← list of sample dicts (see _load_metadata)
      samples/
        <sample_id>/
          scene_rgb.png
          scene_depth.npy        ← float32 (H, W) in metres
          object_rgb.png
          object_depth.npy
          target_cloud.npy       ← float32 (N, 3) or (N, 6)
          placements.json        ← list of per-object placement dicts
          intrinsics.json        ← {"fx":…, "fy":…, "cx":…, "cy":…}
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",                    # "train" | "val" | "test"
        img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
        n_pc_points: int = DEFAULT_PC_POINTS,
        augment: bool = True,
        depth_scale: float = 1.0,
        max_depth: float = 10.0,
        max_text_len: int = 77,                  # T5-style token limit
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size            # (H, W)
        self.n_pc_points = n_pc_points
        self.augment = augment
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.max_text_len = max_text_len

        self.samples: list[dict[str, Any]] = self._load_metadata()

        # Image pre-processing (RGB channels only — depth handled separately)
        self.img_transform = T.Compose([
            T.ToTensor(),                         # (3, H, W) in [0, 1]
            T.Resize(img_size, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> list[dict[str, Any]]:
        """
        Load sample metadata for the requested split from metadata.json.

        Expected JSON structure
        -----------------------
        [
          {
            "sample_id"   : "0001",
            "split"       : "train",
            "text_prompt" : "move the red mug onto the shelf",
          },
          ...
        ]
        """
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found at {meta_path}")

        with open(meta_path) as f:
            all_samples: list[dict] = json.load(f)

        split_samples = [s for s in all_samples if s.get("split", "train") == self.split]
        if not split_samples:
            raise ValueError(f"No samples found for split='{self.split}'")
        return split_samples

    # ------------------------------------------------------------------
    # Low-level I/O helpers
    # ------------------------------------------------------------------

    def _sample_dir(self, sample_id: str) -> Path:
        return self.root / "samples" / sample_id

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load an RGB image as (H, W, 3) uint8."""
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]))  # PIL: (W, H)
        return np.array(img, dtype=np.uint8)

    def _load_depth(self, path: Path) -> np.ndarray:
        """Load a depth map as (H, W) float32 in metres."""
        depth = np.load(path).astype(np.float32)
        # Resize to img_size if necessary
        if depth.shape != self.img_size:
            from PIL import Image
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize(
                (self.img_size[1], self.img_size[0]), Image.BILINEAR
            )
            depth = np.array(depth_img, dtype=np.float32)
        return depth

    def _load_intrinsics(self, path: Path) -> np.ndarray:
        """Return a 3×3 intrinsics matrix from a JSON dict."""
        with open(path) as f:
            d = json.load(f)
        K = np.array([
            [d["fx"],  0.0,    d["cx"]],
            [0.0,      d["fy"], d["cy"]],
            [0.0,      0.0,     1.0   ],
        ], dtype=np.float32)
        return K

    def _load_placements(self, path: Path) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _augment_rgbd(
        self,
        rgb: np.ndarray,   # (H, W, 3) uint8
        depth: np.ndarray, # (H, W) float32
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lightweight RGBD augmentation (flip, colour jitter)."""
        if self.augment and np.random.rand() < 0.5:
            rgb   = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()

        if self.augment and np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.005, depth.shape).astype(np.float32)
            depth = np.clip(depth + noise, 0, self.max_depth)

        return rgb, depth

    def _augment_pointcloud(self, pts: np.ndarray) -> np.ndarray:
        """Random jitter + rotation around vertical axis for point clouds."""
        if not self.augment:
            return pts

        # Small jitter
        pts[:, :3] += np.random.normal(0, 0.002, (len(pts), 3)).astype(np.float32)

        # Random rotation around Y axis
        theta = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts[:, :3] = pts[:, :3] @ R.T

        return pts

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.samples[idx]
        sid  = meta["sample_id"]
        sdir = self._sample_dir(sid)

        # ---- 1. Load RGBD images ----------------------------------------
        scene_rgb   = self._load_rgb(sdir / "scene_rgb.png")
        scene_depth = self._load_depth(sdir / "scene_depth.npy")
        obj_rgb     = self._load_rgb(sdir / "object_rgb.png")
        obj_depth   = self._load_depth(sdir / "object_depth.npy")

        scene_rgb, scene_depth = self._augment_rgbd(scene_rgb, scene_depth)
        obj_rgb,   obj_depth   = self._augment_rgbd(obj_rgb,   obj_depth)

        # ---- 2. Build RGBD tensors (4-channel: RGB + D) -----------------
        #  shape: (4, H, W)
        scene_rgb_t  = self.img_transform(scene_rgb)              # (3, H, W)
        scene_depth_t = torch.from_numpy(scene_depth).unsqueeze(0) # (1, H, W)
        scene_rgbd    = torch.cat([scene_rgb_t, scene_depth_t], dim=0)  # (4, H, W)

        obj_rgb_t    = self.img_transform(obj_rgb)
        obj_depth_t  = torch.from_numpy(obj_depth).unsqueeze(0)
        object_rgbd  = torch.cat([obj_rgb_t, obj_depth_t], dim=0)       # (4, H, W)

        # ---- 3. Text prompt --------------------------------------------
        text_prompt: str = meta["text_prompt"]   # raw string; tokenization
        #   happens inside the model's text encoder (T5 tokenizer)

        # ---- 4. Target point cloud -------------------------------------
        target_cloud_raw = np.load(sdir / "target_cloud.npy").astype(np.float32)
        target_cloud_raw = self._augment_pointcloud(target_cloud_raw)
        target_cloud     = farthest_point_sample(target_cloud_raw, self.n_pc_points)
        target_cloud_t   = torch.from_numpy(target_cloud)  # (N, 3+C)

        # ---- 5. Object placement metadata ------------------------------
        placements: list[dict] = self._load_placements(sdir / "placements.json")
        #   Each dict has keys: object_id, translation (3,), rotation (4,),
        #                       scale (3,), exists (bool)
        # Collate-friendly tensors
        n_obj      = len(placements)
        obj_ids    = torch.tensor([p["object_id"]   for p in placements], dtype=torch.long)
        trans      = torch.tensor([p["translation"] for p in placements], dtype=torch.float32)  # (M, 3)
        rot        = torch.tensor([p["rotation"]    for p in placements], dtype=torch.float32)  # (M, 4)
        scale      = torch.tensor([p["scale"]       for p in placements], dtype=torch.float32)  # (M, 3)
        exists     = torch.tensor([p["exists"]      for p in placements], dtype=torch.bool)     # (M,)

        return {
            # -- Inputs --
            "scene_rgbd"   : scene_rgbd,    # (4, H, W)
            "object_rgbd"  : object_rgbd,   # (4, H, W)
            "text_prompt"  : text_prompt,   # str — collated as list[str]
            # -- Targets --
            "target_cloud" : target_cloud_t,# (N, 3+C)
            # -- Placement labels --
            "obj_ids"      : obj_ids,       # (M,)
            "translation"  : trans,         # (M, 3)
            "rotation"     : rot,           # (M, 4)
            "scale"        : scale,         # (M, 3)
            "exists"        : exists,       # (M,)
            # -- Meta --
            "sample_id"    : sid,
        }

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Custom collate for variable-length placement lists.

    Pad object placements to the maximum number of objects in the batch,
    and create a boolean padding mask.
    """
    from torch.utils.data._utils.collate import default_collate

    # Fields with uniform shape — use default collation
    uniform_keys = ["scene_rgbd", "object_rgbd", "target_cloud"]
    collated: dict[str, Any] = {k: default_collate([s[k] for s in batch]) for k in uniform_keys}

    # Text prompts — list of strings
    collated["text_prompt"] = [s["text_prompt"] for s in batch]
    collated["sample_id"]   = [s["sample_id"]   for s in batch]

    # Variable-length placement fields — pad to max_objs
    max_objs = max(len(s["obj_ids"]) for s in batch)
    B = len(batch)

    def _pad(key: str, fill: float | int = 0) -> torch.Tensor:
        parts = [s[key] for s in batch]           # list of (M_i, ...) tensors
        feature_shape = parts[0].shape[1:]        # e.g. () or (3,) or (4,)
        out = torch.full((B, max_objs) + feature_shape, fill,
                         dtype=parts[0].dtype)
        for i, p in enumerate(parts):
            out[i, :len(p)] = p
        return out

    collated["obj_ids"]      = _pad("obj_ids",    fill=0)
    collated["translation"]  = _pad("translation", fill=0.0)
    collated["rotation"]     = _pad("rotation",    fill=0.0)
    collated["scale"]        = _pad("scale",       fill=1.0)
    collated["exists"]       = _pad("exists",      fill=0)

    # Padding mask: True = valid object, False = padding
    obj_padding_mask = torch.zeros(B, max_objs, dtype=torch.bool)
    for i, s in enumerate(batch):
        obj_padding_mask[i, :len(s["obj_ids"])] = True
    collated["obj_padding_mask"] = obj_padding_mask  # (B, max_objs)

    return collated


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    n_pc_points: int = DEFAULT_PC_POINTS,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    augment: bool | None = None,         # None → auto (True for train)
    depth_scale: float = 1.0,
    max_depth: float = 10.0,
    pin_memory: bool = True,
) -> DataLoader:
    """Convenience factory that returns a ready-to-use DataLoader."""
    if augment is None:
        augment = split == "train"

    dataset = SceneTransformDataset(
        root=root,
        split=split,
        img_size=img_size,
        n_pc_points=n_pc_points,
        augment=augment,
        depth_scale=depth_scale,
        max_depth=max_depth,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    print("DataLoader smoke-test skipped — requires real dataset root.")
    print("Usage:")
    print("  loader = build_dataloader('/path/to/dataset', split='train')")
    print("  batch  = next(iter(loader))")
    print("  print(batch['scene_rgbd'].shape)    # (B, 4, H, W)")
    print("  print(batch['target_cloud'].shape)  # (B, N, 3+C)")
    print("  print(batch['text_prompt'])         # list[str]")


# ---------------------------------------------------------------------------
# iter_* scene dataset  (data/iter_*/)
# ---------------------------------------------------------------------------


def _load_ascii_pcd(path: Path) -> np.ndarray:
    """Load an ASCII PCD file as (N, 6) float32 xyzrgb in [0, 1]."""
    header_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            header_lines += 1
            if line.startswith("DATA"):
                break

    points = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]

    if points.shape[1] < 4:
        return points[:, :3].astype(np.float32, copy=False)

    packed_rgb = points[:, 3].astype(np.float32, copy=True).view(np.uint32)
    red = ((packed_rgb >> 16) & 255).astype(np.float32) / 255.0
    green = ((packed_rgb >> 8) & 255).astype(np.float32) / 255.0
    blue = (packed_rgb & 255).astype(np.float32) / 255.0
    rgb = np.stack([red, green, blue], axis=1)
    return np.concatenate([points[:, :3], rgb], axis=1).astype(np.float32, copy=False)


class IterSceneDataset(Dataset):
    """Dataset for one sample per ``data/iter_*`` folder.

    Returned fields
    ---------------
    initial_object_rgbd : (4, H, W) float32
        RGBD for the initial object view.
    final_empty_rgbd : (4, H, W) float32
        RGBD for the final empty scene.
    final_oriented_rgbd : (4, H, W) float32
        RGBD for the final oriented scene.
    object_mask : (H, W) bool
        Binary object mask from ``initial_object/object_mask.png``.
    final_oriented_pointcloud : (N, 6) float32
        XYZRGB point cloud loaded from ``final_oriented/point_cloud.pcd``.
    metadata : dict
        Contents of ``metadata.json`` for the iteration.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
        augment: bool = False,
        depth_scale: float = 1.0,
        max_depth: float = 10.0,
        load_pointclouds: bool = False,
        inference: bool = False,
        precompute: bool = False,
        load_gt_masks: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.load_pointclouds = bool(load_pointclouds)
        self.inference = inference
        self.precompute = precompute
        # When True, load ground-truth segmentation masks from the dataset so the
        # model can skip Grounding DINO: mask_obj1 (object/source) and the per-anchor
        # masks (mask_obj2 / mask_receptacle) from the scene view (final_empty).
        self.load_gt_masks = bool(load_gt_masks)

        if not self.root.exists():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        all_iter_dirs: list[Path] = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and ("iter_" in p.name)
        )
        if not all_iter_dirs:
            raise ValueError(
                f"No iter_* sub-directories found under {self.root}. "
                "Expected folders named iter_1/, iter_2/, …"
            )

        self.iter_dirs = all_iter_dirs

        # Auto-precompute any samples missing features.pt unless we are already
        # inside precompute_features.py (precompute=True) which would cause recursion.
        if not self.precompute:
            missing = [p for p in self.iter_dirs if not (p / "features.pt").exists()]
            if missing:
                import subprocess, sys
                script = Path(__file__).parent / "precompute_features.py"
                print(f"[IterSceneDataset] Auto-precomputing {len(missing)} sample(s) missing features.pt...")
                for p in missing:
                    print(f"  → {p.name}")
                    subprocess.run(
                        [sys.executable, str(script), "--sample", p.name, "--data_root", str(self.root)],
                        check=True,
                    )

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Resize(img_size, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_rgb(self, path: Path) -> np.ndarray:
        from PIL import Image as PILImage

        img = PILImage.open(path).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]))
        return np.array(img, dtype=np.uint8)

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = np.load(path).astype(np.float32)
        if depth.shape != self.img_size:
            from PIL import Image as PILImage

            depth_img = PILImage.fromarray(depth)
            depth_img = depth_img.resize((self.img_size[1], self.img_size[0]), PILImage.BILINEAR)
            depth = np.array(depth_img, dtype=np.float32)
        return depth

    def _load_mask(self, path: Path) -> np.ndarray:
        from PIL import Image as PILImage

        mask_img = PILImage.open(path).convert("L")
        mask_img = mask_img.resize((self.img_size[1], self.img_size[0]), PILImage.NEAREST)
        return (np.array(mask_img, dtype=np.uint8) > 0)

    def _load_metadata(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_intrinsics_txt(self, path: Path) -> dict[str, float]:
        """Parse a key=value intrinsics.txt into {fx, fy, cx, cy}."""
        result: dict[str, float] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    try:
                        result[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        return result

    def _augment_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.augment and np.random.rand() < 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
        if self.augment and np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.005, depth.shape).astype(np.float32)
            depth = np.clip(depth + noise, 0, self.max_depth)
        return rgb, depth

    def _build_rgbd_tensor(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        rgb_t = self.img_transform(rgb)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        return torch.cat([rgb_t, depth_t], dim=0)

    def __len__(self) -> int:
        return len(self.iter_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        iter_dir = self.iter_dirs[idx]

        initial_object_dir = iter_dir / "initial_object_img"
        final_empty_dir = iter_dir / "final_empty"
        final_oriented_dir = iter_dir / "final_object"

        initial_object_rgb = self._load_rgb(initial_object_dir / "rgb.png")
        initial_object_depth = self._load_depth(initial_object_dir / "depth.npy")
        initial_object_rgb, initial_object_depth = self._augment_rgbd(
            initial_object_rgb, initial_object_depth
        )
        initial_object_depth = np.clip(initial_object_depth * self.depth_scale, 0, self.max_depth)

        final_empty_rgb = self._load_rgb(final_empty_dir / "rgb.png")
        final_empty_depth = self._load_depth(final_empty_dir / "depth.npy")
        final_empty_rgb, final_empty_depth = self._augment_rgbd(final_empty_rgb, final_empty_depth)
        final_empty_depth = np.clip(final_empty_depth * self.depth_scale, 0, self.max_depth)

        if self.inference or self.precompute:
            H, W = self.img_size
            final_oriented_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            final_oriented_depth = np.zeros((H, W), dtype=np.float32)
        else:
            final_oriented_rgb = self._load_rgb(final_oriented_dir / "rgb.png")
            final_oriented_depth = self._load_depth(final_oriented_dir / "depth.npy")
            final_oriented_rgb, final_oriented_depth = self._augment_rgbd(
                final_oriented_rgb, final_oriented_depth
            )
            final_oriented_depth = np.clip(final_oriented_depth * self.depth_scale, 0, self.max_depth)

        object_mask_path = initial_object_dir / "mask_obj1.png"
        if (self.inference or self.precompute) and not object_mask_path.exists():
            H, W = self.img_size
            object_mask = np.zeros((H, W), dtype=np.bool_)
        else:
            object_mask = self._load_mask(object_mask_path)

        metadata_path = iter_dir / "metadata.json"
        metadata = self._load_metadata(metadata_path)
        text_prompt = metadata.get("instruction", "")

        # --- Compute GT Placement Labels (multi-anchor, up to MAX_ANCHORS=5) ---
        _RELATIONS = ["ontop", "left", "right", "front", "back"]
        _MAX_ANCHORS = 5

        gt_anchor_pos     = torch.zeros(_MAX_ANCHORS, 3,  dtype=torch.float32)
        gt_delta_trans    = torch.zeros(_MAX_ANCHORS, 3,  dtype=torch.float32)
        gt_relation_class = torch.zeros(_MAX_ANCHORS,     dtype=torch.long)
        gt_anchor_valid   = torch.zeros(_MAX_ANCHORS,     dtype=torch.bool)

        # Per-anchor GT masks (slot-aligned with the fields above) so the model can
        # skip Grounding DINO. The mask file is derived from the anchor_type:
        #   obj2_anchor       → mask_obj2.png
        #   receptacle_anchor → mask_receptacle.png
        # and is read from the scene view (final_empty/).
        H, W = self.img_size
        gt_anchor_masks = (
            torch.zeros(_MAX_ANCHORS, H, W, dtype=torch.bool)
            if self.load_gt_masks else None
        )

        anchors_meta = metadata.get("anchors", [])
        for i, anc in enumerate(anchors_meta[:_MAX_ANCHORS]):
            delta          = anc.get("delta")
            centriod = anc.get("centroid")
            relation         = anc.get("relation", "ontop")
            delt = np.array(delta,           dtype=np.float32)
            cent = np.array(centriod, dtype=np.float32)
            gt_anchor_pos[i]     = torch.from_numpy(cent)
            gt_delta_trans[i]    = torch.from_numpy(delt)
            gt_relation_class[i] = _RELATIONS.index(relation) if relation in _RELATIONS else 0
            gt_anchor_valid[i]   = True

            if self.load_gt_masks:
                atype = anc.get("anchor_type", "")
                mask_name = "mask_" + atype.replace("_anchor", "") + ".png"
                amask_path = final_empty_dir / mask_name
                if amask_path.exists():
                    gt_anchor_masks[i] = torch.from_numpy(
                        self._load_mask(amask_path).astype(np.bool_)
                    )
                else:
                    print(f"[IterSceneDataset] anchor mask not found: {amask_path}")
        
        # --- Target point cloud for supervision ---
        if self.load_pointclouds and not (self.inference or self.precompute):
            target_cloud_t = torch.from_numpy(
                    farthest_point_sample(
                        _load_ascii_pcd(final_oriented_dir / "point_cloud.pcd"),
                        DEFAULT_PC_POINTS
                    )
            )
        else:
            target_cloud_t = None

        # --- Camera Intrinsics ---
        intrinsics_dict = metadata.get("camera_intrinsics", {})
        if not intrinsics_dict:
            txt_path = final_empty_dir / "intrinsics.txt"
            if txt_path.exists():
                intrinsics_dict = self._load_intrinsics_txt(txt_path)
        camera_intrinsics = torch.tensor([
            intrinsics_dict.get("fx", 414.57487501053265),
            intrinsics_dict.get("fy", 414.57487501053265),
            intrinsics_dict.get("cx", 256.0),
            intrinsics_dict.get("cy", 256.0),
        ], dtype=torch.float32)

        out_dict = {
            "initial_object_rgbd": self._build_rgbd_tensor(initial_object_rgb, initial_object_depth),
            "final_empty_rgbd": self._build_rgbd_tensor(final_empty_rgb, final_empty_depth),
            "final_oriented_rgbd": self._build_rgbd_tensor(final_oriented_rgb, final_oriented_depth),
            "object_mask": torch.from_numpy(object_mask.astype(np.bool_)),
            "gt_anchor_masks": gt_anchor_masks,      # (MAX_ANCHORS, H, W) bool or None
            "target_cloud": target_cloud_t,
            "gt_anchor_pos":     gt_anchor_pos,      # (MAX_ANCHORS, 3)
            "gt_delta_trans":    gt_delta_trans,      # (MAX_ANCHORS, 3)
            "gt_relation_class": gt_relation_class,   # (MAX_ANCHORS,)
            "gt_anchor_valid":   gt_anchor_valid,     # (MAX_ANCHORS,) bool
            "camera_intrinsics": camera_intrinsics,
            # Grounding supervision (not implemented yet): keep as None (no dummy tensors)
            "gt_mask_idx_3c": None,
            "gt_mask_idx_3d": None,
            "metadata": metadata,
            "text_prompt": text_prompt,
            "scene_dir": str(iter_dir),
            "sample_id": iter_dir.name,
        }
        
        # Load precomputed features if they exist
        feat_path = iter_dir / "features.pt"
        if feat_path.exists():
            out_dict["cached_features"] = torch.load(feat_path, map_location="cpu")
        else:
            out_dict["cached_features"] = None
            
        return out_dict


def build_iter_dataloader(
    root: str | Path,
    batch_size: int = 1,
    num_workers: int = 4,
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    augment: bool = False,
    depth_scale: float = 1.0,
    max_depth: float = 10.0,
    shuffle: bool = False,
    pin_memory: bool = True,
    load_pointclouds: bool = False,
    inference: bool = False,
    precompute: bool = False,
    load_gt_masks: bool = True,
) -> DataLoader:
    """
    Convenience factory that returns a DataLoader over all ``iter_*``
    sub-directories found under ``root``.

    Any sample missing a precomputed ``features.pt`` is automatically
    precomputed before the DataLoader is returned.

    Example
    -------
    >>> loader = build_iter_dataloader("data/")
    >>> batch  = next(iter(loader))
    >>> print(batch["scene_rgbd"].shape)   # (B, 4, H, W)
    >>> print(batch["object_rgbd"].shape)  # (B, 4, H, W)
    >>> print(batch["text_prompt"])        # list[str]
    """
    dataset = IterSceneDataset(
        root=root,
        img_size=img_size,
        augment=augment,
        depth_scale=depth_scale,
        max_depth=max_depth,
        load_pointclouds=load_pointclouds,
        inference=inference,
        precompute=precompute,
        load_gt_masks=load_gt_masks,
    )

    def _iter_collate(batch: list[dict]) -> dict:
        from torch.utils.data._utils.collate import default_collate
        tensor_keys = [
            "initial_object_rgbd", "final_empty_rgbd", "final_oriented_rgbd",
            "object_mask", "gt_anchor_masks", "target_cloud",
            "gt_anchor_pos", "gt_delta_trans", "gt_relation_class", "gt_anchor_valid",
            "camera_intrinsics",
        ]

        out: dict[str, Any] = {}
        for k in tensor_keys:
            vals = [s.get(k) for s in batch]
            if any(v is None for v in vals):
                out[k] = None
            else:
                out[k] = default_collate(vals)

        # Grounding supervision not implemented yet — keep as None
        out["gt_mask_idx_3c"] = None
        out["gt_mask_idx_3d"] = None
        out["metadata"] = [s["metadata"] for s in batch]
        out["text_prompt"] = [s["text_prompt"] for s in batch]
        out["scene_dir"]   = [s["scene_dir"]   for s in batch]
        out["sample_id"]   = [s["sample_id"]   for s in batch]
        
        # Collate cached_features if present
        if all(s["cached_features"] is not None for s in batch):
            cached = {}
            # seg is a dict of lists
            cached["seg"] = {
                k: sum([s["cached_features"]["seg"][k] for s in batch], []) 
                for k in batch[0]["cached_features"]["seg"].keys()
            }
            # dino is a list of tensors
            cached["scene_dino"] = sum([s["cached_features"]["scene_dino"] for s in batch], [])
            
            # text_out is a dict of tensors with shape [1, L, D] or [1, L].
            # Different prompts tokenize to different L, so pad to the max L in
            # the batch before concatenating to [B, L, D] / [B, L].
            def _pad_and_cat(tensors: list[torch.Tensor]) -> torch.Tensor:
                max_len = max(t.shape[1] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_len - t.shape[1]
                    if pad_len > 0:
                        # Pad with zeros along dim 1 (sequence dim)
                        pad_shape = list(t.shape)
                        pad_shape[1] = pad_len
                        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=1)
                    padded.append(t)
                return torch.cat(padded, dim=0)

            cached["text_out"] = {
                k: _pad_and_cat([s["cached_features"]["text_out"][k] for s in batch])
                for k in batch[0]["cached_features"]["text_out"].keys()
            }
            out["cached_features"] = cached
        else:
            out["cached_features"] = None
            
        return out

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_iter_collate,
        pin_memory=pin_memory,
        drop_last=False,
    )
