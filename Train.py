import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import build_iter_dataloader
from model import SceneTransformModel, PlacementLoss


_original_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _custom_torch_load


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_root = Path("dataset/flattened_dataset")
    if not data_root.exists():
        print(f"Data root {data_root} does not exist.")
        return

    augment_data = False 
    
    loader = build_iter_dataloader(
        root=data_root,
        batch_size=8,
        num_workers=0,
        shuffle=True,
        augment=augment_data,
        load_pointclouds=False,
    )
    print(f"Found {len(loader)} batches in dataloader.")

    ckpt_dir = Path("checkpoints")
    sam2_ckpt = None
    if ckpt_dir.exists():
        cands = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth"))
        if cands:
            sam2_ckpt = str(next((p for p in cands if "large" in p.name), cands[0]))
    
    if not sam2_ckpt:
        print("Warning: No SAM2 checkpoint found, using a dummy path (will require monkey-patching).")
        sam2_ckpt = "dummy_sam2.pt"

    _ckpt_name = Path(sam2_ckpt).stem
    _config_map = {
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "tiny":  "configs/sam2.1/sam2.1_hiera_t.yaml",
        "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    }
    sam2_cfg = next((v for k, v in _config_map.items() if k in _ckpt_name), "configs/sam2.1/sam2.1_hiera_l.yaml")

    cfg = {
        "sam2_checkpoint":  sam2_ckpt,
        "sam2_config":      sam2_cfg,
        "dino_model":       "dinov2_vitb14",
        "freeze_dino":      True,
        "pnet_out_dim":        256,
        "pnet_n_points":       256,
        "pnet_dense_n_points": 2048,
        "pnet_use_rgb":        True,
        "t5_model":         "t5-base",
        "t5_max_len":       77,
        "freeze_t5":        True,
        "d_model":          512,
        "use_cached_features": not augment_data,
    }

    print("Initializing model...")
    model = SceneTransformModel(cfg).to(device)

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    criterion = PlacementLoss().to(device)

    print(f"Starting training with {len(trainable_params)} trainable tensors...")

    start_epoch = 0
    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  → Resumed at epoch {start_epoch}")

    epochs = 100
    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(loader):
            scene_rgbd = batch["final_empty_rgbd"].to(device)
            object_rgbd = batch["initial_object_rgbd"].to(device)
            text_prompt = batch["text_prompt"]
            sample_id = batch["sample_id"]

            optimizer.zero_grad()

            camera_intrinsics = batch.get("camera_intrinsics")
            if camera_intrinsics is not None:
                camera_intrinsics = camera_intrinsics.to(device)

            gt_anchor_pos     = batch["gt_anchor_pos"].to(device)     
            gt_delta_trans    = batch["gt_delta_trans"].to(device)    
            gt_relation_class = batch["gt_relation_class"].to(device) 
            gt_anchor_valid   = batch["gt_anchor_valid"].to(device)   

            out = model(
                scene_rgbd,
                object_rgbd,
                text_prompt,
                cached_features=batch.get("cached_features"),
                camera_intrinsics=camera_intrinsics,
                gt_relation_class=gt_relation_class,
                gt_anchor_valid=gt_anchor_valid,
                gt_anchor_pos=gt_anchor_pos,
            )

            qwen_valid = out["qwen_valid"].to(gt_anchor_valid.device)
            gt_anchor_valid = gt_anchor_valid & qwen_valid.unsqueeze(1)

            loss_dict = criterion(
                pred_delta_trans=out["placement"]["delta_trans"],
                transformed_o_ref=out["transformed_o_ref"],
                pred_anchor_centroid=out["o_anchor_centroid"],
                relation_logits=out["relation_out"]["logits"],
                gt_anchor_pos=gt_anchor_pos,
                gt_delta_trans=gt_delta_trans,
                gt_anchor_valid=gt_anchor_valid,
                gt_relation_class=gt_relation_class,
                scene_points=out["scene_points"],
                o_anchor_bboxes=out["o_anchor_bboxes"],
                o_anchor_obb=out["o_anchor_obb"],
                rot_matrix=out["placement"]["rot_matrix"],
            )

            loss = loss_dict["total"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            parts = ' | '.join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if k != 'total')
            print(f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss.item():.4f} | {parts}")

        ckpt_dir.mkdir(exist_ok=True)
        save_path = ckpt_dir / f"latest.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"  → Checkpoint saved: {save_path}")
    ckpt_dir.mkdir(exist_ok=True)
    save_path = ckpt_dir / "latest.pt"
    print(f"Saving final model weights to {save_path}...")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print("Training complete.")

if __name__ == "__main__":
    main()
    