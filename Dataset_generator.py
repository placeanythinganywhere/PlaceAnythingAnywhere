import os
import sys
import json
import random
import numpy as np
import torch as th
import cv2
import matplotlib.pyplot as plt
import math
import traceback
from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.object_states import ToggledOn
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.on_top import OnTop
from omnigibson.utils.object_state_utils import sample_kinematics

# =========================================================================
# 1. TO THE LEFT
# =========================================================================
class ToTheLeft(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(OnTop) 
        return deps

    def _set_value(self, other, new_value, anchor=None, camera_pos=None, max_trials=25, use_trav_map=False):
        if not new_value: raise NotImplementedError("ToTheLeft does not support set_value(False)")
        if anchor is None: raise ValueError("ToTheLeft requires an 'anchor' object.")

        MIN_DIST = 0.0  
        MAX_DIST = 0.5     
        SPREAD_ANGLE = 30  
        tan_spread = math.tan(math.radians(SPREAD_ANGLE))

        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)
        else: camera_pos = camera_pos.clone().detach().float()

        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)
        else: ref_pos = ref_pos.clone().detach().float()

        state = og.sim.dump_state(serialized=False)

        V = ref_pos - camera_pos
        V[2] = 0.0  
        V = V / (th.norm(V) + 1e-6)  
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1)  
        Right = Right / (th.norm(Right) + 1e-6)
        Left = -Right
        Front = -V  

        self.obj.wake()

        for _ in range(max_trials):
            success = sample_kinematics("onTop", self.obj, anchor, use_trav_map=use_trav_map)
            if success and self.obj.states[OnTop].get_value(anchor):
                target_pos, _ = self.obj.get_position_orientation()
                if not th.is_tensor(target_pos): target_pos = th.tensor(target_pos, dtype=th.float32)
                else: target_pos = target_pos.clone().detach().float()
                
                D = target_pos - ref_pos
                D[2] = 0.0
                
                dot_L = th.dot(D, Left).item()
                dot_F = th.dot(D, Front).item()
                
                if dot_L > MIN_DIST and dot_L <= MAX_DIST and abs(dot_F) < (dot_L * tan_spread):
                    return True
            
            og.sim.load_state(state, serialized=False)
            og.sim.step_physics() 
        return False

    def _get_value(self, other, camera_pos=None):
        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)

        my_pos, _ = self.obj.get_position_orientation()
        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(my_pos): my_pos = th.tensor(my_pos, dtype=th.float32)
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Left = -Right
        Front = -V

        D = my_pos - ref_pos
        D[2] = 0.0
        dot_L = th.dot(D, Left).item()
        dot_F = th.dot(D, Front).item()
        
        return dot_L > 0.0 and dot_L <= 0.5 and abs(dot_F) < (dot_L * math.tan(math.radians(30)))

# =========================================================================
# 2. TO THE RIGHT
# =========================================================================
class ToTheRight(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(OnTop) 
        return deps

    def _set_value(self, other, new_value, anchor=None, camera_pos=None, max_trials=25, use_trav_map=False):
        if not new_value: raise NotImplementedError("ToTheRight does not support set_value(False)")
        if anchor is None: raise ValueError("ToTheRight requires an 'anchor' object.")

        MIN_DIST = 0.0
        MAX_DIST = 0.5
        SPREAD_ANGLE = 30
        tan_spread = math.tan(math.radians(SPREAD_ANGLE))

        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)
        else: camera_pos = camera_pos.clone().detach().float()

        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)
        else: ref_pos = ref_pos.clone().detach().float()

        state = og.sim.dump_state(serialized=False)

        V = ref_pos - camera_pos
        V[2] = 0.0  
        V = V / (th.norm(V) + 1e-6)  
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1)  
        Right = Right / (th.norm(Right) + 1e-6)
        Front = -V  

        self.obj.wake()

        for _ in range(max_trials):
            success = sample_kinematics("onTop", self.obj, anchor, use_trav_map=use_trav_map)
            if success and self.obj.states[OnTop].get_value(anchor):
                target_pos, _ = self.obj.get_position_orientation()
                if not th.is_tensor(target_pos): target_pos = th.tensor(target_pos, dtype=th.float32)
                else: target_pos = target_pos.clone().detach().float()
                
                D = target_pos - ref_pos
                D[2] = 0.0
                
                dot_R = th.dot(D, Right).item()
                dot_F = th.dot(D, Front).item()
                
                if dot_R > MIN_DIST and dot_R <= MAX_DIST and abs(dot_F) < (dot_R * tan_spread):
                    return True
            
            og.sim.load_state(state, serialized=False)
            og.sim.step_physics() 
        return False

    def _get_value(self, other, camera_pos=None):
        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)

        my_pos, _ = self.obj.get_position_orientation()
        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(my_pos): my_pos = th.tensor(my_pos, dtype=th.float32)
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Front = -V

        D = my_pos - ref_pos
        D[2] = 0.0
        dot_R = th.dot(D, Right).item()
        dot_F = th.dot(D, Front).item()
        
        return dot_R > 0.0 and dot_R <= 0.5 and abs(dot_F) < (dot_R * math.tan(math.radians(30)))

# =========================================================================
# 3. IN FRONT OF
# =========================================================================
class InFrontOf(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(OnTop)
        return deps

    def _set_value(self, other, new_value, anchor=None, camera_pos=None, max_trials=25, use_trav_map=False):
        if not new_value: raise NotImplementedError("InFrontOf does not support set_value(False)")
        if anchor is None: raise ValueError("InFrontOf requires an 'anchor' object.")

        MIN_DIST = 0.0
        MAX_DIST = 0.5
        SPREAD_ANGLE = 30
        tan_spread = math.tan(math.radians(SPREAD_ANGLE))

        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)
        else: camera_pos = camera_pos.clone().detach().float()

        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)
        else: ref_pos = ref_pos.clone().detach().float()

        state = og.sim.dump_state(serialized=False)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Front = -V  

        self.obj.wake()

        for _ in range(max_trials):
            success = sample_kinematics("onTop", self.obj, anchor, use_trav_map=use_trav_map)
            if success and self.obj.states[OnTop].get_value(anchor):
                target_pos, _ = self.obj.get_position_orientation()
                if not th.is_tensor(target_pos): target_pos = th.tensor(target_pos, dtype=th.float32)
                else: target_pos = target_pos.clone().detach().float()
                
                D = target_pos - ref_pos
                D[2] = 0.0
                
                dot_F = th.dot(D, Front).item()
                dot_R = th.dot(D, Right).item()
                
                if dot_F > MIN_DIST and dot_F <= MAX_DIST and abs(dot_R) < (dot_F * tan_spread):
                    return True
            
            og.sim.load_state(state, serialized=False)
            og.sim.step_physics() 
        return False

    def _get_value(self, other, camera_pos=None):
        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)

        my_pos, _ = self.obj.get_position_orientation()
        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(my_pos): my_pos = th.tensor(my_pos, dtype=th.float32)
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Front = -V

        D = my_pos - ref_pos
        D[2] = 0.0
        dot_F = th.dot(D, Front).item()
        dot_R = th.dot(D, Right).item()
        
        return dot_F > 0.0 and dot_F <= 0.5 and abs(dot_R) < (dot_F * math.tan(math.radians(30)))

# =========================================================================
# 4. BEHIND
# =========================================================================
class Behind(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(OnTop)
        return deps

    def _set_value(self, other, new_value, anchor=None, camera_pos=None, max_trials=25, use_trav_map=False):
        if not new_value: raise NotImplementedError("Behind does not support set_value(False)")
        if anchor is None: raise ValueError("Behind requires an 'anchor' object.")

        MIN_DIST = 0.0
        MAX_DIST = 0.5
        SPREAD_ANGLE = 30
        tan_spread = math.tan(math.radians(SPREAD_ANGLE))

        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)
        else: camera_pos = camera_pos.clone().detach().float()

        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)
        else: ref_pos = ref_pos.clone().detach().float()

        state = og.sim.dump_state(serialized=False)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Back = V  

        self.obj.wake()

        for _ in range(max_trials):
            success = sample_kinematics("onTop", self.obj, anchor, use_trav_map=use_trav_map)
            if success and self.obj.states[OnTop].get_value(anchor):
                target_pos, _ = self.obj.get_position_orientation()
                if not th.is_tensor(target_pos): target_pos = th.tensor(target_pos, dtype=th.float32)
                else: target_pos = target_pos.clone().detach().float()
                
                D = target_pos - ref_pos
                D[2] = 0.0
                
                dot_B = th.dot(D, Back).item()
                dot_R = th.dot(D, Right).item()
                
                if dot_B > MIN_DIST and dot_B <= MAX_DIST and abs(dot_R) < (dot_B * tan_spread):
                    return True
            
            og.sim.load_state(state, serialized=False)
            og.sim.step_physics()
        return False

    def _get_value(self, other, camera_pos=None):
        if camera_pos is None: camera_pos = og.sim.viewer_camera.get_position_orientation()[0]
        if not th.is_tensor(camera_pos): camera_pos = th.tensor(camera_pos, dtype=th.float32)

        my_pos, _ = self.obj.get_position_orientation()
        ref_pos, _ = other.get_position_orientation()
        if not th.is_tensor(my_pos): my_pos = th.tensor(my_pos, dtype=th.float32)
        if not th.is_tensor(ref_pos): ref_pos = th.tensor(ref_pos, dtype=th.float32)

        V = ref_pos - camera_pos
        V[2] = 0.0
        V = V / (th.norm(V) + 1e-6)
        
        Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
        Right = th.cross(V, Z, dim=-1) 
        Right = Right / (th.norm(Right) + 1e-6)
        Back = V

        D = my_pos - ref_pos
        D[2] = 0.0
        dot_B = th.dot(D, Back).item()
        dot_R = th.dot(D, Right).item()
        
        return dot_B > 0.0 and dot_B <= 0.5 and abs(dot_R) < (dot_B * math.tan(math.radians(30)))


# ============================================================
# LIGHTWEIGHT SIMULATOR CONFIG (VRAM FRIENDLY)
# ============================================================
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False     
gm.ENABLE_HQ_RENDERING = False   
gm.VISUALIZE_SYSTEM = False      
gm.ENABLE_FLATCACHE = True 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
os.makedirs(DATASET_PATH, exist_ok=True)

with open("assets/object_list.json", "r") as f:
    class_config = json.load(f)
MOVABLE_CATEGORIES = class_config["Object"]
random.shuffle(MOVABLE_CATEGORIES)

with open("assets/rs_int_cam_poses.json", "r") as f:
    pose_data = json.load(f)

# POSE_DICT = {item["id"]: item["poses"] for item in pose_data}

REC_DATA_DICT = {
    item["id"]: {
        "poses": item["poses"],
        "relations": item.get("relations", ["ontop"]) # Fallback to ontop if missing
    } 
    for item in pose_data
}

RELATIONS = ["ontop", "left", "right", "front", "back"]
REC_INIT_NAME = "breakfast_table_skczfi_0"
FLIP_CV_GL = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


# ============================================================
# UTILS
# ============================================================
def to_numpy(data):
    if th.is_tensor(data): return data.detach().cpu().numpy()
    return np.array(data)

def rotation_to_6d(rot_matrix):
    return rot_matrix[:, :2].T.flatten().tolist()

def hide_all_markers(scene):
    for obj in scene.objects:
        if ToggledOn in obj.states: obj.states[ToggledOn].visual_marker.visible = False

def get_mask_by_diff(cam, obj_to_hide, base_obs):
    if obj_to_hide is None: return None
    curr_visible = obj_to_hide.visible
    
    obj_to_hide.visible = False
    for _ in range(3): og.sim.render()
    obs_hidden = cam.get_obs()[0]
    
    obj_to_hide.visible = curr_visible
    for _ in range(3): og.sim.render()
    
    base_seg = to_numpy(base_obs["seg_instance"])
    hidden_seg = to_numpy(obs_hidden["seg_instance"])
    
    return base_seg != hidden_seg

def save_data_node(iter_dir, prefix, obs, masks_dict, cam, save_pcd=False):
    """
    Saves RGB, Depth, and Masks. 
    Calculates Point Cloud Centroids strictly IN MEMORY for hyper-fast execution.
    Only saves point_cloud.pcd to disk if save_pcd=True.
    """
    target_dir = os.path.join(iter_dir, prefix)
    os.makedirs(target_dir, exist_ok=True)
    
    rgb = to_numpy(obs["rgb"])[:, :, :3]
    cv2.imwrite(os.path.join(target_dir, "rgb.png"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    depth = to_numpy(obs["depth_linear"])
    np.save(os.path.join(target_dir, "depth.npy"), depth)
    # plt.imsave(os.path.join(target_dir, "depth_color.png"), np.clip(depth, 0, 8.0), cmap='viridis')
    
    # inst = to_numpy(obs["seg_instance"])
    # cv2.imwrite(os.path.join(target_dir, "seg_instance.png"), cv2.applyColorMap((inst % 256).astype(np.uint8), cv2.COLORMAP_JET))
    # sem = to_numpy(obs["seg_semantic"])
    # cv2.imwrite(os.path.join(target_dir, "seg_semantic.png"), cv2.applyColorMap((sem % 256).astype(np.uint8), cv2.COLORMAP_JET))

    # --- IN-MEMORY GEOMETRY UNPROJECTION ---
    K = to_numpy(cam.intrinsic_matrix)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    valid_mask = (depth > float(to_numpy(cam.clipping_range[0]))) & (depth < float(to_numpy(cam.clipping_range[1])))
    centroids = {}

    # Save physical scene Point Cloud ONLY if explicitly requested
    if save_pcd and np.any(valid_mask):
        rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        z_val = depth[valid_mask]
        points = np.stack(((u[valid_mask] - cx) * z_val / fx, (v[valid_mask] - cy) * z_val / fy, z_val), axis=-1)
        colors = rgb_uint8[valid_mask]
        rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
        rgb_float = rgb_packed.view(np.float32)
        
        with open(os.path.join(target_dir, "point_cloud.pcd"), "w") as f:
            f.write(f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
            np.savetxt(f, np.column_stack((points, rgb_float)), fmt="%.6f %.6f %.6f %g")

    # Fast in-memory centroid calculation (0 disk I/O)
    for name, m_bool in masks_dict.items():
        if m_bool is not None and np.any(m_bool):
            cv2.imwrite(os.path.join(target_dir, f"mask_{name}.png"), (m_bool * 255).astype(np.uint8))
            
            final_mask = valid_mask & m_bool
            if np.any(final_mask):
                z_val = depth[final_mask]
                points = np.stack(((u[final_mask] - cx) * z_val / fx, (v[final_mask] - cy) * z_val / fy, z_val), axis=-1)
                centroids[name] = np.mean(points, axis=0).tolist()
            else:
                centroids[name] = None
        else:
            centroids[name] = None
            
    return centroids

def get_cv_pose(obj, cam_pos, cam_quat):
    if obj is None: return None
    r_cam_inv = R.from_quat(to_numpy(cam_quat)).inv()
    p_cam = to_numpy(cam_pos)
    
    pos_w, orn_w = obj.get_position_orientation()
    pos_cam_gl = r_cam_inv.apply(to_numpy(pos_w) - p_cam)
    orn_cam_gl = (r_cam_inv * R.from_quat(to_numpy(orn_w))).as_matrix()
    
    pos_cv = FLIP_CV_GL @ pos_cam_gl
    R_cv = FLIP_CV_GL @ orn_cam_gl
    r_obj = R.from_matrix(R_cv)
    
    return {
        "pos_cam": pos_cv.tolist(),
        "orn_quat": r_obj.as_quat().tolist(),
        "orn_euler_xyz": r_obj.as_euler('xyz', degrees=False).tolist(),
        "orn_6d": rotation_to_6d(R_cv),
        "_R_mat": R_cv, 
        "_pos_cv": pos_cv 
    }

def clean_pose_for_json(pose_dict):
    """Safely extracts JSON-serializable keys without mutating the original dictionary dictating the math."""
    if pose_dict is None: return None
    return {
        "pos_cam": pose_dict["pos_cam"],
        "orn_quat": pose_dict["orn_quat"],
        "orn_euler_xyz": pose_dict["orn_euler_xyz"],
        "orn_6d": pose_dict["orn_6d"]
    }

def get_global_iteration(path):
    max_iter = -1
    if not os.path.exists(path): return 0
    for d in os.listdir(path):
        if "_iter_" in d:
            try:
                idx = int(d.split("_iter_")[1])
                if idx > max_iter: max_iter = idx
            except: pass
    return max_iter + 1

# ============================================================
# MAIN ENGINE
# ============================================================

def main():
    cfg = {"scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}}
    print("\n[INFO] Loading Environment Scene Map...")
    env = og.Environment(configs=cfg)
    scene = env.scene
    
    cam = og.sim.viewer_camera
    cam.image_height, cam.image_width = 512, 512
    cam.clipping_range = [0.05, 10.0]
    for mod in ["rgb", "depth_linear", "seg_instance", "seg_semantic"]: cam.add_modality(mod)
        
    og.sim.play()
    
    rec_init = next((obj for obj in scene.objects if obj.name == REC_INIT_NAME), None)
    if rec_init is None:
        return print(f"[ERROR] Critical Receptacle '{REC_INIT_NAME}' not found in the scene!")

    all_movable_models = []
    for cat in MOVABLE_CATEGORIES:
        for mod in og.utils.asset_utils.get_all_object_category_models(cat):
            all_movable_models.append((cat, mod))
            
    global_iteration = get_global_iteration(DATASET_PATH)

    # 1. SEQUENTIAL OBJECT LOOP (Obj0 & Obj1, Obj1 & Obj2, etc.)
    for obj_idx in range(len(all_movable_models) - 1):
        obj1_cat, obj1_model = all_movable_models[obj_idx]
        obj2_cat, obj2_model = all_movable_models[obj_idx + 1]

        # Add to scene
        obj1 = DatasetObject(name=f"obj1_{obj_idx}", category=obj1_cat, model=obj1_model)
        obj2 = DatasetObject(name=f"obj2_{obj_idx}", category=obj2_cat, model=obj2_model)
        
        og.sim.stop()
        scene.add_object(obj1)
        scene.add_object(obj2)
        og.sim.play()

        obj1.set_position_orientation(position=th.tensor([100.0, 100.0, -50.0]))
        obj2.set_position_orientation(position=th.tensor([100.0, 100.0, -55.0]))
        obj1.keep_still(); obj2.keep_still()
        for _ in range(5): og.sim.step() 

        # ==========================================
        # STAGE 1: INITIAL SCENE (Obj1 on Breakfast Table)
        # ==========================================
        if not obj1.states[object_states.OnTop].set_value(rec_init, True):
            og.sim.stop(); scene.remove_object(obj1); scene.remove_object(obj2); og.sim.play(); continue
            
        for _ in range(40): og.sim.step()
        hide_all_markers(scene)

        # init_pose_data = random.choice(POSE_DICT[REC_INIT_NAME])
        init_pose_data = random.choice(REC_DATA_DICT[REC_INIT_NAME]["poses"])
        c_pos_init = np.array(init_pose_data["translation"])
        c_quat_init = np.array(init_pose_data["rotation"])
        
        cam.set_position_orientation(position=c_pos_init, orientation=c_quat_init)
        for _ in range(15): og.sim.render()

        obs_init = cam.get_obs()[0]
        masks_init = {
            "obj1": get_mask_by_diff(cam, obj1, obs_init),
            "receptacle": get_mask_by_diff(cam, rec_init, obs_init)
        }
        
        if masks_init["obj1"] is None or not np.any(masks_init["obj1"]):
            print(f"    [!] Initial Object {obj1_model} out of view. Skipping pair.")
            og.sim.stop(); scene.remove_object(obj1); scene.remove_object(obj2); og.sim.play(); continue

        r_cam_init = R.from_quat(c_quat_init)
        r_obj_init_w = R.from_quat(to_numpy(obj1.get_position_orientation()[1]))
        r_relative_lock = r_cam_init.inv() * r_obj_init_w
        
        cv_obj1_init = get_cv_pose(obj1, c_pos_init, c_quat_init)
        cv_rec_init = get_cv_pose(rec_init, c_pos_init, c_quat_init)

        # ==========================================
        # 2. RECEPTACLE LOOP
        # ==========================================
        # for rec_final_name, poses in POSE_DICT.items():
        #     if rec_final_name == REC_INIT_NAME: continue 
        #     rec_final = next((obj for obj in scene.objects if obj.name == rec_final_name), None)
        #     if not rec_final: continue
        for rec_final_name, rec_data in REC_DATA_DICT.items():
            if rec_final_name == REC_INIT_NAME: continue 
            rec_final = next((obj for obj in scene.objects if obj.name == rec_final_name), None)
            if not rec_final: continue
            supported_relations = rec_data["relations"]
            # ==========================================
            # 3. CAMERA ANGLE LOOP
            # ==========================================
            # for cam_pose in poses:
            #     c_pos_final = np.array(cam_pose["translation"])
            #     c_quat_final = np.array(cam_pose["rotation"])
            for cam_pose in rec_data["poses"]:
                c_pos_final = np.array(cam_pose["translation"])
                c_quat_final = np.array(cam_pose["rotation"])
                
                setup_succeeded = False # Track if ANY relation succeeds for this camera angle

                # ==========================================
                # 4. RELATION LOOP
                # ==========================================
                for relation in supported_relations:
                    print(f"\n--- {relation.upper()} | Automated Gen Step: {global_iteration} ---")
                    ins = f"Place the {obj1_cat.replace('_',' ')} " + \
                          (f"on the {rec_final.category.replace('_',' ')}." if relation == "ontop" else \
                           f"to the {relation} of the {obj2_cat.replace('_',' ')} on the {rec_final.category.replace('_',' ')}.")
                    print(f">>>> [INSTRUCTION]: {ins} <<<<")

                    # Reset Table
                    obj1.set_position_orientation(position=th.tensor([100.0, 100.0, -50.0]))
                    obj2.set_position_orientation(position=th.tensor([100.0, 100.0, -55.0]))
                    obj1.enable_gravity()
                    obj2.enable_gravity()
                    obj1.keep_still(); obj2.keep_still()
                    for _ in range(10): og.sim.step()

                    if not obj2.states[object_states.OnTop].set_value(rec_final, True):
                        continue
                    for _ in range(40): og.sim.step()

                    cam.set_position_orientation(position=c_pos_final, orientation=c_quat_final)
                    
                    c_pos_tensor = th.tensor(c_pos_final, dtype=th.float32)
                    success = False

                    try:
                        if relation == "ontop":
                            success = obj1.states[object_states.OnTop].set_value(rec_final, True)
                        else:
                            if relation == "left": state = ToTheLeft(obj1)
                            elif relation == "right": state = ToTheRight(obj1)
                            elif relation == "front": state = InFrontOf(obj1)
                            elif relation == "back": state = Behind(obj1)
                            
                            state.initialize()
                            success = state.set_value(other=obj2, new_value=True, anchor=rec_final, camera_pos=c_pos_tensor, max_trials=25)
                    except Exception as e:
                        print("    [!] Mathematical Placement Failure")

                    if not success:
                        print(f"    [!] Failed to satisfy {relation}. Skipping...")
                        continue 

                    # Apply Camera-Relative Orientation Snap
                    r_obj_final_target = R.from_quat(c_quat_final) * r_relative_lock
                    obj1.set_position_orientation(
                        position=obj1.get_position_orientation()[0], 
                        orientation=r_obj_final_target.as_quat()
                    )
                    obj1.keep_still()
                    obj1.disable_gravity()  # Stop it from falling
                    if obj2:
                        obj2.keep_still()
                        obj2.disable_gravity()
                    for _ in range(50): og.sim.step() 
                    
                    for _ in range(15): og.sim.render()
                    obs_final_obj = cam.get_obs()[0]
                    masks_final_obj = {"obj1": get_mask_by_diff(cam, obj1, obs_final_obj)}
                    
                    if masks_final_obj["obj1"] is None or not np.any(masks_final_obj["obj1"]):
                        print("    [!] Object occluded or out of view after placement. Skipping...")
                        continue

                    # ==========================================
                    # DATA EXTRACTION & SAVING
                    # ==========================================
                    iter_path = os.path.join(DATASET_PATH, f"{relation}_iter_{global_iteration}")
                    os.makedirs(iter_path, exist_ok=True)

                    # Save Stage 1 (Initial) - NO PCD SAVED
                    centroids_init = save_data_node(iter_path, "initial_object_img", obs_init, masks_init, cam, save_pcd=False)
                    obj1_init_partial_centroid = centroids_init.get("obj1")

                    # Save Stage 2 (Placed) - NO PCD SAVED
                    centroids_final = save_data_node(iter_path, "final_object", obs_final_obj, masks_final_obj, cam, save_pcd=False)
                    obj1_final_partial_centroid = centroids_final.get("obj1")
                    cv_obj1_final = get_cv_pose(obj1, c_pos_final, c_quat_final)

                    # Save Stage 3 (Empty Scene) - PCD SAVED
                    obj1.set_position_orientation(position=th.tensor([100.0, 100.0, -50.0]))
                    obj1.keep_still()
                    for _ in range(10): og.sim.step() 
                    for _ in range(10): og.sim.render()
                    
                    obs_final_empty = cam.get_obs()[0]
                    masks_final_empty = {
                        "obj2": get_mask_by_diff(cam, obj2, obs_final_empty),
                        "receptacle": get_mask_by_diff(cam, rec_final, obs_final_empty)
                    }
                    centroids_empty = save_data_node(iter_path, "final_empty", obs_final_empty, masks_final_empty, cam, save_pcd=True)

                    rec_final_partial_centroid = centroids_empty.get("receptacle")
                    obj2_final_partial_centroid = centroids_empty.get("obj2")

                    cv_obj2_final = get_cv_pose(obj2, c_pos_final, c_quat_final)
                    cv_rec_final = get_cv_pose(rec_final, c_pos_final, c_quat_final)

                    # Metadata Math
                    R_delta_mat = cv_obj1_final["_R_mat"] @ np.linalg.inv(cv_obj1_init["_R_mat"])
                    T_delta = cv_obj1_final["_pos_cv"] - (R_delta_mat @ cv_obj1_init["_pos_cv"])
                    r_delta = R.from_matrix(R_delta_mat)

                    pos_obj1 = cv_obj1_final["_pos_cv"]
                    pos_rec = cv_rec_final["_pos_cv"]
                    delta_trans_obj1_to_rec = (pos_obj1 - pos_rec).tolist()
                    
                    delta_trans_obj1_to_obj2 = None
                    if cv_obj2_final is not None:
                        pos_obj2 = cv_obj2_final["_pos_cv"]
                        delta_trans_obj1_to_obj2 = (pos_obj1 - pos_obj2).tolist()

                    partial_delta_trans_obj1_to_rec = None
                    if obj1_final_partial_centroid and rec_final_partial_centroid:
                        partial_delta_trans_obj1_to_rec = (np.array(obj1_final_partial_centroid) - np.array(rec_final_partial_centroid)).tolist()

                    partial_delta_trans_obj1_to_obj2 = None
                    if obj1_final_partial_centroid and obj2_final_partial_centroid:
                        partial_delta_trans_obj1_to_obj2 = (np.array(obj1_final_partial_centroid) - np.array(obj2_final_partial_centroid)).tolist()

                    meta = {
                        "relation": relation,
                        "instruction": ins,
                        "entities": {
                            "obj1": {"category": obj1.category, "model": obj1.model},
                            "obj2": {"category": obj2.category, "model": obj2.model},
                            "receptacle_initial": {"category": rec_init.category, "model": rec_init.name},
                            "receptacle_final": {"category": rec_final.category, "model": rec_final.name}
                        },
                        "opencv_transformation": {
                            "note": "Camera-Relative OpenCV Frame (+X Right, +Y Down, +Z Forward)",
                            "delta_transform": {
                                "translation_delta": T_delta.tolist(),
                                "rotation_delta_quat": r_delta.as_quat().tolist(),
                                "rotation_delta_euler_xyz": r_delta.as_euler('xyz', degrees=False).tolist(),
                                "rotation_delta_6d": rotation_to_6d(R_delta_mat)
                            },
                            "relative_translation": {
                                "ground_truth_physical": {
                                    "obj1_to_receptacle": delta_trans_obj1_to_rec,
                                    "obj1_to_obj2": delta_trans_obj1_to_obj2
                                },
                                "observed_partial_pcd": {
                                    "obj1_to_receptacle": partial_delta_trans_obj1_to_rec,
                                    "obj1_to_obj2": partial_delta_trans_obj1_to_obj2
                                }
                            },
                            "partial_pcd_centroids": {
                                "obj1_initial": obj1_init_partial_centroid,
                                "obj1_final": obj1_final_partial_centroid,
                                "obj2_final": obj2_final_partial_centroid,
                                "receptacle_final": rec_final_partial_centroid
                            },
                            "obj1_initial_pose": clean_pose_for_json(cv_obj1_init),
                            "obj1_final_pose": clean_pose_for_json(cv_obj1_final),
                            "obj2_final_pose": clean_pose_for_json(cv_obj2_final),
                            "receptacle_final_pose": clean_pose_for_json(cv_rec_final)
                        }
                    }

                    with open(os.path.join(iter_path, "metadata.json"), "w") as f:
                        json.dump(meta, f, indent=4)
                        
                    print(f"✅ Generated {iter_path} successfully!")
                    setup_succeeded = True
                    
                # Iterate the counter only after completing all relations for a specific camera angle
                if setup_succeeded:
                    global_iteration += 1

        # End of Receptacle Iterations: Clean up this specific pair
        og.sim.stop()
        scene.remove_object(obj1)
        scene.remove_object(obj2)
        og.sim.play()

if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"Runtime Exception: {e}")
    finally: og.sim.stop()