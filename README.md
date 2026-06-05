# Place-Anything-Anywhere: A Lightweight Robot Agnostic Goal Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1000t586Wy45Fbm8ndL5rsZRFNNdYvjY5?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

While natural language has emerged as a powerful interface for robotic manipulation, translating open-vocabulary instructions into precise physical actions remains a persistent challenge. Recent trends heavily favor end-to-end Vision-Language-Action (VLA) systems that directly map visual and textual inputs to continuous robot trajectories. However, these architectures inherently entangle high-level semantic reasoning with low-level kinematic control, incur substantial computational overhead, and cannot be easily transferred to another robot platform without extensive fine-tuning.

In this work, we argue that for fundamental rigid-body manipulation—such as everyday pick-and-place and scene rearrangement—continuous robot control can be successfully replaced by simply establishing the accurate geometric goal pose of the object prior to execution. Once the target is predicted, safe and collision-aware execution can be reliably delegated to well-established motion planning algorithms.

This repository contains the official implementation of **Place-Anything-Anywhere (PAA)**, a lightweight framework designed specifically for language-conditioned object placement through anchor-relative pose estimation. Our system takes a source object RGB-D, a target RGB-D scene, and a natural language instruction, and outputs a precise spatial destination in the camera frame. By focusing strictly on multimodal geometric reasoning and goal generation, PAA provides a scalable, robot-agnostic interface that operates efficiently without massive parameter counts.

## 🚀 Features & Contributions
* **Lightweight End-to-End Goal Generation:** PAA aligns open-vocabulary instructions and scene structure to account for semantic and spatial relations while avoiding collisions, bypassing the heavy computational constraints of VLA models.
* **Anchor-Relative Continuous SE(3) Prediction:** Instead of generating pixel-level futures or absolute global coordinates, the model predicts a continuous, anchor-relative spatial transformation. This guarantees precise, collision-aware placements directly from partial camera observations.
* **Multi-Modal Scene and Object Encoding:** Treats object grounding and segmentation as modular front-end components. It uses a frozen DINOv2 backbone for visual semantics, PointNet++ for local geometry, and Grounding DINO/SAM2 for explicit object isolation.
* **Relational Q-Former Bottleneck:** Employs a two-stage cross-attention Q-Former that factorizes language into grounded sequence features, continuous relation conditioning, and discrete spatial embeddings, distilling multi-modal context into a compact 256-D action latent.
* **OmniGibson Synthetic Data Pipeline:** Features an automated, physics-aware synthetic generation pipeline built on the BEHAVIOR-1K assets and OmniGibson framework, allowing users to rapidly generate high-fidelity (Object, Scene, Future) training triplets.
* **Accessible & Ready to Use:** We release the complete code along with a Google Colab notebook for quick zero-shot testing with sample datasets and support for user-uploaded data.
---

## 📂 Repository Structure

* `Dataset_generator.py`: **Dataset Creation Pipeline.** Uses the OmniGibson simulator to generate high-fidelity (Initial Object, Initial Scene, Final Scene) triplets. It spawns objects, enforces spatial relations, teleports cameras, and extracts perfectly aligned RGB-D images and Point Clouds.
* `Model.py`: The core PyTorch architecture, including the multi-modal tokenizer, PointNet++ spatial encoder, and the cross-attention placement head.
* `Dataloader.py`: Custom PyTorch Dataset and DataLoader designed to parse the RGB-D tensors, instruction text, and ground truth $SE(3)$ transformations.
<!-- * `inference.py`: End-to-end evaluation script. Loads trained weights, runs the forward pass, decodes the 6D rotation back into a valid mathematical matrix, and animates the predicted placement using Open3D. -->
<!-- * `paper_classes.json` / `rs_int_cam_poses.json`: Config files containing OmniGibson scene settings, valid camera sweeps, and semantic object categorization. -->

---

## 🛠️ Installation

### Prerequisites
We highly recommend using an isolated Conda environment. Note that generating the dataset using `Dataset_generator.py` requires a system capable of running NVIDIA's **OmniGibson / Isaac Sim**. Model training and inference can be run on standard PyTorch-compatible GPUs.

### 1. Create a Conda Environment
```bash
conda create -n place-anything python=3.10 -y
conda activate place-anything
```

### 2. Install PyTorch
Install PyTorch compatible with your CUDA version (e.g., CUDA 11.8 or 12.1):

```bash
# Example for CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Core Dependencies
Install the required packages for model training and inference:

```bash
pip install numpy scipy opencv-python pillow matplotlib transformers trimesh open3d sam2
(Note: transformers is required for CLIP and T5 text encoders. open3d and matplotlib are used in inference.py for visualization).
```

### 4. Install OmniGibson (For Dataset Generation ONLY)
If you plan to generate your own dataset using Dataset_generator.py, you must install OmniGibson. Follow the Official OmniGibson Installation Guide.

## 💻 Usage
### 1. Try it in Google Colab
The fastest way to test inference and visualize the model's capabilities without setting up a local physics simulator is through our interactive Colab Notebook.

👉 Open the Colab Notebook

### 2. Generating the Dataset (Local)
To generate the physics-grounded synthetic dataset using Isaac Sim / OmniGibson:


```Bash
python Dataset_generator.py
```
This script iterates through the scenes defined in rs_int_cam_poses.json and the objects in paper_classes.json. It will save the RGB, Depth, extracted point clouds, and metadata.json files to your designated data directory.

```Bash
python Process_dataset.py
```
This script pre-processes the dataset and precomputes certain values and structures the dataset before training. 

### 3. Training the Model
(Assuming you have generated the data or downloaded our pre-computed dataset)
Update the DATA_ROOT or DATASET_PATH in data_loader.py/train.py and run your standard PyTorch training loop:

```Bash
python train.py
```
Make sure dataset paths are correct inside the train.py.


<!-- ### 4. Running Inference & Visualization
To test a trained checkpoint on a validation sample and visualize the predicted 3D placement against the ground truth:

```bash
python inference.py --checkpoint checkpoints/entity_world_model_ep10.pth --data_dir final_dataset --sample_idx 0
```
This will open an interactive Open3D window where:

Colored Points: The base scene geometry.
Green Box/Sphere: Ground Truth target placement.
Red Object / Blue Points: The model's predicted 3D placement based on the language prompt. -->

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
The synthetic dataset generation pipeline relies heavily on the OmniGibson simulator and the BEHAVIOR-1K asset library.
Visual and semantic embeddings utilize DINOv2, CLIP, and SAM2.
