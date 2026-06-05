import json
from pathlib import Path

dataset_root = Path("./dataset/flattened_dataset")
for subfolder in dataset_root.iterdir():

    if not subfolder.is_dir():
        continue

    json_files = list(subfolder.glob("*.json"))

    if len(json_files) == 0:
        print(f"[WARNING] No JSON found in {subfolder}")
        continue

    for json_path in json_files:

        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)

            relation = metadata.get("relation", "unknown")

            opencv_data = metadata["opencv_transformation"]

            obj1_final_pos = opencv_data["obj1_final_pose"]["pos_cam"]

            anchors = []

            obj2_entity = metadata.get("entities", {}).get("obj2")

            obj2_pose = opencv_data.get("obj2_final_pose")

            if obj2_entity is not None and obj2_pose is not None:

                anchors.append({
                    "anchor_type": "obj2_anchor",
                    "pos_cam": obj2_pose["pos_cam"],
                    "final_obj_pos_cam": obj1_final_pos,
                    "relation": relation,
                    "centroid": opencv_data.get("partial_pcd_centroids")["obj2_final"],
                    "delta":opencv_data.get("relative_translation")["observed_partial_pcd"]["obj1_to_obj2"],
                })

            receptacle_pose = opencv_data.get("receptacle_final_pose")

            if receptacle_pose is not None:

                anchors.append({
                    "anchor_type": "receptacle_anchor",
                    "pos_cam": receptacle_pose["pos_cam"],
                    "final_obj_pos_cam": obj1_final_pos,
                    "relation": "ontop",
                    "centroid": opencv_data.get("partial_pcd_centroids")["receptacle_final"],
                    "delta":opencv_data.get("relative_translation")["observed_partial_pcd"]["obj1_to_receptacle"],
                })

            metadata["anchors"] = anchors

            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"[UPDATED] {json_path}")

        except Exception as e:
            print(f"[ERROR] Failed processing {json_path}")
            print(e)