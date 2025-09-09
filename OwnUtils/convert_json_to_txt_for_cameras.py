import json
import numpy as np

# Input JSON file path
input_path = "/mnt/data/tijaz/dataSets/MipNeRF360/tandt_db/db/drjohnson/training/cameras.json"
output_path = "/mnt/data/tijaz/dataSets/MipNeRF360/tandt_db/db/drjohnson/sparse/0/cameras_final.txt"

# Load JSON data
with open(input_path, "r") as f:
    data = json.load(f)

with open(output_path, "w") as f:
    # Header for clarity (optional)
    f.write("# id img_name px py pz nx ny nz\n")

    for cam in data:
        cam_id = cam["id"]
        img_name = cam["img_name"]
        px, py, pz = cam["position"]

        # Convert rotation list to numpy array
        R_wc = np.array(cam["rotation"], dtype=float)  # world->camera rotation

        # Get camera->world rotation (transpose)
        R_cw = R_wc.T

        # Forward vector (camera's Z-axis in world coordinates)
        nx, ny, nz = R_cw[:, 2]  # 3rd column is forward direction

        f.write(f"{cam_id} {img_name} {px:.6f} {py:.6f} {pz:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")

print(f"Saved {len(data)} camera positions with directions to {output_path}")
